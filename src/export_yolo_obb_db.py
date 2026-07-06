import argparse
import zipfile
from pathlib import Path

from sqlalchemy.orm import Session

from src.db.database import SessionLocal
from src.db.models import Annotation, AnnotationBox, Frame, Prediction, QueueItem, Review
from src.video_loader import load_config


EXPORTABLE_REVIEWS = {
    "correct_detection",
    "correct_no_detection",
    "wrong_box",
    "box_needs_split",
    "real_missed_detection",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export reviewed PostgreSQL frames as a CVAT-style YOLO OBB dataset."
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Export folder. Default: <exports_dir>/yolo_obb_export",
    )
    parser.add_argument(
        "--order-code",
        default="all",
        help="Export one order_code or all.",
    )
    parser.add_argument(
        "--camera",
        default="all",
        help="Export one camera or all.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Create empty label files for reviewed correct_no_detection frames.",
    )
    parser.add_argument(
        "--include-unreviewed-high-confidence",
        action="store_true",
        help="Also export skipped high_confidence frames using model boxes.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create result.zip in the export folder.",
    )

    return parser.parse_args()


def normalize_points(points: list[list[float]], width: int | None, height: int | None) -> list[list[float]]:
    normalized = []

    for point in points:
        x = float(point[0])
        y = float(point[1])
        nx = x / width if width else 0.0
        ny = y / height if height else 0.0
        normalized.append(
            [
                max(0.0, min(1.0, nx)),
                max(0.0, min(1.0, ny)),
            ]
        )

    return normalized


def get_latest_reviews(db: Session, queue_item_ids: list[int]) -> dict[int, Review]:
    if not queue_item_ids:
        return {}

    reviews = (
        db.query(Review)
        .filter(Review.queue_item_id.in_(queue_item_ids))
        .order_by(Review.queue_item_id, Review.id.desc())
        .all()
    )

    latest = {}

    for review in reviews:
        if review.queue_item_id not in latest:
            latest[review.queue_item_id] = review

    return latest


def get_manual_boxes(db: Session, frame_ids: list[int]) -> dict[int, list[dict]]:
    if not frame_ids:
        return {}

    rows = (
        db.query(Annotation, AnnotationBox)
        .join(AnnotationBox, AnnotationBox.annotation_id == Annotation.id)
        .filter(Annotation.frame_id.in_(frame_ids))
        .filter(Annotation.annotation_source == "manual_draw")
        .filter(Annotation.status == "saved")
        .order_by(Annotation.frame_id, AnnotationBox.id)
        .all()
    )

    boxes_by_frame = {}

    for annotation, box in rows:
        boxes_by_frame.setdefault(annotation.frame_id, []).append(
            {
                "class_id": box.class_id or 0,
                "points": box.points_json or [],
                "normalized_points": box.normalized_points_json,
                "source": box.box_source,
            }
        )

    return boxes_by_frame


def get_saved_manual_annotation_frame_ids(db: Session, frame_ids: list[int]) -> set[int]:
    if not frame_ids:
        return set()

    return {
        row[0]
        for row in (
            db.query(Annotation.frame_id)
            .filter(Annotation.frame_id.in_(frame_ids))
            .filter(Annotation.annotation_source == "manual_draw")
            .filter(Annotation.status == "saved")
            .distinct()
            .all()
        )
    }


def get_latest_queue_items_by_frame(db: Session, frame_ids: list[int]) -> dict[int, QueueItem]:
    if not frame_ids:
        return {}

    queue_items = (
        db.query(QueueItem)
        .filter(QueueItem.frame_id.in_(frame_ids))
        .order_by(QueueItem.frame_id, QueueItem.id.desc())
        .all()
    )

    latest = {}

    for queue_item in queue_items:
        if queue_item.frame_id not in latest:
            latest[queue_item.frame_id] = queue_item

    return latest


def prediction_boxes(prediction: Prediction | None, frame: Frame) -> list[dict]:
    if not prediction:
        return []

    boxes = []

    for box in prediction.boxes_json or []:
        points = box.get("points") or []

        if len(points) != 4:
            continue

        boxes.append(
            {
                "class_id": 0,
                "points": points,
                "normalized_points": box.get("normalized_points")
                or normalize_points(points, frame.width, frame.height),
                "source": "model",
            }
        )

    return boxes


def should_export(
    queue_item: QueueItem | None,
    review: Review | None,
    boxes: list[dict],
    has_manual_annotation: bool,
    include_empty_labels: bool,
    include_unreviewed_high_confidence: bool,
) -> tuple[bool, str]:
    if has_manual_annotation:
        return True, "manual_annotation"

    if boxes:
        return True, "model_boxes"

    if review:
        if review.review_status == "ignore_frame":
            return False, "ignored"

        if review.review_status == "correct_no_detection":
            return include_empty_labels, "correct_no_detection"

        if review.review_status == "correct_detection":
            return True, review.review_status

        if review.review_status in EXPORTABLE_REVIEWS:
            return False, "needs_manual_annotation"

        return False, f"review_{review.review_status}"

    if (
        include_unreviewed_high_confidence
        and queue_item
        and queue_item.queue_reason == "high_confidence"
        and queue_item.status == "skipped"
    ):
        return True, "unreviewed_high_confidence"

    return False, "not_reviewed"


def label_lines_for_boxes(boxes: list[dict], frame: Frame) -> list[str]:
    lines = []

    for box in boxes:
        points = box.get("points") or []

        if len(points) != 4:
            continue

        normalized_points = box.get("normalized_points") or normalize_points(
            points,
            frame.width,
            frame.height,
        )

        values = ["0"]

        for point in normalized_points:
            values.append(f"{float(point[0]):.6f}")
            values.append(f"{float(point[1]):.6f}")

        lines.append(" ".join(values))

    return lines


def write_data_yaml(output_dir: Path) -> None:
    content = "names:\n  0: product\npath: .\ntrain: train.txt\n"

    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def dataset_stem(frame_index: int) -> str:
    return f"frame_{frame_index:06d}"


def dataset_image_path(stem: str) -> str:
    return f"data/images/train/{stem}.png"


def create_zip(output_dir: Path, files: list[Path]) -> Path:
    zip_path = output_dir / "result.zip"

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(output_dir).as_posix())

    return zip_path


def export_dataset(args) -> dict:
    config = load_config()
    output_dir = Path(args.output_dir or Path(config.get("exports_dir", "exports")) / "yolo_obb_export")
    labels_train_dir = output_dir / "labels" / "train"
    train_path = output_dir / "train.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()

    try:
        query = (
            db.query(Frame, Prediction)
            .outerjoin(Prediction, Prediction.frame_id == Frame.id)
        )

        if args.order_code != "all":
            query = query.filter(Frame.order_code == args.order_code)

        if args.camera != "all":
            query = query.filter(Frame.camera == args.camera)

        rows = (
            query.order_by(Frame.order_code, Frame.camera, Frame.video_id, Frame.frame_number)
            .all()
        )

        frame_ids = [frame.id for frame, _ in rows]
        queue_items_by_frame = get_latest_queue_items_by_frame(db, frame_ids)
        queue_item_ids = [queue_item.id for queue_item in queue_items_by_frame.values()]
        reviews_by_queue_item = get_latest_reviews(db, queue_item_ids)
        manual_boxes_by_frame = get_manual_boxes(db, frame_ids)
        saved_manual_annotation_frame_ids = get_saved_manual_annotation_frame_ids(db, frame_ids)

        train_rows = []
        zip_files = []
        exported_count = 0
        skipped_count = 0

        for frame_index, (frame, prediction) in enumerate(rows):
            stem = dataset_stem(frame_index)
            train_rows.append(dataset_image_path(stem))

            queue_item = queue_items_by_frame.get(frame.id)
            review = reviews_by_queue_item.get(queue_item.id) if queue_item else None
            manual_boxes = manual_boxes_by_frame.get(frame.id, [])
            has_manual_annotation = frame.id in saved_manual_annotation_frame_ids
            boxes = manual_boxes if has_manual_annotation else prediction_boxes(prediction, frame)

            export, reason = should_export(
                queue_item=queue_item,
                review=review,
                boxes=manual_boxes,
                has_manual_annotation=has_manual_annotation,
                include_empty_labels=args.include_empty_labels,
                include_unreviewed_high_confidence=args.include_unreviewed_high_confidence,
            )

            if export and not manual_boxes and review and review.review_status == "correct_detection":
                boxes = prediction_boxes(prediction, frame)

            if export and not manual_boxes and not review and reason == "unreviewed_high_confidence":
                boxes = prediction_boxes(prediction, frame)

            if export and reason not in {"correct_no_detection", "manual_annotation"} and not boxes:
                export = False
                reason = "no_boxes_to_export"

            if export:
                lines = label_lines_for_boxes(boxes, frame)
                if lines:
                    label_output = labels_train_dir / f"{stem}.txt"
                    label_output.write_text("\n".join(lines), encoding="utf-8")
                    zip_files.append(label_output)
                    exported_count += 1
                else:
                    skipped_count += 1
            else:
                skipped_count += 1

        write_data_yaml(output_dir)
        train_path.write_text("\n".join(train_rows) + ("\n" if train_rows else ""), encoding="utf-8")
        zip_files = [output_dir / "data.yaml", train_path, *zip_files]

        zip_path = create_zip(output_dir, zip_files) if args.zip else None

        return {
            "output_dir": str(output_dir),
            "labels_dir": str(labels_train_dir),
            "train_path": str(train_path),
            "data_yaml_path": str(output_dir / "data.yaml"),
            "zip_path": str(zip_path) if zip_path else None,
            "total_rows": len(rows),
            "exported_count": exported_count,
            "skipped_count": skipped_count,
        }

    finally:
        db.close()


def main():
    args = parse_args()
    result = export_dataset(args)

    print("YOLO OBB export finished.")
    print(f"Output dir: {result['output_dir']}")
    print(f"Labels dir: {result['labels_dir']}")
    print(f"Train file: {result['train_path']}")
    print(f"Data yaml: {result['data_yaml_path']}")

    if result["zip_path"]:
        print(f"Zip: {result['zip_path']}")

    print(f"Total frames checked: {result['total_rows']}")
    print(f"Label files exported: {result['exported_count']}")
    print(f"Frames without labels: {result['skipped_count']}")


if __name__ == "__main__":
    main()
