from collections import defaultdict
import argparse

from src.db.database import SessionLocal
from src.db.models import Frame, Prediction, QueueItem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create review/annotation queue items from PostgreSQL predictions."
    )

    parser.add_argument(
        "--filter-conf",
        type=float,
        default=0.70,
        help="Confidence threshold for queue filtering. Example: --filter-conf 0.70",
    )

    return parser.parse_args()


def is_possible_merged_detection(records, index: int) -> bool:
    if index <= 0 or index >= len(records) - 1:
        return False

    _, prev_prediction = records[index - 1]
    _, current_prediction = records[index]
    _, next_prediction = records[index + 1]

    return (
        current_prediction.box_count == 1
        and prev_prediction.box_count >= 2
        and next_prediction.box_count >= 2
    )


def create_queue_item(frame, prediction, filter_conf: float, is_merged: bool) -> QueueItem:
    if is_merged:
        return QueueItem(
            frame_id=frame.id,
            prediction_id=prediction.id,
            queue_type="review_queue",
            queue_reason="possible_merged_detection",
            decision="send_to_review",
            warning="possible_merged_detection",
            status="pending",
        )

    if prediction.prediction_status == "no_detection" or prediction.box_count == 0:
        return QueueItem(
            frame_id=frame.id,
            prediction_id=prediction.id,
            queue_type="review_queue",
            queue_reason="no_detection",
            decision="send_to_review",
            warning=None,
            status="pending",
        )

    if prediction.max_confidence >= filter_conf:
        return QueueItem(
            frame_id=frame.id,
            prediction_id=prediction.id,
            queue_type="skipped_queue",
            queue_reason="high_confidence",
            decision="skipped_high_confidence",
            warning=None,
            status="skipped",
        )

    return QueueItem(
        frame_id=frame.id,
        prediction_id=prediction.id,
        queue_type="annotation_queue",
        queue_reason="low_confidence",
        decision="send_to_annotation",
        warning=None,
        status="pending",
    )


def main():
    args = parse_args()
    filter_conf = args.filter_conf

    print("Creating DB queue items...")
    print(f"Filter confidence threshold: {filter_conf}")

    db = SessionLocal()

    try:
        prediction_count = db.query(Prediction).count()

        if prediction_count == 0:
            print("No predictions found.")
            print("Run this first:")
            print("python -m src.run_inference_db --conf 0.25")
            return

        old_queue_count = db.query(QueueItem).count()

        if old_queue_count > 0:
            print(f"Existing queue items found: {old_queue_count}")
            print("Deleting old queue items to avoid duplicates...")
            db.query(QueueItem).delete()
            db.commit()

        rows = (
            db.query(Frame, Prediction)
            .join(Prediction, Prediction.frame_id == Frame.id)
            .order_by(Frame.order_code, Frame.camera, Frame.frame_number)
            .all()
        )

        grouped = defaultdict(list)

        for frame, prediction in rows:
            group_key = (frame.order_code, frame.camera)
            grouped[group_key].append((frame, prediction))

        total_created = 0
        summary_by_reason = defaultdict(int)
        summary_by_decision = defaultdict(int)
        summary_by_status = defaultdict(int)

        for group_key, records in grouped.items():
            order_code, camera = group_key

            print(f"Processing group: order={order_code}, camera={camera}, frames={len(records)}")

            merged_frame_ids = set()

            for index in range(len(records)):
                frame, _ = records[index]

                if is_possible_merged_detection(records, index):
                    merged_frame_ids.add(frame.id)

            for frame, prediction in records:
                is_merged = frame.id in merged_frame_ids

                queue_item = create_queue_item(
                    frame=frame,
                    prediction=prediction,
                    filter_conf=filter_conf,
                    is_merged=is_merged,
                )

                db.add(queue_item)

                total_created += 1
                summary_by_reason[queue_item.queue_reason] += 1
                summary_by_decision[queue_item.decision] += 1
                summary_by_status[queue_item.status] += 1

        db.commit()

        print("\nQueue creation finished.")
        print(f"Total queue items created: {total_created}")

        print("\nSummary by reason:")
        for reason, count in sorted(summary_by_reason.items()):
            print(f"{reason}: {count}")

        print("\nSummary by decision:")
        for decision, count in sorted(summary_by_decision.items()):
            print(f"{decision}: {count}")

        print("\nSummary by status:")
        for status, count in sorted(summary_by_status.items()):
            print(f"{status}: {count}")

    finally:
        db.close()


if __name__ == "__main__":
    main()