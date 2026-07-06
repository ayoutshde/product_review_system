import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_camera_regions(path: Path = Path("camera_regions.json")) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            "camera_regions.json not found. Run: python src/create_camera_polygon.py"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_boxes_json(value: str) -> list[dict]:
    if not value:
        return []

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return []


def box_center(points: list[list[float]]) -> tuple[float, float] | None:
    if len(points) != 4:
        return None

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)

    return center_x, center_y


def point_inside_polygon(point: tuple[float, float], polygon: list[list[int]]) -> bool:
    polygon_np = np.array(polygon, dtype=np.int32)

    result = cv2.pointPolygonTest(
        polygon_np,
        (float(point[0]), float(point[1])),
        False
    )

    return result >= 0


def filter_boxes_by_roi(boxes: list[dict], fridge_polygon: list[list[int]]) -> tuple[list[dict], int]:
    """
    Keep boxes outside the fridge polygon.
    Ignore boxes whose center is inside the fridge polygon.
    """
    kept_boxes = []
    ignored_count = 0

    for box in boxes:
        points = box.get("points", [])
        center = box_center(points)

        if center is None:
            ignored_count += 1
            continue

        is_inside_fridge = point_inside_polygon(center, fridge_polygon)

        if is_inside_fridge:
            ignored_count += 1
        else:
            kept_boxes.append(box)

    return kept_boxes, ignored_count


def main():
    predictions_path = Path("inference_results") / "predictions.csv"
    raw_backup_path = Path("inference_results") / "predictions_raw.csv"
    output_path = Path("inference_results") / "predictions_roi_filtered.csv"

    if not predictions_path.exists():
        print(f"[ERROR] predictions.csv not found: {predictions_path}")
        print("Run inference first:")
        print("python src/run_inference.py")
        return

    try:
        camera_regions = load_camera_regions()
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return

    # Keep original raw predictions as backup
    if not raw_backup_path.exists():
        shutil.copy2(predictions_path, raw_backup_path)
        print(f"Raw predictions backup saved to: {raw_backup_path}")
    else:
        print(f"Raw predictions backup already exists: {raw_backup_path}")

    df = pd.read_csv(predictions_path)

    if "camera" not in df.columns:
        print("[ERROR] predictions.csv does not have camera column.")
        print("Run updated extract_frames.py and run_inference.py first.")
        return

    updated_rows = []

    total_raw_boxes = 0
    total_kept_boxes = 0
    total_ignored_boxes = 0
    frames_all_inside_roi = 0
    frames_with_kept_boxes = 0

    for _, row in df.iterrows():
        camera = str(row.get("camera", ""))

        boxes = parse_boxes_json(str(row.get("boxes_json", "[]")))
        raw_box_count = len(boxes)

        total_raw_boxes += raw_box_count

        if camera not in camera_regions:
            # If no polygon for this camera, keep original boxes.
            kept_boxes = boxes
            ignored_count = 0
            roi_filter_status = "no_polygon_for_camera"

        else:
            fridge_polygon = camera_regions[camera]["fridge_polygon"]
            kept_boxes, ignored_count = filter_boxes_by_roi(boxes, fridge_polygon)

            if raw_box_count == 0:
                roi_filter_status = "no_boxes"
            elif len(kept_boxes) == 0:
                roi_filter_status = "all_boxes_inside_fridge_polygon"
                frames_all_inside_roi += 1
            elif ignored_count > 0:
                roi_filter_status = "partially_filtered"
                frames_with_kept_boxes += 1
            else:
                roi_filter_status = "kept_all_boxes"
                frames_with_kept_boxes += 1

        kept_box_count = len(kept_boxes)
        max_confidence = max([float(box.get("confidence", 0.0)) for box in kept_boxes]) if kept_boxes else 0.0
        prediction_status = "detected" if kept_box_count > 0 else "no_detection"

        total_kept_boxes += kept_box_count
        total_ignored_boxes += ignored_count

        updated_row = row.to_dict()

        updated_row["raw_box_count"] = raw_box_count
        updated_row["ignored_inside_roi_count"] = ignored_count
        updated_row["kept_outside_roi_count"] = kept_box_count
        updated_row["roi_filter_status"] = roi_filter_status

        updated_row["prediction_status"] = prediction_status
        updated_row["max_confidence"] = round(max_confidence, 4)
        updated_row["box_count"] = kept_box_count
        updated_row["boxes_json"] = json.dumps(kept_boxes)

        updated_rows.append(updated_row)

    output_df = pd.DataFrame(updated_rows)

    output_df.to_csv(output_path, index=False, encoding="utf-8")

    # Also overwrite predictions.csv so existing pipeline continues normally
    output_df.to_csv(predictions_path, index=False, encoding="utf-8")

    print("\nROI filtering finished.")
    print(f"ROI-filtered predictions saved to: {output_path}")
    print(f"predictions.csv updated with ROI-filtered results: {predictions_path}")

    print("\nBox counts:")
    print(f"Raw boxes: {total_raw_boxes}")
    print(f"Ignored inside fridge polygon: {total_ignored_boxes}")
    print(f"Kept outside fridge polygon: {total_kept_boxes}")

    print("\nFrame status counts:")
    print(output_df["roi_filter_status"].value_counts())

    print("\nPrediction status after ROI filter:")
    print(output_df["prediction_status"].value_counts())


if __name__ == "__main__":
    main()