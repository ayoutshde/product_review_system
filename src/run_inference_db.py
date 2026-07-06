from pathlib import Path
import argparse

import torch
from ultralytics import YOLO

from src.video_loader import load_config
from src.db.database import SessionLocal
from src.db.models import Frame, Prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO OBB inference and save predictions to PostgreSQL."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO inference confidence threshold. Example: --conf 0.25",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLO image size. Example: --imgsz 640",
    )

    return parser.parse_args()


def get_model_path(config: dict) -> str:
    possible_keys = [
        "model_path",
        "yolo_model_path",
        "obb_model_path",
        "trained_model_path",
    ]

    for key in possible_keys:
        if key in config and config[key]:
            return config[key]

    default_paths = [
        "models/yolo11m-obb.pt",
        "models/yolo11m.pt",
        "yolo11m-obb.pt",
        "yolo11m.pt",
    ]

    for path in default_paths:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "Model path not found. Add model_path to config.json or place model in models/yolo11m-obb.pt"
    )


def tensor_to_list(value):
    if value is None:
        return []

    if hasattr(value, "cpu"):
        return value.cpu().numpy().tolist()

    return value


def extract_obb_boxes(result, frame_width: int, frame_height: int) -> list:
    boxes = []

    if result.obb is None:
        return boxes

    obb = result.obb

    points_list = tensor_to_list(obb.xyxyxyxy)
    conf_list = tensor_to_list(obb.conf)
    cls_list = tensor_to_list(obb.cls)

    for index, points in enumerate(points_list):
        confidence = float(conf_list[index])
        class_id = int(cls_list[index])

        clean_points = []
        normalized_points = []

        for point in points:
            x = float(point[0])
            y = float(point[1])

            clean_points.append([round(x, 2), round(y, 2)])

            nx = x / frame_width if frame_width else 0
            ny = y / frame_height if frame_height else 0

            normalized_points.append([round(nx, 6), round(ny, 6)])

        xs = [p[0] for p in clean_points]
        ys = [p[1] for p in clean_points]

        box_data = {
            "box_type": "obb",
            "class_id": class_id,
            "confidence": round(confidence, 6),
            "points": clean_points,
            "normalized_points": normalized_points,
            "bbox_xyxy": [
                round(min(xs), 2),
                round(min(ys), 2),
                round(max(xs), 2),
                round(max(ys), 2),
            ],
        }

        boxes.append(box_data)

    return boxes


def save_prediction(db, frame: Frame, model_name: str, boxes: list) -> None:
    if boxes:
        prediction_status = "detected"
        max_confidence = max(box["confidence"] for box in boxes)
        box_count = len(boxes)
    else:
        prediction_status = "no_detection"
        max_confidence = 0.0
        box_count = 0

    existing = db.query(Prediction).filter(Prediction.frame_id == frame.id).first()

    if existing:
        existing.model_name = model_name
        existing.prediction_status = prediction_status
        existing.max_confidence = max_confidence
        existing.box_count = box_count
        existing.boxes_json = boxes
    else:
        prediction = Prediction(
            frame_id=frame.id,
            model_name=model_name,
            prediction_status=prediction_status,
            max_confidence=max_confidence,
            box_count=box_count,
            boxes_json=boxes,
        )

        db.add(prediction)


def main():
    args = parse_args()
    config = load_config()

    model_path = get_model_path(config)

    confidence_threshold = (
        args.conf
        if args.conf is not None
        else float(config.get("inference_confidence_threshold", 0.25))
    )

    image_size = (
        args.imgsz
        if args.imgsz is not None
        else int(config.get("image_size", 640))
    )

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Loading YOLO model...")
    print(f"Model path: {model_path}")
    print(f"Inference confidence threshold: {confidence_threshold}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Using CPU.")

    model = YOLO(model_path)

    db = SessionLocal()

    try:
        frames = db.query(Frame).order_by(Frame.id).all()

        if not frames:
            print("No frames found in database.")
            print("Run this first:")
            print("python -m src.extract_frames_db")
            return

        print(f"Total frames found in DB: {len(frames)}")

        processed_count = 0
        detected_count = 0
        no_detection_count = 0
        missing_file_count = 0

        for frame in frames:
            frame_path = Path(frame.frame_path)

            if not frame_path.exists():
                print(f"[WARNING] Missing frame file: {frame_path}")
                missing_file_count += 1
                continue

            results = model.predict(
                source=str(frame_path),
                imgsz=image_size,
                conf=confidence_threshold,
                device=device,
                verbose=False,
            )

            result = results[0]

            boxes = extract_obb_boxes(
                result=result,
                frame_width=frame.width,
                frame_height=frame.height,
            )

            save_prediction(
                db=db,
                frame=frame,
                model_name=Path(model_path).name,
                boxes=boxes,
            )

            processed_count += 1

            if boxes:
                detected_count += 1
            else:
                no_detection_count += 1

            if processed_count % 50 == 0:
                db.commit()
                print(f"Processed {processed_count}/{len(frames)} frames...")

        db.commit()

        print("\nDB inference finished.")
        print(f"Processed frames: {processed_count}")
        print(f"Detected frames: {detected_count}")
        print(f"No detection frames: {no_detection_count}")
        print(f"Missing frame files: {missing_file_count}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
#python -m src.run_inference_db --conf 0.25