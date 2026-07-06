from pathlib import Path

import cv2

from src.video_loader import load_config, get_grouped_videos
from src.db.database import SessionLocal
from src.db.models import Video, Frame


def create_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_or_create_video(db, video_info: dict) -> Video:
    existing_video = (
        db.query(Video)
        .filter(Video.video_path == video_info["video_path"])
        .first()
    )

    if existing_video:
        return existing_video

    video = Video(
        order_code=video_info["order_code"],
        camera=video_info["camera"],
        video_name=video_info["video_name"],
        video_path=video_info["video_path"],
        device_id=video_info.get("device_id", ""),
        session_id=video_info.get("session_id", ""),
        date=video_info.get("date", ""),
        time=video_info.get("time", ""),
    )

    db.add(video)
    db.commit()
    db.refresh(video)

    return video


def frame_already_exists(db, video_id: int, frame_number: int) -> bool:
    existing_frame = (
        db.query(Frame)
        .filter(
            Frame.video_id == video_id,
            Frame.frame_number == frame_number
        )
        .first()
    )

    return existing_frame is not None


def extract_frames_from_video(db, video_info: dict, output_root: Path) -> int:
    video_path = Path(video_info["video_path"])

    order_code = video_info["order_code"]
    camera = video_info["camera"]

    output_dir = output_root / order_code / camera
    create_dir(output_dir)

    video_record = get_or_create_video(db, video_info)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nProcessing video")
    print(f"Order code: {order_code}")
    print(f"Camera: {camera}")
    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames reported: {total_frames}")

    frame_number = 0
    inserted_count = 0
    skipped_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        frame_filename = f"frame_{frame_number:06d}.jpg"
        frame_path = output_dir / frame_filename

        cv2.imwrite(str(frame_path), frame)

        if frame_already_exists(db, video_record.id, frame_number):
            skipped_count += 1
            continue

        timestamp = frame_number / fps if fps and fps > 0 else 0.0

        frame_record = Frame(
            video_id=video_record.id,
            order_code=order_code,
            camera=camera,
            frame_number=frame_number,
            frame_path=str(frame_path),
            timestamp=round(timestamp, 4),
            width=width,
            height=height,
        )

        db.add(frame_record)
        inserted_count += 1

        if inserted_count % 100 == 0:
            db.commit()
            print(f"Inserted {inserted_count} frame records...")

    db.commit()
    cap.release()

    print(f"Done: {video_path.name}")
    print(f"Inserted frames: {inserted_count}")
    print(f"Skipped existing frames: {skipped_count}")

    return inserted_count


def main():
    config = load_config()

    output_root = Path(config["extracted_frames_dir"])
    create_dir(output_root)

    grouped_videos = get_grouped_videos(config)

    if not grouped_videos:
        print("No videos found. Put videos inside input_videos folder first.")
        return

    db = SessionLocal()

    try:
        total_inserted = 0

        for order_code, videos in grouped_videos.items():
            print("\n==============================")
            print(f"Order Code: {order_code}")
            print(f"Videos in order: {len(videos)}")
            print("==============================")

            for video_info in videos:
                inserted = extract_frames_from_video(db, video_info, output_root)
                total_inserted += inserted

        print("\nDB frame extraction finished.")
        print(f"Total inserted frame records: {total_inserted}")

    finally:
        db.close()


if __name__ == "__main__":
    main()