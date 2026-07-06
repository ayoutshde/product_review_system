from pathlib import Path
import json
import re
from collections import defaultdict


def load_config(config_path: str = "config.json") -> dict:
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file.resolve()}")

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_video_filename(video_path: Path) -> dict:
    """
    Expected filename example:
    2ed663b2d7c3_camera1_260617174607098772953_31bc0523_20260617_174613.mp4
    """

    stem = video_path.stem
    parts = stem.split("_")

    metadata = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "device_id": "",
        "camera": "",
        "order_code": "",
        "session_id": "",
        "date": "",
        "time": "",
        "parse_status": "failed"
    }

    if len(parts) >= 6:
        metadata["device_id"] = parts[0]
        metadata["camera"] = parts[1]
        metadata["order_code"] = parts[2]
        metadata["session_id"] = parts[3]
        metadata["date"] = parts[4]
        metadata["time"] = parts[5]
        metadata["parse_status"] = "ok"
        return metadata

    camera_match = re.search(r"(camera\d+)", stem)
    order_match = re.search(r"(\d{12,})", stem)

    if camera_match:
        metadata["camera"] = camera_match.group(1)

    if order_match:
        metadata["order_code"] = order_match.group(1)

    if metadata["camera"] and metadata["order_code"]:
        metadata["parse_status"] = "partial"

    return metadata


def get_video_files(config: dict) -> list[Path]:
    input_dir = Path(config["input_video_dir"])
    supported_extensions = config.get(
        "supported_video_extensions",
        [".mp4", ".avi", ".mov", ".mkv"]
    )

    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created input video folder: {input_dir.resolve()}")

    video_files = []

    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            video_files.append(file_path)

    video_files.sort()
    return video_files


def get_grouped_videos(config: dict) -> dict:
    video_files = get_video_files(config)
    grouped = defaultdict(list)

    for video_path in video_files:
        metadata = parse_video_filename(video_path)
        order_code = metadata.get("order_code") or "unknown_order"
        grouped[order_code].append(metadata)

    return dict(grouped)


def main():
    config = load_config()
    grouped = get_grouped_videos(config)

    if not grouped:
        print("No videos found.")
        print(f"Put your videos inside: {Path(config['input_video_dir']).resolve()}")
        return

    total_videos = sum(len(videos) for videos in grouped.values())

    print(f"Found {total_videos} video(s)")
    print(f"Grouped into {len(grouped)} order(s):\n")

    for order_code, videos in grouped.items():
        print(f"Order Code: {order_code}")

        for video in videos:
            print(
                f"  - {video['camera']} | "
                f"{video['video_name']} | "
                f"parse_status={video['parse_status']}"
            )

        cameras = sorted([video["camera"] for video in videos])
        print(f"  Cameras found: {cameras}\n")


if __name__ == "__main__":
    main()