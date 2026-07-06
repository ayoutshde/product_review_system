import json
from pathlib import Path

import cv2
import pandas as pd


MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


def load_metadata() -> pd.DataFrame:
    metadata_path = Path("inference_results") / "frames_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "frames_metadata.csv not found. Run: python src/extract_frames.py"
        )

    return pd.read_csv(metadata_path)


def choose_sample_frame_for_each_camera(df: pd.DataFrame) -> dict:
    """
    Pick one representative frame for each camera.
    We use a middle frame instead of the first frame because the fridge view is usually clearer.
    """
    samples = {}

    for camera, group in df.groupby("camera"):
        group_sorted = group.sort_values("frame_id").reset_index(drop=True)

        middle_index = len(group_sorted) // 2
        row = group_sorted.iloc[middle_index]

        samples[camera] = {
            "camera": camera,
            "order_code": str(row.get("order_code", "")),
            "frame_id": int(row["frame_id"]),
            "frame_path": str(row["frame_path"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
        }

    return samples


def resize_for_display(image):
    height, width = image.shape[:2]

    scale_w = MAX_DISPLAY_WIDTH / width
    scale_h = MAX_DISPLAY_HEIGHT / height
    scale = min(scale_w, scale_h, 1.0)

    display_width = int(width * scale)
    display_height = int(height * scale)

    display_image = cv2.resize(image, (display_width, display_height))

    return display_image, scale


def draw_ui(display_image, display_points, camera):
    canvas = display_image.copy()

    # Draw points
    for index, point in enumerate(display_points):
        x, y = point
        cv2.circle(canvas, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            canvas,
            str(index + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # Draw lines
    if len(display_points) >= 2:
        for i in range(len(display_points) - 1):
            cv2.line(canvas, display_points[i], display_points[i + 1], (0, 255, 255), 2)

    # Close polygon preview
    if len(display_points) >= 3:
        cv2.line(canvas, display_points[-1], display_points[0], (0, 255, 255), 2)

    instructions = [
        f"Camera: {camera}",
        "Left click: add polygon point",
        "Right click: undo last point",
        "R: reset points",
        "S or ENTER: save polygon",
        "Q or ESC: quit",
        "Draw ONLY the inside area of the fridge.",
    ]

    y = 30
    for text in instructions:
        cv2.putText(
            canvas,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        y += 28

    return canvas


def collect_polygon_for_camera(camera: str, frame_info: dict) -> dict | None:
    frame_path = Path(frame_info["frame_path"])

    if not frame_path.exists():
        print(f"[ERROR] Frame not found: {frame_path}")
        return None

    image = cv2.imread(str(frame_path))

    if image is None:
        print(f"[ERROR] Could not read image: {frame_path}")
        return None

    display_image, scale = resize_for_display(image)

    original_points = []
    display_points = []

    window_name = f"Draw fridge polygon - {camera}"

    def mouse_callback(event, x, y, flags, param):
        nonlocal original_points, display_points

        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(x / scale)
            original_y = int(y / scale)

            original_points.append([original_x, original_y])
            display_points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if original_points:
                original_points.pop()
                display_points.pop()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nDrawing polygon for {camera}")
    print(f"Frame: {frame_path}")
    print("Left click = add point")
    print("Right click = undo")
    print("S or ENTER = save")
    print("R = reset")
    print("Q or ESC = quit")

    while True:
        canvas = draw_ui(display_image, display_points, camera)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(20) & 0xFF

        # q or esc
        if key == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            print("Quit without saving.")
            return None

        # r
        if key == ord("r"):
            original_points = []
            display_points = []

        # s or enter
        if key == ord("s") or key == 13:
            if len(original_points) < 3:
                print("Polygon needs at least 3 points.")
                continue

            cv2.destroyWindow(window_name)

            return {
                "camera": camera,
                "fridge_polygon": original_points,
                "sample_frame_path": str(frame_path),
                "sample_order_code": frame_info["order_code"],
                "sample_frame_id": frame_info["frame_id"],
                "image_width": frame_info["width"],
                "image_height": frame_info["height"],
            }


def main():
    df = load_metadata()

    if "camera" not in df.columns:
        print("[ERROR] frames_metadata.csv does not have camera column.")
        print("Run the updated extract_frames.py first.")
        return

    samples = choose_sample_frame_for_each_camera(df)

    output_path = Path("camera_regions.json")
    regions = {}

    for camera, frame_info in samples.items():
        polygon_data = collect_polygon_for_camera(camera, frame_info)

        if polygon_data is not None:
            regions[camera] = polygon_data

    if not regions:
        print("No polygons saved.")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, ensure_ascii=False, indent=2)

    print(f"\nCamera regions saved to: {output_path}")
    print("Saved cameras:")

    for camera in regions:
        print(f"- {camera}: {len(regions[camera]['fridge_polygon'])} points")


if __name__ == "__main__":
    main()