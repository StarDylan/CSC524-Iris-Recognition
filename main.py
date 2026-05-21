from cv2 import VideoCapture
import cv2
import numpy as np
import argparse
from dataclasses import dataclass
from typing import cast
from cv2.videoio_registry import getBackendName
from cv2_enumerate_cameras import supported_backends, enumerate_cameras
from pathlib import Path
import iris
import matplotlib.pyplot as plt
import time
from db_manager import Db
import threading
import time


def good_brightness(frame: np.ndarray, min_threshold: float = 50, max_threshold: float = 200) -> tuple[bool, float]:
    brightness = float(np.mean(frame))
    return brightness > min_threshold and brightness < max_threshold, brightness


@dataclass
class SharpCandidate:
    frame: np.ndarray
    iris_pipeline: iris.IRISPipeline
    output: dict[str, object]
    sharpness: float
    brightness: float
    elapsed: float


def keep_top_candidates(candidates: list[SharpCandidate], candidate: SharpCandidate, limit: int = 5) -> None:
    candidates.append(candidate)
    candidates.sort(key=lambda item: item.sharpness, reverse=True)
    del candidates[limit:]


def select_candidate(candidates: list[SharpCandidate]) -> SharpCandidate:
    print("\nTop sharp images:")
    for index, candidate in enumerate(candidates, start=1):
        print(f"{index}: Sharpness {candidate.sharpness:.2f}, Brightness {candidate.brightness:.2f}, Elapsed {candidate.elapsed:.2f}s")

    plt.ion()
    fig, axes = plt.subplots(1, len(candidates), figsize=(4 * len(candidates), 4))
    if len(candidates) == 1:
        axes = [axes]

    for index, (axis, candidate) in enumerate(zip(axes, candidates, strict=False), start=1):
        axis.imshow(candidate.frame, cmap="gray")
        axis.set_title(str(index))
        axis.axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

    while True:
        try:
            choice = input(f"Select image to continue analysis [1-{len(candidates)}] (or q to quit): ").strip()
            if choice.lower() == "q":
                plt.close(fig)
                raise SystemExit(0)

            if choice.isdigit():
                selected_index = int(choice)
                if 1 <= selected_index <= len(candidates):
                    selected = candidates[selected_index - 1]
                    plt.close(fig)
                    return selected

            print("Invalid selection. Please enter a valid number from the list.")
        except (KeyboardInterrupt):
            continue

latest_frame = None
lock = threading.Lock()
running = True

def grab_frames(cap: VideoCapture):
    global latest_frame
    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame


def main():

    config = parse_args()

    using_camera = config.filename is None

    if using_camera:
        cam = select_camera()
        cap = cv2.VideoCapture(cam.index, cam.backend)
    else:
        filename = config.filename
        if filename is None:
            raise ValueError("A filename is required when not using a camera")
        cap = cv2.VideoCapture(filename)
        print(f"Using video file {filename}")

    if not cap.isOpened():
        print("Failed to open")
        print(cap.getExceptionMode())

    fps = cap.get(cv2.CAP_PROP_FPS)
    top_candidates: list[SharpCandidate] = []

    if using_camera:
        thread = threading.Thread(target=lambda: grab_frames(cap), daemon=True)
        thread.start()

    while True:
        if using_camera:
            with lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ok, brightness = good_brightness(frame)

        if not ok:
            continue

        cv2.imshow("Frame", frame)

        iris_pipeline = iris.IRISPipeline()
        
        start = time.time()
        raw_output = iris_pipeline.run(
            iris.IRImage(
                img_data=frame,
                image_id="sharpness_test",
                eye_side="right",
            )
        )
        output = cast(dict[str, object], raw_output)

        elapsed = time.time() - start

        if not using_camera:
            for _ in range(int(fps * elapsed)):
                _ = cap.grab()

        if output.get("error") is not None:
            error = cast(dict[str, object], output["error"])
            print("Image skipped:", error["error_type"])
            if cv2.waitKey(30) == ord("q"):
                break
            continue
        
        metadata = cast(dict[str, object], output["metadata"])
        sharpness = float(cast(float, metadata["sharpness_score"]))
        print(output)

        print(f"Sharpness: {sharpness:.2f}, Brightness: {brightness:.2f}, Elapsed: {elapsed:.2f}s")
        candidate = SharpCandidate(
            frame=frame.copy(),
            iris_pipeline=iris_pipeline,
            output=output,
            sharpness=sharpness,
            brightness=brightness,
            elapsed=elapsed,
        )
        keep_top_candidates(top_candidates, candidate)

        if sharpness > 461:
            print("-"*50)
            print("Found good image!")
            print("-"*50)

        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not top_candidates:
        print("No frames processed")
        exit(0)
    
    selected_candidate = select_candidate(top_candidates)
    selected_output = selected_candidate.output
    selected_frame = selected_candidate.frame
    selected_pipeline = selected_candidate.iris_pipeline
    selected_template = cast(iris.IrisTemplate, selected_output["iris_template"])
    selected_metadata = cast(dict[str, object], selected_output["metadata"])
    selected_call_trace = cast(dict[str, object], selected_pipeline.call_trace)
    selected_geometry = cast(iris.GeometryPolygons, selected_call_trace["geometry_estimation"])
    selected_eye_orientation = cast(iris.EyeOrientation, selected_call_trace["eye_orientation"])
    selected_eye_center = cast(iris.EyeCenters, selected_call_trace["eye_center_estimation"])
    selected_normalized_iris = cast(iris.NormalizedIris, selected_call_trace["normalization"])

    iris_visualizer = iris.visualisation.IRISVisualizer()
    
    geometry_canvas = iris_visualizer.plot_all_geometry(
        ir_image=iris.IRImage(img_data=selected_frame, eye_side="right", image_id=None),
        geometry_polygons=selected_geometry,
        eye_orientation=selected_eye_orientation,
        eye_center=selected_eye_center,
    )
    assert geometry_canvas is not None

    plt.show()
    
    template_canvas = iris_visualizer.plot_iris_template(selected_template)
    assert template_canvas is not None

    plt.show()

    normalized_canvas = iris_visualizer.plot_normalized_iris(
        normalized_iris=selected_normalized_iris,
    )
    assert normalized_canvas is not None
    plt.show()

    mode = input("Do you want to store? (y/n)")

    db = Db()

    
    if mode == "y" or mode == "Y":
        name = input("Enter a name: ")
        db.replace(name, selected_template, cast(float, selected_metadata["sharpness_score"]))

    for eye in db.get_all_eye_templates():
    
        matcher = iris.HammingDistanceMatcher()

        distance = matcher.run(selected_template, eye[0])

        if distance < 0.33:
            print(f"====Match==== {eye[1]} - {distance} ||||||||")
        else:
            print(f"------------- {eye[1]} - {distance}")



@dataclass
class Config:
    filename: str | None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="CSC524 IRIS Recognition System")
    
    _ = parser.add_argument(
        "-f",
        "--filename",
        nargs="?",
        help="Optional input filename"
    )

    args = parser.parse_args()
    filename = cast(str | None, args.filename)


    if filename is not None and not Path(filename).exists():
        print(f"Error! Can't find file {filename}")
        exit()


    return Config(filename=filename)


def list_cameras():
    cameras = []

    first_backend = True
    for backend in supported_backends:
        first = True 
        for camera_info in enumerate_cameras(backend):
            if first:
                if not first_backend:
                    print()
                print(getBackendName(backend))
                print("-"*20)
                first = False
                first_backend = False
            print(f"{len(cameras)}: {camera_info}")
            cameras.append(camera_info)

    return cameras

def select_camera():
    # List cameras
    cams = list_cameras()

    if not cams:
        print("No Cameras. Please specify video file via -f")
        exit(0)
    # Select camera
    cam_id = int(input(f"\nSelect camera {cams}: "))
    if cam_id not in list(range(len(cams))):
        raise ValueError("Invalid camera selected")

    return cams[cam_id]

if __name__ == "__main__":
    main()
