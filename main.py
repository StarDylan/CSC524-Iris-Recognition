import cv2
import numpy as np
import argparse
from dataclasses import dataclass
from cv2.videoio_registry import getBackendName
from cv2_enumerate_cameras import supported_backends, enumerate_cameras
from pathlib import Path
import iris
import matplotlib.pyplot as plt
import time
import json
from db_manager import Db

def good_brightness(frame, min_threshold=50, max_threshold=200):
    brightness = np.mean(frame)
    return brightness > min_threshold and brightness < max_threshold, brightness

def main():

    config = parse_args()

    using_camera = config.filename is None

    if using_camera:
        cam = select_camera()
        cap = cv2.VideoCapture(cam.index, cam.backend)
    else:
        cap = cv2.VideoCapture(config.filename)
        print(f"Using video file {config.filename}")

    if not cap.isOpened():
        print("Failed to open")
        print(cap.getExceptionMode())

    highest_sharp = None
    highest_sharp_frame = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    highest_sharp_iris_pipeline: iris.IRISPipeline | None = None
    highest_sharp_output = None

    while True:
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
        output = iris_pipeline.run(
            iris.IRImage(
                img_data=frame,
                image_id="sharpness_test",
                eye_side="right",
            )
        )

        elapsed = time.time() - start

        if not using_camera:
            for _ in range(int(fps * elapsed)):
                cap.grab()

        if output.get("error") is not None:
            print("Image skipped: ", output["error"]["error_type"])
            if cv2.waitKey(30) == ord("q"):
                break
            continue
        
        sharpness = output["metadata"]["sharpness_score"]
        print(output)

        print(f"Sharpness: {sharpness:.2f}, Brightness: {brightness:.2f}, Elapsed: {elapsed:.2f}s")
        if highest_sharp is None or sharpness > highest_sharp:
            highest_sharp_frame = frame
            highest_sharp = sharpness
            highest_sharp_iris_pipeline = iris_pipeline
            highest_sharp_output = output

            if sharpness > 461:
                print("-"*50)
                print("Found good image!")
                print("-"*50)

        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if highest_sharp_iris_pipeline is None:
        print("No frames processed")
        exit(0)
    
    assert highest_sharp_output is not None
    assert highest_sharp_frame is not None

    # print(json.dumps(highest_sharp_output, indent=4))

    iris_visualizer = iris.visualisation.IRISVisualizer()
    
    canvas = iris_visualizer.plot_all_geometry(
        ir_image=iris.IRImage(img_data=highest_sharp_frame, eye_side="right", image_id=None),
        geometry_polygons=highest_sharp_iris_pipeline.call_trace['geometry_estimation'],
        eye_orientation=highest_sharp_iris_pipeline.call_trace['eye_orientation'],
        eye_center=highest_sharp_iris_pipeline.call_trace['eye_center_estimation'],
    )

    plt.show()
    
    canvas = iris_visualizer.plot_iris_template(highest_sharp_output["iris_template"])

    plt.show()

    canvas = iris_visualizer.plot_normalized_iris(
        normalized_iris=highest_sharp_iris_pipeline.call_trace['normalization'],
    )
    plt.show()

    mode = input("Do you want to store? (y/n)")

    db = Db()

    
    if mode == "y" or mode == "Y":
        name = input("Enter a name: ")
        db.replace(name, highest_sharp_output['iris_template'], highest_sharp_output["metadata"]["sharpness_score"])

    for eye in db.get_all_eye_templates():
    
        matcher = iris.HammingDistanceMatcher()

        distance = matcher.run(highest_sharp_output["iris_template"], eye[0])

        if distance < 0.33:
            print(f"====Match==== {eye[1]} - {distance} ||||||||")
        else:
            print(f"------------- {eye[1]} - {distance}")



@dataclass
class Config:
    filename: str | None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="CSC524 IRIS Recognition System")
    
    parser.add_argument(
        "-f",
        "--filename",
        nargs="?",
        help="Optional input filename"
    )

    args = parser.parse_args()


    if args.filename and not Path(args.filename).exists():
        print(f"Error! Can't find file {args.filename}")
        exit()


    return Config(filename=args.filename)


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
