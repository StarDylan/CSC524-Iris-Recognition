import cv2
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Optional
from cv2.videoio_registry import getBackendName
from cv2_enumerate_cameras import supported_backends, enumerate_cameras
from pathlib import Path

def good_brightness(frame, min_threshold=50, max_threshold=200):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > min_threshold and brightness < max_threshold, brightness

def main():

    config = parse_args()

    if config.filename is None:
        cam = select_camera()
        cap = cv2.VideoCapture(cam.index, cam.backend)
    else:
        cap = cv2.VideoCapture(config.filename)
        print(f"Using video file {config.filename}")

    if not cap.isOpened():
        print("Failed to open")
        print(cap.getExceptionMode())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, brightness = good_brightness(frame)

        if not ok:
            continue

        cv2.imshow("Frame", frame)

        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



@dataclass
class Config:
    filename: Optional[str]


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
