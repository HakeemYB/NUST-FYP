import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use XCB (X11) platform plugin

import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import easyocr
import csv
import uuid
import numpy as np
from datetime import datetime

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='yolov8 live.')
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 640],
        nargs=2,
        type=int)
    args = parser.parse_args()
    return args

def crop_and_ocr(image, detections, reader):
    cropped_images = []
    plate_numbers = []

    for i in range(len(detections.xyxy)):
        confidence = detections.confidence[i].item()
        class_id = detections.class_id[i].item()
        bbox = detections.xyxy[i]

        if confidence > 0.7:
            x_min, y_min, x_max, y_max = bbox
            plate_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            plate_number = reader.readtext(plate_image)

            if plate_number:
                cropped_images.append(plate_image)
                plate_numbers.append(plate_number[0][1])

    return cropped_images, plate_numbers

def save_results(image, plate_number, folder_path, csv_filename):
    img_name = f'{uuid.uuid1()}.jpg'
    image_path = os.path.join(folder_path, img_name)
    cv2.imwrite(image_path, image)

    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([img_name, plate_number, timestamp])

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("/home/hakym/fyp_nust/model/best .pt")

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Use the appropriate language for your plates

    folder_path = 'Detected_Plates'
    os.makedirs(folder_path, exist_ok=True)

    csv_filename = 'results.csv'

    prev_plate = None  # To track the previous plate for avoiding duplicates

    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        for i in range(len(detections.xyxy)):
            confidence = detections.confidence[i].item()
            class_id = detections.class_id[i].item()
            bbox = detections.xyxy[i]

            if confidence > 0.7:
                x_min, y_min, x_max, y_max = bbox
                plate_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                plate_number = reader.readtext(plate_image)

                if plate_number and plate_number[0][1] != prev_plate:
                    save_results(plate_image, plate_number[0][1], folder_path, csv_filename)
                    prev_plate = plate_number[0][1]

        labels = []
        for confidence, class_id in zip(detections.confidence, detections.class_id):
            if class_id is not None:
                label = f"{model.model.names[class_id]} {confidence:0.2f}"
                labels.append(label)
            else:
                label = "Unknown"
                labels.append(label)

        cv2.imshow('Realtime Detection', frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()
