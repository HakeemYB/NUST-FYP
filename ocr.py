#------------- EasyOCR------------
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use XCB (X11) platform plugin

import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import easyocr  # Import EasyOCR library
import csv
import uuid
import numpy as np

region_threshold = 0.7
detection_threshold = 0.7

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1
]
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][2], result[0][0]))  # Corrected the index here
        height = np.sum(np.subtract(result[0][3], result[0][1]))  # Corrected the index here
        
        if (length * height) / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

def ocr_it(image, detections, detection_threshold, region_threshold):
    # Scores, boxes, and classes above threshold
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # Initialize EasyOCR reader outside the loop
    reader = easyocr.Reader(['en'])

    # Create an empty list to store the results
    results = []

    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]

        ocr_result = reader.readtext(region)

        text = filter_text(region, ocr_result, region_threshold)
        results.append((text, region))

    return results

def save_results(text, region, csv_filename, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())

    cv2.imwrite(os.path.join(folder_path, img_name), region)

    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for t in text:
            csv_writer.writerow([img_name, t])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='yolov8 live.')
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 640],
        nargs=2,
        type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)  # open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("/home/hakym/fyp_nust/model/best .pt")  # Removed extra space in the model path

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Initialize EasyOCR reader outside the loop
    reader = easyocr.Reader(['en'])  # Use the appropriate language for your plates

    while True:
        ret, frame = cap.read()  # get a new frame from the video capture object
        image_np = np.array(frame)
        image_np_with_detections = image_np.copy()

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        labels = []

        for confidence, class_id in zip(detections.confidence, detections.class_id):
            if class_id is not None:
                label = f"{model.model.names[class_id]} {confidence:0.2f}"
                try:
                    results = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
                    for text, region in results:
                        save_results(text, region, 'realtimeresults.csv', 'Detection_Images')
                        label += f" Plate: {', '.join(text)}"  # Join multiple text results
                except:
                    pass

                labels.append(label)
            else:
                label = "Unknown"  # Handle the case when class_id is None
                labels.append(label)

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        cv2.imshow('Realtime Detection', frame)  # Corrected the window name

        if (cv2.waitKey(30) == 27):
            break  # waits for a key event

if __name__ == "__main__":
    main()
