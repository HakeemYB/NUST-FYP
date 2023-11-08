import cv2
import argparse
import datetime
from ultralytics import YOLO
import supervision as sv
import openpyxl
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use XCB (X11) platform plugin
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# Replace these with your Azure Cognitive Services credentials
subscription_key = '19acc7b9d3ce44cfa252aceda64a0936'
endpoint = 'https://seecs23fyp.cognitiveservices.azure.com/'

# Create a Computer Vision client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='yolov8 live.')
    parser.add_argument(
        "--webcam-resolution", 
        default=[640, 640], 
        nargs=2, 
        type=int)
    args = parser.parse_args()
    return args

def save_log(log_data, sheet):
    timestamp, plate_number, class_id, confidence = log_data

    # Remove illegal characters from plate_number
    plate_number = ''.join(char for char in plate_number if char.isprintable())

    # Append the log data
    sheet.append([timestamp, plate_number, class_id, confidence])

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)  # open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("/home/hakym/fyp_nust/model/best .pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    save_folder = 'cropped_images'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    detected_plates = {}  # Dictionary to keep track of detected plates

    try:
        workbook = openpyxl.load_workbook('vehicle_logs.xlsx')
        sheet = workbook.active
    except FileNotFoundError:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Timestamp", "Plate Number", "Class ID", "Confidence"])

    while True:
        ret, frame = cap.read()  # get a new frame from the video capture object

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        labels = []
        for confidence, class_id in zip(detections.confidence, detections.class_id):
            if class_id is not None:
                label = f"{model.model.names[class_id]} {confidence:0.2f}"
                if confidence > 0.7:
                    # If confidence is greater than 75%, perform OCR
                    plate_image = frame[int(detections.xyxy[0][1]):int(detections.xyxy[0][3]),
                                       int(detections.xyxy[0][0]):int(detections.xyxy[0][2])]

                    # Save the cropped image with a timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    image_name = f"{save_folder}/plate_{timestamp}.jpg"
                    cv2.imwrite(image_name, plate_image)

                    # Perform OCR using Azure Computer Vision service
                    with open(image_name, 'rb') as image_stream:
                        try:
                            results = computervision_client.recognize_printed_text_in_stream(image_stream)

                            # Process the recognized text
                            recognized_text = ""
                            for region in results.regions:
                                for line in region.lines:
                                    for word in line.words:
                                        word_text = word.text
                                        # Filter out unwanted characters
                                        word_text = ''.join(char for char in word_text if char.isalnum())
                                        recognized_text += word_text

                                        

                            if recognized_text:
                                label += f" Plate: {recognized_text}"

                                if recognized_text not in detected_plates:
                                    image_name = f"{save_folder}/{recognized_text}.jpg"
                                    cv2.imwrite(image_name, plate_image)
                                    detected_plates[recognized_text] = True

                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_data = [timestamp, recognized_text, model.model.names[class_id], confidence]
                                save_log(log_data, sheet)
                        except Exception as e:
                            print("OCR operation failed with an error:", e)

                labels.append(label)
            else:
                label = "Unknown"  # Handle the case when class_id is None
                labels.append(label)

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        cv2.imshow("/home/hakym/fyp_nust/model/best .pt", frame)

        if (cv2.waitKey(30) == 27):
            break  # waits for a key event

    cv2.destroyAllWindows()
    workbook.save('vehicle_logs.xlsx')

if __name__ == "__main__":
    main()
