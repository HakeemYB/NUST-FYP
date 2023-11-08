#-----------pytesseract-----------------
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import pytesseract
import datetime
import openpyxl
import os

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

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
                if confidence > 0.8:
                    # If confidence is greater than 75%, perform OCR
                    plate_image = frame[int(detections.xyxy[0][1]):int(detections.xyxy[0][3]),
                                       int(detections.xyxy[0][0]):int(detections.xyxy[0][2])]
                    # Perform OCR using Tesseract
                    plate_number = pytesseract.image_to_string(plate_image, config='--psm 7')
                    if plate_number:
                        # Display plate number along with class_id and confidence
                        label += f" Plate: {plate_number}"
                        
                        # Check if this plate is detected for the first time
                        if plate_number not in detected_plates:
                            # Save the cropped image with a timestamp
                            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            image_name = f"{save_folder}/plate_{timestamp}.jpg"
                            cv2.imwrite(image_name, plate_image)
                            detected_plates[plate_number] = True

                        # Create a timestamp for the log entry
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Prepare log data
                        log_data = [timestamp, plate_number, model.model.names[class_id], confidence]

                        # Save the log entry
                        save_log(log_data, sheet)
                        
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


#------------- EasyOCR------------
# import os
# os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use XCB (X11) platform plugin

# import cv2
# import argparse
# from ultralytics import YOLO
# import supervision as sv
# import easyocr  # Import EasyOCR library

# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description='yolov8 live.')
#     parser.add_argument(
#         "--webcam-resolution", 
#         default=[640, 640], 
#         nargs=2, 
#         type=int)

#     args = parser.parse_args()
#     return args

# def main():
#     args = parse_arguments()
#     frame_width, frame_height = args.webcam_resolution

#     cap = cv2.VideoCapture(0)  # open the default camera
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#     model = YOLO("/home/hakym/fyp_nust/model/best .pt")

#     box_annotator = sv.BoxAnnotator(
#         thickness=2,
#         text_thickness=2,
#         text_scale=1
#     )

#     # Initialize EasyOCR reader
#     reader = easyocr.Reader(['en'])  # Use the appropriate language for your plates

#     while True:
#         ret, frame = cap.read()  # get a new frame from the video capture object

#         result = model(frame)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         print(detections)

#         labels = []
#         for confidence, class_id in zip(detections.confidence, detections.class_id):
#             if class_id is not None:
#                 label = f"{model.model.names[class_id]} {confidence:0.2f}"
#                 if confidence > 0.7:
#                     # If confidence is greater than 70%, perform OCR
#                     plate_image = frame[int(detections.xyxy[0][1]):int(detections.xyxy[0][3]),
#                                        int(detections.xyxy[0][0]):int(detections.xyxy[0][2])]
#                     plate_number = reader.readtext(plate_image)
#                     print(plate_number)
#                     if plate_number:
#                         # Display plate number along with class_id and confidence
#                         label += f" Plate: {plate_number[0][1]}"
#                 labels.append(label)
#             else:
#                 label = "Unknown"  # Handle the case when class_id is None
#                 labels.append(label)

#         frame = box_annotator.annotate(
#             scene=frame, 
#             detections=detections, 
#             labels=labels
#         )
#         cv2.imshow("/home/hakym/fyp_nust/model/best .pt", frame)

#         if (cv2.waitKey(30) == 27):
#             break  # waits for a key event

# if __name__ == "__main__":
#     main()
