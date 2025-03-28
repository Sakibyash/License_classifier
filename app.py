import gradio as gr
import torch
import cv2
import pytesseract

# Load the trained YOLOv5 model (replace 'best.pt' with your actual model path)
model = torch.hub.load('ultralytics/yolov10n', 'custom', path='best(3).pt')

def process_video(input_video):
    # Read video frames
    cap = cv2.VideoCapture(input_video.name)
    output_video = "output.mp4"

    # Get video details
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLO model to detect license plates
        results = model(frame)
        detected_boxes = results.xyxy[0]  # Bounding boxes, confidence scores, and class IDs

        # Loop through all the detected bounding boxes
        for box in detected_boxes:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])  # Extract bounding box coordinates and confidence
            if conf > 0.5:  # You can adjust the confidence threshold as needed
                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Optionally, draw the confidence score and label (use class names for 3 classes)
                if cls == 0:
                    label = "Analog License Plate"
                elif cls == 1:
                    label = "Digital License Plate"
                elif cls == 2:
                    label = "Non-License Plate"
                else:
                    label = "Unknown"
                
                # Draw label and confidence on frame
                cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Optionally, collect the bounding box coordinates for further processing (e.g., OCR)
                license_plate = frame[y1:y2, x1:x2]
                # Convert to grayscale for better OCR results
                gray_license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

                # Use Tesseract OCR to extract text from Bangla license plates (adjust config as needed)
                text = pytesseract.image_to_string(gray_license_plate, config="--psm 6 -l ben")  # 'ben' is for Bangla
                print(f"Detected License Plate Text: {text.strip()}")

        # Write the annotated frame to output video
        out.write(frame)

    cap.release()
    out.release()
    return output_video

# Create Gradio Interface
interface = gr.Interface(fn=process_video, inputs=gr.inputs.Video(), outputs=gr.outputs.Video())
interface.launch()
