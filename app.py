import streamlit as st
import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🚦 AI Smart Helmet Traffic Signal System")

model = YOLO("helmet.pt")

# Create violation folder
if not os.path.exists("violations"):
    os.makedirs("violations")

start = st.checkbox("Start Live Camera")

col1, col2 = st.columns([3, 1])
frame_placeholder = col1.empty()
signal_placeholder = col2.empty()
zoom_placeholder = col2.empty()

violation_count = 0

if start:
    cap = cv2.VideoCapture(0)

    while start:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        helmet_detected = False
        no_helmet_detected = False
        zoom_image = None

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name.lower() == "helmet":
                helmet_detected = True

            if "no" in class_name.lower():
                no_helmet_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                zoom_image = frame[y1:y2, x1:x2]

        frame_placeholder.image(annotated_frame, channels="BGR")

        # Traffic Logic
        if no_helmet_detected:
            signal_placeholder.markdown(
                "<h1 style='color:red;'>🔴 STOP</h1>",
                unsafe_allow_html=True,
            )

            if zoom_image is not None:
                zoom_placeholder.image(zoom_image, channels="BGR", caption="Violator")

                # Save violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"violations/violation_{timestamp}.jpg"
                cv2.imwrite(filename, zoom_image)

        elif helmet_detected:
            signal_placeholder.markdown(
                "<h1 style='color:green;'>🟢 GO</h1>",
                unsafe_allow_html=True,
            )

        time.sleep(0.03)

    cap.release()