from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np
import time
class_list = ['Full PPE', 'Incomplete PPE', 'No PPE']

def _display_detected_frames(conf, model, st_frame, image, area=None, fps=None):
    """
    Display the detected objects on a video frame using the YOLO11 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLO11): An instance of the `YOLO11` class containing the YOLO11 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :param area (list): Optional list of points representing the detection area.
    :param fps (float): Frames per second.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLO11 model
    res = model.predict(image, conf=conf)

    # Check if the user has enabled zone creation
    if area:
        area_np = np.array(area, np.int32)
        cv2.polylines(image, [area_np], isClosed=True, color=(255, 255, 0), thickness=2)
    
    alarm_triggered = False

    # Iterate through the detected boxes
    for box in res[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = class_list[class_id]

        # If area is provided, check if the detected object is inside the zone
        if area:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            if cv2.pointPolygonTest(area_np, (center_x, center_y), False) >= 0:
                if class_name in ['Incomplete PPE', 'No PPE']:
                    alarm_triggered = True
                    cv2.putText(image, "ALARM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # Without zone, just check for violations and display alarm if found
            if class_name in ['Incomplete PPE', 'No PPE']:
                alarm_triggered = True
                cv2.putText(image, "ALARM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Draw bounding box and label
        color = (0, 255, 0) if class_name == 'Full PPE' else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(image, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # If no alarm is triggered, display "OK"
    if not alarm_triggered:
        cv2.putText(image, "OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display FPS in the top-right corner
    if fps is not None:
        cv2.putText(image, f"FPS: {fps:.2f}", (image.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Plot the detected objects on the video frame
    st_frame.image(image, caption='Detected Video', channels="BGR", use_container_width=True)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLO11 model
    :param model: An instance of the `YOLO11` class containing the YOLO11 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    use_zone = st.sidebar.checkbox("Enable Zone Detection", value=True)
    area = []

    if use_zone:
        try:
            with open('dangerzone.txt', 'r') as file:
                default_area = file.read().strip()
        except FileNotFoundError:
            default_area = "353,391;446,390;443,180;364,180"

        area_input = st.text_area(
            "Enter detection area coordinates (format: x1,y1;x2,y2;x3,y3;x4,y4)",
            value=default_area
        )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st_frame = st.empty()

                    prev_time = time.time()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            current_time = time.time()
                            fps = 1 / (current_time - prev_time)
                            prev_time = current_time

                            if area_input:
                                area = [tuple(map(int, point.split(','))) for point in area_input.split(';')]
                            _display_detected_frames(conf, model, st_frame, image, area, fps)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLO11 model
    :param model: An instance of the `YOLO11` class containing the YOLO11 model.
    :return: None
    """
    run_button = st.empty()
    stop_button = st.empty()
    
    run = run_button.button("Run")
    stop = stop_button.button("Stop running")

    if run:
        try:
            vid_cap = cv2.VideoCapture(0)  # local camera
            st_frame = st.empty()
            prev_time = time.time()
            while not stop:
                success, image = vid_cap.read()
                if success:
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time)
                    prev_time = current_time

                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        fps=fps
                    )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
        finally:
            vid_cap.release()
            run_button.empty()
            stop_button.empty()



@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLO11 model
    :param model: An instance of the `YOLO11` class containing the YOLO11 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_container_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_container_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


