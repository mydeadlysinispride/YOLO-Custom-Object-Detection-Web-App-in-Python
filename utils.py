from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLO11 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLO11): An instance of the `YOLO11` class containing the YOLO11 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLO11 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True
                   )


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

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
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
            while not stop:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image
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