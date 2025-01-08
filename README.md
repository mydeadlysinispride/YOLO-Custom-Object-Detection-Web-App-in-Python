# YOLO-Custom-Object-Detection-Web-App-in-Python
Project Object Detection và Object Tracking cho custom dataset, triển khai thành Web app

## Objective:
- Collect Data & Labelling
- Train YOLOv8-11 model
- Get Predictions

## Features
- Feature1: Object detection task.
- Feature2: Multiple detection models. `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- Feature3: Multiple input formats. `Image`, `Video`, `Webcam`


## Cấu hình để tải dữ liệu từ Roboflow

Để sử dụng tập dữ liệu đã được chuẩn bị sẵn trên Roboflow, chúng ta cần cấu hình hệ thống để tải dữ liệu từ nền tảng này.

### Chỉnh sửa file config.yaml

1. **Mở file:** Sử dụng trình soạn thảo văn bản để mở file `config/config.yaml`.
2. **Tìm đến dòng:** Tìm đến dòng chứa tùy chọn `roboflow_option`.
3. **Đặt giá trị:** Đặt giá trị của tùy chọn này thành `True`.

```yaml
# config/config.yaml
roboflow_option: True  # Cấu hình để tải dữ liệu từ Roboflow. 
                       # Khi đặt thành True, hệ thống sẽ tải dữ liệu từ Roboflow 
                       # thay vì từ nguồn dữ liệu khác.
```

## Installation
### Create a new conda environment
```commandline
# create
conda create -n yolov8-streamlit python=3.8 -y

# activate
conda activate yolov8-streamlit
```

### Create python environment
```
#create
python -m venv env

#activate
env\Scripts\activate
```

### Clone repository
```commandline
git clone https://github.com/JackDance/YOLOv8-streamlit-app
```

### Install packages
```commandline
# yolov8 dependencies
pip install ultralytics

# Streamlit dependencies
pip install streamlit
```
### Download Pre-trained YOLOv8 Detection Weights
Create a directory named `weights` and create a subdirectory named `detection` and save the downloaded YOLOv8 object detection weights inside this directory. The weight files can be downloaded from the table below.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |


## Run
```commandline
streamlit run app.py
```
Then will start the Streamlit server and open your web browser to the default Streamlit page automatically.



