import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Load model và danh sách class
model = YOLO('.\\weights\\detection\\yolo11m_openvino_model')
class_list = ['Full PPE', 'Incomplete PPE', 'No PPE']

# Load video
cap = cv2.VideoCapture('.\\test_data\\DangerZoneCheck.mp4')

# Khu vực phát hiện
area1 = [(353, 391), (446, 390), (443, 180), (364, 180)]

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame (nếu cần)
    frame = cv2.resize(frame, (720, int(720 * (9 / 16))))

    # Phát hiện object bằng YOLO
    results = model(frame, stream=True)

    # Cờ để kiểm tra xem có phát hiện class "Incomplete PPE" hoặc "No PPE" hay không
    alarm_triggered = False

    for result in results:
        for box in result.boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2)

            class_id = int(box.cls[0])
            class_name = class_list[class_id]

            # Vẽ bounding box lên frame
            color = (0, 255, 0) if class_name == 'Full PPE' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=1)
            cv2.putText(frame, f'{class_name} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Lấy tâm của bounding box
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Kiểm tra xem tâm của bounding box có nằm trong khu vực area1 không
            if cv2.pointPolygonTest(np.array(area1, np.int32), (x_center, y_center), False) >= 0:
                if class_name in ['Incomplete PPE', 'No PPE']:
                    alarm_triggered = True

    # Vẽ khu vực phát hiện (ROI)
    cv2.polylines(frame, [np.array(area1, np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)

    # Hiển thị thông báo ALARM hoặc OK
    if alarm_triggered:
        cv2.putText(frame, "ALARM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Hiển thị frame
    cv2.imshow("Video", frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
