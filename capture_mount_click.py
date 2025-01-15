import cv2

# Danh sách lưu các điểm được chọn
points = []

# Hàm callback để chọn điểm bằng chuột
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Selected point: {x}, {y}")

        # Vẽ điểm vừa chọn lên frame
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Video", frame)

# Đọc video
cap = cv2.VideoCapture(".\\test_data\\DangerZoneCheck.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame về 720p (16:9 ratio)
    frame = cv2.resize(frame, (720, int(720 * (9 / 16))))

    # Hiển thị frame và chờ chọn điểm
    cv2.imshow("Video", frame)

    # Thiết lập callback để chọn điểm bằng chuột
    cv2.setMouseCallback("Video", select_point)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng các cửa sổ
cap.release()
cv2.destroyAllWindows()

# In ra danh sách các điểm đã chọn
print("Selected points:", points)
