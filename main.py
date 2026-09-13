import cv2
from src.pose_detector import PoseDetector

def main():
    video_source = r"D:\DATN\Physical_Therapy_Model\data\11.mp4"
    cap = cv2.VideoCapture(video_source)

    detector = PoseDetector(model_complexity=1)

    print("Hệ thống đang khởi động. Nhấn 'q' để thoát.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Không thể đọc frame hoặc video đã kết thúc.")
            break

        frame = detector.find_pose(frame, draw=True)

        landmarks = detector.get_landmarks(frame, normalize_pixel=False)

        frame, current_fps = detector.draw_fps(frame)

        cv2.imshow("Physiotheraphy Pose Estimation", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()