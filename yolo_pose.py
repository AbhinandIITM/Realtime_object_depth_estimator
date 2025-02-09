import ultralytics
from ultralytics import YOLO

# Load YOLO pose estimation model
pose_model = YOLO('yolo11m-pose.pt')


results = pose_model.track(source=0, show=True, save=True)

# import cv2
# cap = cv2.VideoCapture(2)
# while True:
#     ret,frame = cap.read()
#     if not ret:
#         continue
#     cv2.imshow("Camera Feed", frame)  # Correct usage

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()