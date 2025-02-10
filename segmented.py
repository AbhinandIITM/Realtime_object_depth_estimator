import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(3)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Smoothing factor for fingertip positions
SMOOTHING_FACTOR = 0.3
last_positions = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            # Index fingertip (primary pointer)
            index_fingertip = [5, 8]  # Index finger start and tip
            start_idx, end_idx = index_fingertip
            start = np.array([hand_landmarks.landmark[start_idx].x, hand_landmarks.landmark[start_idx].y])
            end = np.array([hand_landmarks.landmark[end_idx].x, hand_landmarks.landmark[end_idx].y])

            # Compute direction vector and extend the fingertip position
            vector = end - start
            vector /= np.linalg.norm(vector)
            extended_tip = end + vector * 0.1  # Extend by a small factor

            # Convert to pixel coordinates
            extended_tip_pixel = (int(extended_tip[0] * w), int(extended_tip[1] * h))

            # Apply smoothing
            if start_idx not in last_positions:
                last_positions[start_idx] = extended_tip_pixel
            smoothed_tip = (
                int(last_positions[start_idx][0] * (1 - SMOOTHING_FACTOR) + extended_tip_pixel[0] * SMOOTHING_FACTOR),
                int(last_positions[start_idx][1] * (1 - SMOOTHING_FACTOR) + extended_tip_pixel[1] * SMOOTHING_FACTOR)
            )
            last_positions[start_idx] = smoothed_tip

            x, y = smoothed_tip

            # Define a small ROI around the fingertip
            roi_size = 50
            x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
            x2, y2 = min(w, x + roi_size), min(h, y + roi_size)
            roi = frame[y1:y2, x1:x2]

            # Skip GrabCut if ROI is too small
            if roi.shape[0] < 5 or roi.shape[1] < 5:
                continue

            # GrabCut Segmentation
            mask = np.zeros(roi.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (2, 2, roi.shape[1] - 4, roi.shape[0] - 4)  # Avoid edges
            cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Extract the object by applying the mask
            roi_segmented = roi * mask2[:, :, np.newaxis]

            # Find contours on the segmented object
            gray = cv2.cvtColor(roi_segmented, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contour around the object
            for cnt in contours:
                cv2.drawContours(frame[y1:y2, x1:x2], [cnt], -1, (0, 0, 255), 2)  # Red boundary

            # Draw a green rectangle around the ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Object Segmentation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
