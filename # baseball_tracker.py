# baseball_tracker.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import sys

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True
)

# Camera setup
camera_index = 0  # update as needed
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Could not open camera with index {camera_index}.")
    sys.exit(1)

# Load the trained model
try:
    model = tf.keras.models.load_model('models/baseball_model.h5')
except Exception as e:
    print("Error loading model 'models/baseball_model.h5':", e)
    print("Make sure the path is correct and the model is a Keras .h5 or SavedModel.")
    model = None  # allow running without model for debugging

def preprocess_frame(frame):
    # frame expected as BGR crop
    try:
        resized = cv2.resize(frame, (64, 64))
    except Exception:
        # if crop is too small, return None
        return None
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def filter_circles(circles, min_radius=15, max_radius=60):
    filtered = []
    if circles is not None:
        # cv2.HoughCircles returns shape (1, N, 3)
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if min_radius <= r <= max_radius:
                filtered.append((x, y, r))
    return filtered

# strike-zone direction locking
strike_zone_direction = None
last_direction_update_time = 0
direction_hold_time = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip if you want mirror view; optional: frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=70
    )

    filtered_circles = filter_circles(circles)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        h, w, _ = frame.shape

        # get landmarks safely
        L = results.pose_landmarks.landmark
        try:
            left_shoulder = L[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = L[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = L[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = L[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = L[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = L[mp_pose.PoseLandmark.RIGHT_KNEE]
        except Exception:
            left_shoulder = right_shoulder = left_hip = right_hip = left_knee = right_knee = None

        if left_shoulder and right_shoulder and left_hip and right_hip and left_knee and right_knee:
            shoulder_mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            shoulder_mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)

            hip_mid_x = int((left_hip.x + right_hip.x) / 2 * w)
            hip_mid_y = int((left_hip.y + right_hip.y) / 2 * h)

            knee_mid_y = int((left_knee.y + right_knee.y) / 2 * h)

            # vertical limits (shoulder->knee)
            scale_factor = 50  # tune as needed
            upper_limit = max(0, int((shoulder_mid_y + hip_mid_y) / 2) - scale_factor)
            lower_limit = min(h - 1, knee_mid_y)

            # determine stance direction candidate
            current_time = time.time()
            current_direction = 1 if left_shoulder.x < right_shoulder.x else -1

            if strike_zone_direction is None:
                strike_zone_direction = current_direction
                last_direction_update_time = current_time
            else:
                if current_direction != strike_zone_direction:
                    if (current_time - last_direction_update_time) > direction_hold_time:
                        strike_zone_direction = current_direction
                        last_direction_update_time = current_time
                else:
                    last_direction_update_time = current_time

            strike_zone_shift = int(100 * strike_zone_direction)  # pixels, tune
            strike_zone_width = 150  # tune
            strike_zone_center_x = shoulder_mid_x + strike_zone_shift
            strike_zone_left = max(0, strike_zone_center_x - strike_zone_width // 2)
            strike_zone_right = min(w - 1, strike_zone_center_x + strike_zone_width // 2)

            # draw strike zone
            cv2.rectangle(frame, (strike_zone_left, upper_limit), (strike_zone_right, lower_limit), (0, 255, 0), 2)

            for (x, y, r) in filtered_circles:
                if strike_zone_left < x < strike_zone_right and upper_limit < y < lower_limit:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                    cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                    # safe crop bounds
                    y1 = max(0, y - r)
                    y2 = min(h, y + r)
                    x1 = max(0, x - r)
                    x2 = min(w, x + r)
                    crop_img = frame[y1:y2, x1:x2]

                    if crop_img.size != 0 and model is not None:
                        crop_input = preprocess_frame(crop_img)
                        if crop_input is not None:
                            try:
                                prediction = model.predict(crop_input)
                                predicted_class = int(np.argmax(prediction, axis=1)[0])
                                confidence = float(prediction[0][predicted_class])
                                label = "Baseball" if predicted_class == 0 else "Not Baseball"
                                label_confidence = f"{label}: {confidence*100:.1f}%"

                                if confidence > 0.6:
                                    cv2.putText(frame, label_confidence, (x - r, y - r - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                else:
                                    cv2.putText(frame, "Low Confidence", (x - r, y - r - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            except Exception as e:
                                # prediction failure should not crash loop
                                cv2.putText(frame, "Prediction error", (x - r, y - r - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show frame
    cv2.imshow('Live Video - Baseball Detection in Strike Zone', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()