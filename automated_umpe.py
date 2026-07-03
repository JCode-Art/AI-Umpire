import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# ---------------------------
# Initialize MediaPipe Pose
# ---------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True
)

# ---------------------------
# Camera setup
# ---------------------------
camera_index = 1  # Change if wrong camera
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Could not open camera with index {camera_index}.")
    exit()

# ---------------------------
# Load trained baseball model
# ---------------------------
model = tf.keras.models.load_model('models/baseball_model.h5')
print("Model loaded: models/baseball_model.h5")

# ---------------------------
# Helper functions
# ---------------------------
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def filter_circles(circles, min_radius=20, max_radius=40):
    filtered_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if min_radius <= r <= max_radius:
                filtered_circles.append((x, y, r))
    return filtered_circles

# Fixed swing detection
def detect_swing(landmarks, last_left_wrist, last_right_wrist, dt):
    # Safe landmark access
    def safe_landmark(landmarks, idx):
        if landmarks and len(landmarks) > idx:
            return landmarks[idx]
        return None

    lw = safe_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = safe_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

    swing_detected = False
    if lw and rw and last_left_wrist and last_right_wrist:
        # Wrist movement speed
        lw_speed = abs(lw.x - last_left_wrist.x) / dt
        rw_speed = abs(rw.x - last_right_wrist.x) / dt

        # Threshold to detect swing
        if lw_speed > 0.05 or rw_speed > 0.05:
            swing_detected = True

    return swing_detected, lw, rw

# ---------------------------
# Variables for strike zone logic
# ---------------------------
strike_zone_direction = None  # 1 (right shift) or -1 (left shift)
last_direction_update_time = 0
direction_hold_time = 5  # seconds to lock stance

last_left_wrist = None
last_right_wrist = None
last_time = time.time()

print("Starting automated umpire (single-camera). Press 'q' to quit, 'r' to reset counts.")

# ---------------------------
# Main loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Timing for swing detection
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # Pose estimation
    results = pose.process(frame)

    # Ball detection (Hough Circles)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=15, maxRadius=40
    )
    filtered_circles = filter_circles(circles)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        h, w, _ = frame.shape

        # Key landmarks
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

        # Midpoints
        shoulder_mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
        shoulder_mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
        hip_mid_x = int((left_hip.x + right_hip.x) / 2 * w)
        hip_mid_y = int((left_hip.y + right_hip.y) / 2 * h)
        knee_mid_y = int((left_knee.y + right_knee.y) / 2 * h)

        # Strike zone vertical limits
        scale_factor = 50
        upper_limit = int((shoulder_mid_y + hip_mid_y) / 2) - scale_factor
        lower_limit = knee_mid_y

        # Batter stance detection
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

        # Apply horizontal strike zone shift
        strike_zone_shift = 100 * strike_zone_direction
        strike_zone_width = 150
        strike_zone_center_x = shoulder_mid_x + strike_zone_shift
        strike_zone_left = strike_zone_center_x - strike_zone_width // 2
        strike_zone_right = strike_zone_center_x + strike_zone_width // 2

        # Draw strike zone
        cv2.rectangle(frame, (strike_zone_left, upper_limit), (strike_zone_right, lower_limit), (0, 255, 0), 2)

        # Detect swing
        swing_detected, lw, rw = detect_swing(
            list(results.pose_landmarks.landmark),
            last_left_wrist, last_right_wrist, dt
        )

        if swing_detected:
            cv2.putText(frame, "Swing Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        last_left_wrist = lw
        last_right_wrist = rw

        # Ball detection inside strike zone
        for (x, y, r) in filtered_circles:
            if strike_zone_left < x < strike_zone_right and upper_limit < y < lower_limit:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                crop_img = frame[max(y - r, 0):y + r, max(x - r, 0):x + r]
                if crop_img.size != 0:
                    crop_input = preprocess_frame(crop_img)
                    prediction = model.predict(crop_input)
                    predicted_class = prediction.argmax(axis=1)[0]
                    confidence = prediction[0][predicted_class]
                    label = "Baseball" if predicted_class == 0 else "Not Baseball"
                    label_confidence = f"{label}: {confidence*100:.2f}%"

                    if confidence > 0.6:
                        cv2.putText(frame, label_confidence, (x - r, y - r - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Low Confidence", (x - r, y - r - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show frame
    cv2.imshow('Live Video - Baseball Detection & Umpire', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
