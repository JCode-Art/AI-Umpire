import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=True)

# Camera setup
camera_index = 1  # Update based on your camera setup
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Could not open camera with index {camera_index}.")
    exit()

# Load the trained model
model = tf.keras.models.load_model('models/baseball_model.h5')

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

# Variables to hold strike zone direction and timing
strike_zone_direction = None  # Can be 1 (right shift) or -1 (left shift)
last_direction_update_time = 0
direction_hold_time = 5  # seconds to hold strike zone direction before allowing change

# Ball tracking variables
ball_in_zone = False
ball_entered_time = None
ball_exited_time = None
ball_was_hit = False
last_ball_center = None
last_ball_time = None

# Helper to calculate speed in pixels per second
def calculate_speed(last_pos, last_time, current_pos, current_time):
    if last_pos is None or last_time is None:
        return 0
    dist = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
    dt = current_time - last_time
    if dt == 0:
        return 0
    return dist / dt

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = pose.process(frame)
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

        # Get key points normalized (x,y)
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

        # Midpoints in pixel coordinates
        shoulder_mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
        shoulder_mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)

        hip_mid_x = int((left_hip.x + right_hip.x) / 2 * w)
        hip_mid_y = int((left_hip.y + right_hip.y) / 2 * h)

        knee_mid_y = int((left_knee.y + right_knee.y) / 2 * h)

        # Vertical strike zone limits (shoulder to knee range + buffer)
        scale_factor = 50  # vertical buffer in pixels (~6 inches)
        upper_limit = int((shoulder_mid_y + hip_mid_y) / 2) - scale_factor
        lower_limit = knee_mid_y

        # Determine batter stance direction and lock it
        current_time = time.time()

        # FIXED: flip direction logic to match actual pose x coords
        # If left_shoulder.x > right_shoulder.x, batter facing right (right-handed), shift strike zone left (-1)
        current_direction = -1 if left_shoulder.x > right_shoulder.x else 1

        if strike_zone_direction is None:
            # First detection, lock the direction
            strike_zone_direction = current_direction
            last_direction_update_time = current_time
        else:
            # Hold current direction for a minimum time before switching
            if current_direction != strike_zone_direction:
                if (current_time - last_direction_update_time) > direction_hold_time:
                    strike_zone_direction = current_direction
                    last_direction_update_time = current_time
            else:
                # Same direction, update last seen time
                last_direction_update_time = current_time

        # Debug print to check shoulder positions and strike zone direction
        print(f"Left shoulder x: {left_shoulder.x:.2f}, Right shoulder x: {right_shoulder.x:.2f}, Strike zone direction: {strike_zone_direction}")

        # Apply strike zone shift based on locked direction
        strike_zone_shift = 100 * strike_zone_direction

        # Horizontal strike zone width (typical plate width ~17 inches, scaled)
        strike_zone_width = 150  # you can tune this

        # Center X of strike zone shifted forward by strike_zone_shift
        strike_zone_center_x = shoulder_mid_x + strike_zone_shift

        strike_zone_left = strike_zone_center_x - strike_zone_width // 2
        strike_zone_right = strike_zone_center_x + strike_zone_width // 2

        # Draw strike zone rectangle
        cv2.rectangle(frame, (strike_zone_left, upper_limit), (strike_zone_right, lower_limit), (0, 255, 0), 2)

        # Track ball entry/exit and speed
        ball_detected_this_frame = False

        for (x, y, r) in filtered_circles:
            ball_detected_this_frame = True

            # Check if circle center is inside strike zone
            in_zone_now = strike_zone_left < x < strike_zone_right and upper_limit < y < lower_limit

            # Calculate speed of the ball center
            current_center = (x, y)
            speed = calculate_speed(last_ball_center, last_ball_time, current_center, current_time)
            speed_threshold = 50  # pixels per second, adjust as needed

            if in_zone_now and not ball_in_zone:
                ball_in_zone = True
                ball_entered_time = current_time
                ball_was_hit = False  # reset hit flag on new entry
                print("[INFO] Ball entered strike zone")

            elif not in_zone_now and ball_in_zone:
                ball_in_zone = False
                ball_exited_time = current_time

                # Decide if ball was hit or not based on speed
                if speed > speed_threshold:
                    print("[RESULT] Strike! Ball passed cleanly through strike zone.")
                else:
                    ball_was_hit = True
                    print("[RESULT] Ball was hit (possible foul or fair).")

            last_ball_center = current_center
            last_ball_time = current_time

            # Draw circle and rectangle on ball
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

        # If no ball detected this frame, reset last_ball_center to avoid false speed calc
        if not ball_detected_this_frame:
            last_ball_center = None
            last_ball_time = None

    cv2.imshow('Live Video - Baseball Detection in Strike Zone', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
