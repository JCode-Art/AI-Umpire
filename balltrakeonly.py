from tracknet import TrackNet  # the tennis model code
import cv2
import numpy as np

# Load your retrained weights
model = TrackNet()
model.load_weights('WeightsTracknet/baseball_weights.h5')

cap = cv2.VideoCapture('baseball_game.mp4')

while True:
    ret, frame = cap.read()
    if not ret: break

    # Preprocess: resize to model input size (e.g., 640×360)
    inp = cv2.resize(frame, (640, 360)) / 255.0
    heatmap = model.predict(np.expand_dims(inp, 0))[0, :, :, 0]

    # Find ball location
    y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    cv2.circle(frame, (int(x * frame.shape[1] / heatmap.shape[1]), int(y * frame.shape[0] / heatmap.shape[0])), 10, (0,0,255), 4)

    cv2.imshow('Baseball Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
