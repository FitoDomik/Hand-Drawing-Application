import cv2
import numpy as np
import mediapipe as mp
from collections import deque
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_selfie = mp.solutions.selfie_segmentation
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
selfie_segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
cap = cv2.VideoCapture(0)
canvas = None
prev_point = None
brush_color = (0, 255, 0)  
colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255),
}
palette_y = 10
palette_h = 50
palette_w = 80
finger_state_buffer = deque(maxlen=5)
def draw_palette(frame):
    for i, (name, color) in enumerate(colors.items()):
        x1 = i * palette_w
        x2 = x1 + palette_w
        cv2.rectangle(frame, (x1, palette_y), (x2, palette_y + palette_h), color, -1)
        cv2.putText(frame, name, (x1 + 5, palette_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return frame
def check_palette_hover(x, y):
    if y < palette_y or y > palette_y + palette_h:
        return None
    index = x // palette_w
    if index < len(colors):
        return list(colors.values())[index]
    return None
def fingers_up(landmarks):
    """
    Return list of bools for [thumb, index, middle, ring, pinky].
    True if finger is up.
    """
    if landmarks is None:
        return [False]*5
    fingers = []
    if landmarks[4].x < landmarks[3].x:
        fingers.append(True)
    else:
        fingers.append(False)
    for tip_id, pip_id in zip([8,12,16,20],[6,10,14,18]):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers
print("Starting drawing app. Press 'q' to quit, 'c' to clear.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_result = selfie_segmentation.process(rgb)
    mask = seg_result.segmentation_mask
    condition = mask > 0.6
    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    frame_blurred_bg = np.where(condition[:, :, None], frame, blurred)
    results = hands.process(rgb)
    finger_states = [False]*5
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        finger_states = fingers_up(landmarks)
        mp_drawing.draw_landmarks(frame_blurred_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        index_x = int(landmarks[8].x * w)
        index_y = int(landmarks[8].y * h)
        cv2.circle(frame_blurred_bg, (index_x, index_y), 8, (0, 0, 255), -1)
        finger_state_buffer.append(finger_states)
    else:
        finger_state_buffer.append([False]*5)
        prev_point = None
    if len(finger_state_buffer) == finger_state_buffer.maxlen:
        counts = np.sum(finger_state_buffer, axis=0)
        smoothed_states = counts > (finger_state_buffer.maxlen // 2)
    else:
        smoothed_states = finger_states
    if results.multi_hand_landmarks:
        hovered_color = check_palette_hover(index_x, index_y)
        if hovered_color is not None:
            brush_color = hovered_color
    if smoothed_states[1] and smoothed_states[2] and not any([smoothed_states[0], smoothed_states[3], smoothed_states[4]]):
        if prev_point:
            cv2.line(canvas, prev_point, (index_x, index_y), brush_color, 5)
        prev_point = (index_x, index_y)
    else:
        prev_point = None
    output = cv2.add(frame_blurred_bg, canvas)
    output = draw_palette(output)
    cv2.imshow("Hand Drawing with Background Blur and Color Palette", output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
cap.release()
cv2.destroyAllWindows()
