import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

WIDTH = 1200
HEIGHT = 678

cap = cv2.VideoCapture(0)

def find_position(results):
    landmark_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, pt in enumerate(hand_landmarks.landmark):
                x = int(pt.x * WIDTH)
                y = int(pt.y * HEIGHT)
                landmark_list.append((x, y))
    return landmark_list

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
        flipped_frame = cv2.flip(frame_resized, 1)
        canvas = np.zeros_like(flipped_frame)

        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        rgb_flipped_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        landmarks = find_position(results)

        if len(landmarks) > 8:  # Ensure there are enough landmarks
            x, y = landmarks[8]  # Index 8 corresponds to the tip of the index finger
            cv2.circle(canvas, (x, y), 10, (255, 0, 0), -1)

        flipped_canvas = cv2.flip(canvas, 1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frames
        cv2.imshow("Original Frame", frame_resized)
        cv2.imshow("Flipped Canvas", flipped_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
