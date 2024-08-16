import mediapipe as mp
import os
import cv2
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

# Map numeric directory names to corresponding alphabet labels
alphabet_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
               10: 'L', 11: 'M', 12: 'O', 13: 'R', 14: 'U', 15: 'V', 16: 'W', 17: 'Y', 18: 'Hello',
               19: 'Okay', 20: 'Bathroom', 21: '1', 22: '3', 23: '4', 24: '7'}

for dir_ in os.listdir(DATA_DIR):
    dir_index = int(dir_)  # Assuming directory names are 0-25
    label = alphabet_labels[dir_index]  # Map to corresponding alphabet

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:  # Check if landmarks are detected
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if data_aux:  # Ensure data is not empty
                data.append(data_aux)
                labels.append(label)

# Save data and labels to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
