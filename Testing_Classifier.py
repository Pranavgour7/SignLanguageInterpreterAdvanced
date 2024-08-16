import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the model and PCA (if PCA was used)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
pca = model_dict.get('pca', None)  # Load PCA if it exists

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set to real-time mode
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define label mapping
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
               10: 'L', 11: 'M', 12: 'O', 13: 'R', 14: 'U', 15: 'V', 16: 'W', 17: 'Y', 18: 'Hello',
               19: 'Okay', 20: 'Bathroom', 21: '1', 22: '3', 23: '4', 24: '7'}

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # Define bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Ensure data is padded or truncated to match the PCA input length (42 features)
        fixed_length = 24  # Adjust to your model's input size
        if len(data_aux) < fixed_length:
            data_aux += [0] * (fixed_length - len(data_aux))  # Pad with zeros
        elif len(data_aux) > fixed_length:
            data_aux = data_aux[:fixed_length]  # Truncate to fixed length

        # Apply PCA if it was used during training
        if pca is not None:
            data_aux = pca.transform([data_aux])[0]

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])

        # Handle both integer and string predictions
        predicted_index = prediction[0]
        if isinstance(predicted_index, str):
            predicted_character = predicted_index  # Use the string directly
        else:
            # Check if the predicted index is in the labels_dict
            predicted_character = labels_dict.get(int(predicted_index), "Unknown")

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                    cv2.LINE_AA)

    # Display frame
    cv2.imshow('frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
