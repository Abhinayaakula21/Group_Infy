# Import required libraries
import cv2                      # OpenCV for video capture
import mediapipe as mp          # MediaPipe for hand detection
import math                     # For calculating distance

# ---------------- STEP 1: Hand Detection Setup ----------------

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands

# Drawing utility to draw landmarks
mp_draw = mp.solutions.drawing_utils

# Create Hands object with detection settings
hands = mp_hands.Hands(
    static_image_mode=False,        # Detect in video stream
    max_num_hands=1,                # Detect only one hand
    model_complexity=0,             # Faster but slightly less accurate
    min_detection_confidence=0.6,   # Minimum confidence for detection
    min_tracking_confidence=0.6     # Minimum confidence for tracking
)

# Start webcam
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 640)   # Width
cap.set(4, 480)   # Height

# Main loop
while True:
    success, frame = cap.read()     # Capture frame

    if not success:
        print("Failed to access camera")
        break

    # Convert BGR to RGB (MediaPipe needs RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb)

    # If hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand landmarks on frame
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get frame dimensions
            h, w, _ = frame.shape

            # ---------------- STEP 2: Distance Between Thumb & Index ----------------

            # Landmark 4 = Thumb tip
            thumb = hand_landmarks.landmark[4]

            # Landmark 8 = Index finger tip
            index = hand_landmarks.landmark[8]

            # Convert normalized coordinates to pixel values
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Calculate distance using Pythagoras formula
            distance = math.hypot(x2 - x1, y2 - y1)

            # Gesture classification based on distance
            if distance < 30:
                gesture = "SELECT"
            elif distance < 80:
                gesture = "HOLD"
            else:
                gesture = "RELEASE"

            # Draw circles on finger tips
            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)

            # Draw line between thumb and index
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Display distance
            cv2.putText(frame, f"Distance: {int(distance)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2)

            # Display detected gesture
            cv2.putText(frame, f"Gesture: {gesture}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2)

    # Show output window
    cv2.imshow("Hand Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()