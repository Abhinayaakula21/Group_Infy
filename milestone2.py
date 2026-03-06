# Import OpenCV library for camera and image processing
import cv2

# Import MediaPipe for hand tracking
import mediapipe as mp

# Import specific hands solution from MediaPipe
import mediapipe.solutions.hands as mp_hands

# Import drawing utilities to draw hand landmarks
import mediapipe.solutions.drawing_utils as mp_draw

# Import math for distance calculation
import math


# ===================== STEP 1: HAND DETECTION SETUP =====================

# Initialize the hand detection model
hands = mp_hands.Hands(
    static_image_mode=False,          # False means continuous video detection
    max_num_hands=1,                  # Detect only one hand
    model_complexity=0,               # Faster model (0 = lightweight)
    min_detection_confidence=0.6,     # Minimum confidence for detection
    min_tracking_confidence=0.6       # Minimum confidence for tracking
)

# Start webcam (0 means default camera)
cap = cv2.VideoCapture(0)

# Set camera width
cap.set(3, 640)

# Set camera height
cap.set(4, 480)


# Infinite loop to read camera frames
while True:

    # Read frame from camera
    success, frame = cap.read()

    # If frame is not captured properly, exit loop
    if not success:
        break

    # Convert BGR image to RGB (MediaPipe requires RGB format)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb)

    # If hand landmarks are detected
    if result.multi_hand_landmarks:

        # Loop through each detected hand
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand landmarks and connections on the frame
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get frame height and width
            h, w, _ = frame.shape


            # ===================== STEP 2: DISTANCE CALCULATION =====================

            # Get thumb tip landmark (ID = 4)
            thumb = hand_landmarks.landmark[4]

            # Get index finger tip landmark (ID = 8)
            index = hand_landmarks.landmark[8]

            # Convert normalized coordinates to pixel coordinates
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Calculate Euclidean distance between thumb and index finger
            distance = math.hypot(x2 - x1, y2 - y1)

            # Draw circle on thumb tip
            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)

            # Draw circle on index tip
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)

            # Draw line between thumb and index finger
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Display distance value on screen
            cv2.putText(frame, f"Distance: {int(distance)}",
                        (20, 40),                     # Position of text
                        cv2.FONT_HERSHEY_SIMPLEX,     # Font style
                        0.8,                          # Font size
                        (0, 255, 255),                # Text color
                        2)                            # Thickness


    # Show the final output window
    cv2.imshow("Hand Detection", frame)

    # Press 'q' to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release camera after loop ends
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Close MediaPipe hands model
hands.close()