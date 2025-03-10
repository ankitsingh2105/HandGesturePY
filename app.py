import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Smoothing variables
prev_x, prev_y = 0, 0
alpha = 0.6  # Smoothing factor

# Function to find an active camera
def find_active_camera():
    for i in range(5):  # Try indexes 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Using camera at index {i}")
            return cap
        cap.release()
    print("❌ No active camera found!")
    return None

# Find an active camera
cap = find_active_camera()

if cap is None:
    exit()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("❌ Failed to capture frame!")
        break

    img = cv2.flip(img, 1)  # Mirror the image
    h, w, _ = img.shape  # Get frame dimensions

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Ring Finger Tip coordinates (for movement)
            ring_tip = hand_landmarks.landmark[16]
            x, y = int(ring_tip.x * w), int(ring_tip.y * h)

            # Convert to screen coordinates with smoothing
            screen_x = int((ring_tip.x * screen_w) * alpha + prev_x * (1 - alpha))
            screen_y = int((ring_tip.y * screen_h) * alpha + prev_y * (1 - alpha))

            # Update previous position
            prev_x, prev_y = screen_x, screen_y

            # Move the cursor smoothly
            pyautogui.moveTo(screen_x, screen_y)

            # Get Finger Tips
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            little_tip = hand_landmarks.landmark[20]

            # Convert finger positions to screen space
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
            little_x, little_y = int(little_tip.x * w), int(little_tip.y * h)

            # Compute distances
            thumb_little_dist = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([little_x, little_y]))
            thumb_index_dist = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))
            thumb_middle_dist = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([middle_x, middle_y]))

            # Scroll Down (Thumb + Little Finger Pinch)
            if thumb_little_dist < 30:
                pyautogui.scroll(-5)  # Scroll down

            # Scroll Up (Thumb + Middle + Index Pinch)
            if thumb_index_dist < 30 and thumb_middle_dist < 30:
                pyautogui.scroll(5)  # Scroll up

            # Left Click (Thumb + Index Pinch)
            if thumb_index_dist < 30 and thumb_middle_dist >= 30:
                pyautogui.click()

            # Right Click (Thumb + Middle Pinch)
            if thumb_middle_dist < 30 and thumb_index_dist >= 30:
                pyautogui.rightClick()

    # Show the webcam feed
    cv2.imshow("Hand Tracking Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()
