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
alpha = 0.6  # Smoothing factor (0 = no smoothing, 1 = instant movement)

# Function to find an active camera
def find_active_camera():
    for i in range(5):  # Try indexes 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Using camera at index {i}")
            return cap  # Return the first active camera
        cap.release()
    print("❌ No active camera found!")
    return None

# Find an active camera
cap = find_active_camera()

if cap is None:
    exit()  # Stop execution if no camera is found

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

            # Get Middle Finger Tip (for scrolling)
            middle_tip = hand_landmarks.landmark[12]
            mid_x, mid_y = int(middle_tip.x * w), int(middle_tip.y * h)

            # Scroll based on hand movement
            if abs(y - mid_y) > 50:  # Adjust sensitivity
                if y < mid_y:  # Hand moving up
                    pyautogui.scroll(5)
                else:  # Hand moving down
                    pyautogui.scroll(-5)

            # Click Detection
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Convert finger positions to screen space
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Distance between thumb and index (for left click)
            left_click_distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))

            # Get Middle Finger Tip (for right click)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

            # Distance between thumb and middle (for right click)
            right_click_distance = np.linalg.norm(np.array([middle_x, middle_y]) - np.array([thumb_x, thumb_y]))

            if left_click_distance < 30:  # If thumb and index are close together
                pyautogui.click()  # Left click

            if right_click_distance < 30:  # If thumb and middle are close together
                pyautogui.rightClick()  # Right click

    # Show the webcam feed
    cv2.imshow("Hand Tracking Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()
