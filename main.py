import cv2
import mediapipe as mp
import os
import urllib.request
import webbrowser
import pyautogui
import time
import math

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.5)
landmarker = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
youtube_khula = False
last_action_time = 0

prev_x = None
prev_y = None

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = landmarker.detect(mp_image)
    action_text = ""
    gesture_name = ""

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            lm = []
            for point in hand:
                x, y = int(point.x * w), int(point.y * h)
                lm.append([x, y])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            for s, e in CONNECTIONS:
                cv2.line(img, tuple(lm[s]), tuple(lm[e]), (0, 255, 0), 2)

            wrist = lm[0]
            thumb_tip = lm[4]
            thumb_ip = lm[3]
            index_tip = lm[8]
            index_pip = lm[6]
            middle_tip = lm[12]
            middle_pip = lm[10]
            ring_tip = lm[16]
            ring_pip = lm[14]
            pinky_tip = lm[20]
            pinky_pip = lm[18]

            index_up = index_tip[1] < index_pip[1]
            middle_up = middle_tip[1] < middle_pip[1]
            ring_up = ring_tip[1] < ring_pip[1]
            pinky_up = pinky_tip[1] < pinky_pip[1]
            thumb_up = thumb_tip[1] < thumb_ip[1]
            finger_count = sum([index_up, middle_up, ring_up, pinky_up])
            is_pinch = dist(thumb_tip, index_tip) < 40
            now = time.time()

            if prev_x is not None and prev_y is not None:
                dy = wrist[1] - prev_y
                dx = wrist[0] - prev_x
                if dy < -35:
                    gesture_name = "Haath UPAR - Scroll Up"
                    if now - last_action_time > 0.8:
                        pyautogui.scroll(300)
                        pyautogui.press("up")
                        action_text = "Scroll Up"
                        last_action_time = now
                elif dy > 35:
                    gesture_name = "Haath NEECHE - Scroll Down"
                    if now - last_action_time > 0.8:
                        pyautogui.scroll(-300)
                        pyautogui.press("down")
                        action_text = "Scroll Down"
                        last_action_time = now
                elif dx > 40:
                    gesture_name = "Haath RIGHT - 10 Sec Aage"
                    if now - last_action_time > 1.0:
                        pyautogui.press("right")
                        action_text = "Forward 10 Sec"
                        last_action_time = now
                elif dx < -40:
                    gesture_name = "Haath LEFT - 10 Sec Peeche"
                    if now - last_action_time > 1.0:
                        pyautogui.press("left")
                        action_text = "Backward 10 Sec"
                        last_action_time = now
            prev_x, prev_y = wrist[0], wrist[1]

            if now - last_action_time > 1.2:
                if is_pinch:
                    gesture_name = "PINCH - Click"
                    if not youtube_khula:
                        webbrowser.open("https://www.youtube.com")
                        youtube_khula = True
                        action_text = "YouTube Khul Raha Hai..."
                    else:
                        pyautogui.click()
                        action_text = "Click Kiya"
                    last_action_time = now
                elif finger_count == 5 and not youtube_khula:
                    gesture_name = "Pura Haath - Open YouTube"
                    webbrowser.open("https://www.youtube.com")
                    youtube_khula = True
                    action_text = "YouTube Khul Raha Hai..."
                    last_action_time = now
                elif finger_count == 1 and youtube_khula:
                    gesture_name = "1 Finger - Play / Pause"
                    pyautogui.press("space")
                    action_text = "Play / Pause"
                    last_action_time = now
                elif finger_count == 2 and youtube_khula:
                    gesture_name = "2 Finger - Mute"
                    pyautogui.press("m")
                    action_text = "Mute / Unmute"
                    last_action_time = now
                elif finger_count == 3 and youtube_khula:
                    gesture_name = "3 Finger - Full Screen"
                    pyautogui.press("f")
                    action_text = "Full Screen"
                    last_action_time = now
                elif thumb_up and finger_count == 0 and youtube_khula:
                    gesture_name = "Thumb UP - Next Video"
                    pyautogui.hotkey("shift", "n")
                    action_text = "Next Video"
                    last_action_time = now
    else:
        prev_x, prev_y = None, None

    cv2.putText(img, gesture_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, action_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Full YouTube Hand Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()