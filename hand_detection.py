import cv2
import mediapipe as mp
import os
import urllib.request

# Model file download karna (pehli baar me ek baar hoga)
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Model download ho raha hai, thoda wait karo...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download ho gaya!")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5)
landmarker = vision.HandLandmarker.create_from_options(options)

tip_ids = [4, 8, 12, 16, 20]

# Haath ki lines ke liye connections
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
               (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = landmarker.detect(mp_image)

    fingers = 0

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            lm_list = []
            h, w, c = img.shape
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                lm_list.append([x, y])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

            for start, end in CONNECTIONS:
                cv2.line(img, tuple(lm_list[start]), tuple(lm_list[end]), (0, 255, 0), 2)

            if len(lm_list) != 0:
                if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0]-1][0]:
                    fingers += 1
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id]-2][1]:
                        fingers += 1

    cv2.putText(img, f'Fingers: {fingers}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()