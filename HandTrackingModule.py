from typing import Sequence

import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = time.time()



while True:
    currTime = time.time()
    fps = int(1/(currTime-prevTime))

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    text = f"FPS : {fps}"
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)
    thickness = 2

    cv2.putText(img, text, position, font, font_scale, color, thickness)



    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id == 0:
                    cv2.circle(img,(cx,cy),20,(255,0,255),-1)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("image", img)
    prevTime = currTime


    if(cv2.waitKey(1) == ord('q')):
        break


cv2.destroyAllWindows()