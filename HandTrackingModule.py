from typing import Sequence

import cv2
import time
import mediapipe as mp




class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionConf=0.5,trackConf=0.5):
        self.mode = mode
        self.maxHands = 2
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,min_detection_confidence=self.detectionConf,min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img, handNo=0,draw=True):
        lmList = []
        h,w,c = img.shape
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        return lmList

def main():
    prevTime = time.time()
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        currTime = time.time()
        fps = int(1/(currTime-prevTime))
        prevTime = currTime
        success, img = cap.read()
        img = cv2.flip(img,1)
        cv2.putText(img,str(fps),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


        finalImg = detector.findHands(img)
        lmList = detector.findPosition(finalImg)
        if len(lmList)!=0:
            print(lmList[4])

        cv2.imshow("Image", finalImg)

        if(cv2.waitKey(1) == ord('q')):
            break


    cv2.destroyAllWindows()








if __name__ == "__main__":
    main()