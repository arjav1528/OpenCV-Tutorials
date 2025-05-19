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
        xList = []
        yList = []
        bbox = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
                bbox = [min(xList),min(yList),max(xList),max(yList)]
            
            if draw:
                cv2.rectangle(img,(min(xList)-20,min(yList)-20),(max(xList)+20,max(yList)+20),(255,255,255),2)

        # bbox = [min(xList),min(yList),max(xList),max(yList)]
        

        return bbox,lmList
    
    def fingersUp(self, img):
        count = [0, 0, 0, 0, 0]
        bbox, lmList = self.findPosition(img, draw=False)
        if len(lmList) != 0:
            # Thumb
            if ((lmList[4][1] < lmList[3][1] and lmList[3][1] < lmList[2][1] and lmList[2][1] < lmList[1][1]) or 
                (lmList[4][1] > lmList[3][1] and lmList[3][1] > lmList[2][1] and lmList[2][1] > lmList[1][1])):
                count[0] = 1
            
            # Index finger
            if lmList[8][2] < lmList[7][2] and lmList[7][2] < lmList[6][2] and lmList[6][2] < lmList[5][2]:
                count[1] = 1
                
            # Middle finger
            if lmList[12][2] < lmList[11][2] and lmList[11][2] < lmList[10][2] and lmList[10][2] < lmList[9][2]:
                count[2] = 1
                
            # Ring finger
            if lmList[16][2] < lmList[15][2] and lmList[15][2] < lmList[14][2] and lmList[14][2] < lmList[13][2]:
                count[3] = 1
                
            # Pinky
            if lmList[20][2] < lmList[19][2] and lmList[19][2] < lmList[18][2] and lmList[18][2] < lmList[17][2]:
                count[4] = 1

        return count


def main():
    prevTime = time.time()
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    # count = [0,0,0,0,0]

    while True:
        currTime = time.time()
        fps = int(1/(currTime-prevTime))
        prevTime = currTime
        success, img = cap.read()
        img = cv2.flip(img,1)
        cv2.putText(img,f"FPS : {fps}",(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


        finalImg = detector.findHands(img)
        bbox,lmList = detector.findPosition(finalImg)
        count = detector.fingersUp(img=img)

        

        print(sum(count))
        cv2.putText(img,f"Count : {sum(count)}",(10,150),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("Image", finalImg)

        if(cv2.waitKey(1) == ord('q')):
            print(bbox)
            break


    cv2.destroyAllWindows()








if __name__ == "__main__":
    main()