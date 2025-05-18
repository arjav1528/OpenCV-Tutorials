import time
import cv2
import numpy as np
import mediapipe as mp


cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

prevTime = time.time()


while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    currTime = time.time()
    fps = int(1 / (currTime - prevTime))
    prevTime = currTime
    cv2.putText(image, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    
    imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(image)

    # print(len(results.detections))  
    

    if results.detections:
        for id,detection in enumerate(results.detections):
            h,w,c = image.shape
            xmin = detection.location_data.relative_bounding_box.xmin * w
            xmax = (detection.location_data.relative_bounding_box.xmin+detection.location_data.relative_bounding_box.width) * w
            ymin = int((detection.location_data.relative_bounding_box.ymin) * h)
            x = int(xmin)
            score = round(detection.score[0]*100,2)
            cv2.putText(image,f"{score}%",(x,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
            mpDraw.draw_detection(image,detection)
    
    

    # print(results)
    cv2.imshow("Image", image)

    if(cv2.waitKey(1) == ord('q')):
        break



cv2.destroyAllWindows()