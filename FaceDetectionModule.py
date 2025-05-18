import time
import cv2
import numpy as np
import mediapipe as mp


class FaceDetectionModule:
    def __init__(self,):
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def detectFace(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(image)
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                if draw:
                    self.mpDraw.draw_detection(image,detection)
        return image
    def writeScore(self,image):
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):

                h,w,c = image.shape
                xmin = detection.location_data.relative_bounding_box.xmin * w
                xmax = (detection.location_data.relative_bounding_box.xmin+detection.location_data.relative_bounding_box.width) * w
                ymin = int((detection.location_data.relative_bounding_box.ymin) * h)
                x = int(xmin)
                score = round(detection.score[0]*100,2)

                cv2.putText(image,f"{score}%",(x,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)

def main():
    cap = cv2.VideoCapture(0)
    face = FaceDetectionModule()
    while True:
        success, image = cap.read()
        if success:
            image = face.detectFace(image)     
            face.writeScore(image)
            cv2.imshow('image',image)
            if(cv2.waitKey(1) == ord('q')):
                break
        else:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()