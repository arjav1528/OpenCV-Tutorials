import cv2
import HandAnnotationModule as ha
import HandTrackingModule as htm
import FaceDetectionModule as fdm



def main():
    faceDetector = fdm.FaceDetectionModule()
    handTracker = htm.HandTrackingModule()
    handAnnotater = ha.HandAnnotationModule()
    handAnnotater.initialize_camera()

    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()

        if success:
            image = cv2.flip(image,1)


            # image = faceDetector.detectFace(image)
            # faceDetector.writeScore(image)

            # image = handTracker.findHands(image)
            # bbox,lmList = handTracker.findPosition(image)
            # count,image = handTracker.fingersUp(image)


            combined, _ = handAnnotater.process_frame(image)  
            image = handAnnotater.add_instructions(combined)


            cv2.imshow('image',image)


            if(cv2.waitKey(1) == ord('q')):
                break
            elif cv2.waitKey(1) == ord('c'):
                handAnnotater.clear_canvas()

                
        else:
            break

    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


