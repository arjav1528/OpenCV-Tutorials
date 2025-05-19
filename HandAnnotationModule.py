import cv2
import numpy as np
import HandTrackingModule as htm

class HandAnnotationModule:
    def __init__(self, width=1920, height=1080, fps=30, line_color=(255, 0, 255), line_thickness=10):
        self.width = width
        self.height = height
        self.fps = fps
        self.line_color = line_color
        self.line_thickness = line_thickness
        
        self.detector = htm.HandTrackingModule()
        
        self.prev_drawing = False
        self.prev_x, self.prev_y = 0, 0
        
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.initialized = False
        
    def initialize_camera(self, camera_id=0):
        if not self.initialized:
            temp_cap = cv2.VideoCapture(camera_id)
            success, img = temp_cap.read()
            if success:
                self.height, self.width = img.shape[:2]
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.initialized = True
            temp_cap.release()
        return True
    
    def clear_canvas(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.canvas
    
    def process_frame(self, image, show_hand=True, show_marker=True):
        h, w, c = image.shape
        if self.canvas.shape[:2] != (h, w):
            self.height, self.width = h, w
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        image = self.detector.findHands(image, draw=show_hand)
        bbox, lmList = self.detector.findPosition(image, draw=False)
        
        if len(lmList) != 0:
            fingers,image = self.detector.fingersUp(image,draw=False)
            
            if len(fingers) >= 2 and fingers[0] == 1 and fingers[1] == 1:
                x1, y1 = lmList[8][1], lmList[8][2]  # Index finger
                
                if self.prev_drawing:
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), 
                             self.line_color, self.line_thickness)
                
                self.prev_x, self.prev_y = x1, y1
                self.prev_drawing = True
                
                if show_marker:
                    cv2.circle(image, (x1, y1), 10, (0, 255, 0), -1)
            else:
                self.prev_drawing = False
        else:
            self.prev_drawing = False
        
        combined = cv2.addWeighted(image, 0.7, self.canvas, 0.3, 0)
        
        return combined, self.canvas
    
    def add_instructions(self, image):
        cv2.putText(image, "Thumb + Index = Draw", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 'c' to clear, 'q' to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image
    
    def findHands(self, image, draw=True):
        return self.detector.findHands(image, draw)
    
    def findPosition(self, image, handNo=0, draw=True):
        return self.detector.findPosition(image, handNo, draw)
    
    def fingersUp(self, image):
        return self.detector.fingersUp(image)
    
    def drawOnCanvas(self, image, x1, y1, show_marker=True):
        if self.prev_drawing:
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), 
                    self.line_color, self.line_thickness)
        
        self.prev_x, self.prev_y = x1, y1
        self.prev_drawing = True
        
        if show_marker:
            cv2.circle(image, (x1, y1), 10, (0, 255, 0), -1)
        
        return image
    
    def getCanvas(self):
        return self.canvas
    
    def getCombinedImage(self, image):
        return cv2.addWeighted(image, 0.7, self.canvas, 0.3, 0)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        self.initialize_camera()
        
        while True:
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            combined, _ = self.process_frame(image)
            combined = self.add_instructions(combined)
            
            cv2.imshow('Virtual Painter', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_canvas()
        
        cap.release()
        cv2.destroyAllWindows()
        return True

def main():
    annotator = HandAnnotationModule()
    annotator.run()

if __name__ == "__main__":
    main()