
import math
import time
import mediapipe 
import cv2



class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon 
        self.trackCon = trackCon
        self.mpHands = mediapipe.solutions.hands 
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,max_num_hands=self.maxHands,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)
        self.mpDraw = mediapipe.solutions.drawing_utils
        self.lmList =[]
        self.fingerTips =[4,8,12,16,20]
    
    def findHands(self,img,draw=True):
        imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)
        
        if self.results.multi_hand_landmarks:    
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        
        return img     
    
    def findPostition(self,img,handNo=0,draw=True):
        self.lmList = []
        
        if self.results.multi_hand_landmarks:    
            myHand = self.results.multi_hand_landmarks[handNo]
         
            for id,lm in enumerate(myHand.landmark): 
                h,w,c = img.shape 
                cx,cy = int(lm.x*w),int(lm.y*h) 
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,255,0),cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        # For thumb
        if self.lmList[self.fingerTips[0]][1] < self.lmList[self.fingerTips[0]- 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # For 4 fingers
        for i in range(1,5):
            if self.lmList[self.fingerTips[i]][2] < self.lmList[self.fingerTips[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
     
        totalFingers = fingers.count(1)
        return fingers,totalFingers
        
    def findDistance(self,p1,p2,img,draw=True):
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        
        cx,cy = (x1+x2)//2,(y1+y2)//2 
        if draw:
            cv2.circle(img, (x1,y1),12,(255,0,255),cv2.FILLED)
            cv2.circle(img, (x2,y2),12,(255,0,255),cv2.FILLED)
            cv2.circle(img, (cx,cy),12,(255,0,255),cv2.FILLED)
            cv2.line(img, (x1,y1),(x2,y2),(255,0,255),3)
        length= math.hypot(x2-x1,y2-y1)
        
        if length<50:
            cv2.circle(img, (cx,cy),12,(0,255,0),cv2.FILLED)
        return length,img,[x1,y1,x2,y2,cx,cy]
        



def main():

    currentTime = 0
    previousTime = 0
    capture = cv2.VideoCapture(0)
 
  
    detector = HandDetector()

    while True:
        success,img = capture.read()
        if success:
            img = detector.findHands(img)
            
            lmList = detector.findPostition(img)
            if len(lmList) !=0:
                print(lmList[8])
      
        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
    
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)



if __name__ == "__main__":
     main()
     
     
