
import cv2 
import time
import os 
from handTrackingModule import HandDetector




wCam,hCam = 640,480 
capture = cv2.VideoCapture(0)
capture.set(3,wCam)
capture.set(4,hCam)

images_path = "assets/fingers"
imagesList = os.listdir(images_path)

fingerTips = [4,8,12,16,20]
overlayList = []

for imPath in imagesList:
    image = cv2.imread(f"{images_path}/{imPath}")
    # print(f"{images_path}/{imPath}")
    overlayList.append(image)



pTime  = 0 

detector = HandDetector()

while True: 

    success,img = capture.read()

    img =  detector.findHands(img)
    lmList = detector.findPostition(img,draw=False)
    
    if len(lmList)!=0:
        fingers = []
        # For thumb
        if lmList[fingerTips[0]][1] > lmList[fingerTips[0]- 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # For 4 fingers
        for i in range(1,5):
            if lmList[fingerTips[i]][2] < lmList[fingerTips[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        print("Fingers ",fingers)
        totalFingers = fingers.count(1)
        
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]
        
        cv2.rectangle(img,(20,265),(170,465),(255,255,0),cv2.FILLED)
        cv2.putText(img,f"{totalFingers}",(45,425),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),12)

        


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f"FPS: {int(fps)}",(410,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Finger Counter",img)
    cv2.waitKey(1)
