import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0) #0 for computer, 1 for phone

while(True):
    start = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Bright point with functions
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    frame = cv2.circle(frame, maxLoc, 10, (0, 0, 255), 5) #Red circle
    
    #Red point with functions:
    b, g, r = cv2.split(frame) 
    red = r.astype(float) / (b.astype(float) + g.astype(float) + 1e-6)
    #red = r -(b+g)
    red = (red*255).astype(np.uint8)
    min_val, max_val, min_loc, maxLoc = cv2.minMaxLoc(red)
    frame = cv2.circle(frame, maxLoc, 10, (255, 0, 0), 5) #Blue circle

    #Bright point with for-loop
    # brightnessmax = 0
    # brightestpoint = (0,0)
    # height,width = gray.shape
    # for i in range(height):
    #     for j in range(width):
    #         if gray[i,j] > brightnessmax:
    #             brightnessmax = gray[i,j]
    #             brightest_location = (j,i)
    #             print(brightnessmax)
    # frame = cv2.circle(frame, brightest_location, 10, (0, 0, 255), 5) #Red circle 

    end = time.time()

    second = end - start #time stamp
    fps = 1 / second if second > 0 else 0 #fps calculation
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    txt ="FPS: {fps:.2f}"

    #Printing
    cv2.putText(frame,  
                txt.format(fps = fps),  
                (200, 440),  
                font, 1,  
                (255, 255, 255),  
                2,  
                cv2.LINE_4) 

    cv2.imshow('frame',frame)

    #print(f",{second:.4f}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()