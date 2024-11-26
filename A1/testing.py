import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0) # 0 for computer, 1 for phone

while(True):
    start = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Bright point with functions
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    frame = cv2.circle(frame, maxLoc, 10, (0, 0, 255), 5) #Red circle
    
    #Red point with functions:
    b, g, r = cv2.split(frame) 
    red = r = r.astype(float) / (b.astype(float) + g.astype(float) + 1e-6)
    #red = r -(b+g)
    red = (red).astype(np.uint8)
    min_val, max_val, min_loc, maxLoc = cv2.minMaxLoc(red)
    frame = cv2.circle(frame, maxLoc, 10, (255, 0, 0), 5) #Blue circle

    #New red point
    # b, g, r = cv2.split(frame) 
    # norm_r = r.astype(float) / (b.astype(float) + g.astype(float) + 1e-6)
    # # Threshold for red dominance
    # redness = np.where((r > 150) & (r > g) & (r > b), norm_r, 0)  # Highlight red-dominant regions
    # # Convert redness to 8-bit for visualization
    # redness = (redness * 255).astype(np.uint8)
    # # Locate the reddest spot
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(redness)
    # # Mark the reddest spot on the frame
    # frame = cv2.circle(frame, max_loc, 10, (255, 0, 0), 5)

    # red = r / (b+g+1e-6)
    # #red = r -(b+g)
    # red = red.astype(np.uint8)
    # min_val, max_val, min_loc, maxLoc = cv2.minMaxLoc(red)
    # frame = cv2.circle(frame, maxLoc, 10, (255, 0, 0), 5) #Blue circle

    #red test
    # b, g, r = cv2.split(frame)  # Split into blue, green, and red channels
    # redness = r.astype(np.uint8) - np.maximum(g, b)  # Calculate redness as red - max(green, blue)
    # redness[redness < 0] = 0  # Ignore negative values
    # min_val, max_val, min_loc, red_loc = cv2.minMaxLoc(redness)
    # cv2.circle(frame, red_loc, 10, (255, 0, 0), 2)  # Mark the reddest spot with a red circle

    # b, g, r = cv2.split(frame)
    # # Normalize to avoid influence of overall intensity
    # norm_r = r.astype(float) / (b.astype(float) + g.astype(float) + 1e-6)

    # # Threshold for red dominance
    # redness = np.where((r > 100) & (r > g) & (r > b), norm_r, 0)  # Highlight red-dominant regions

    # # Convert redness to 8-bit for visualization
    # redness = (redness * 255).astype(np.uint8)

    # # Locate the reddest spot
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(redness)

    # # Mark the reddest spot on the frame
    # frame = cv2.circle(frame, max_loc, 10, (255, 0, 0), 5)

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