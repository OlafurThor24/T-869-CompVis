import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0) #0 for computer, 1 for phone

while(True):
    start = time.time()
    ret, frame = cap.read()

    # Convert image to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Use canny edge detection
    edges = cv2.Canny(gray,50,150,apertureSize=3)

    # Apply HoughLinesP method to 
    # to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=100, # Min number of votes for valid line
                minLineLength=5, # Min allowed length of line
                maxLineGap=10 # Max allowed gap between line for joining them
                )
    # Iterate over points
    if lines is not None:
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            #cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            lines_list.append(((x1,y1,x2,y2),len))
            
        sorted(lines_list)
        bestlines = lines_list[:4]

        for line,_ in bestlines:
            x1,y1,x2,y2 = line
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    end = time.time()
    second = end - start #time stamp
    fps = 1 / second if second > 0 else 0 #fps calculation

    #Printing    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    txt ="FPS: {fps:.2f}"
    cv2.putText(frame, txt.format(fps = fps),  (200, 440), font, 1, (255, 255, 255), 2, cv2.LINE_4) 

    cv2.imshow('frame',frame)
    cv2.imshow('edge',edges)

    #print(f",{second:.4f}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()