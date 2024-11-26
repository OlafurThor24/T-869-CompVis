import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0) #0 for computer, 1 for phone

while(True):
    start = time.time()
    ret, frame = cap.read()

    #Sett parameter values 
    lower = 50  #Lower Threshold 
    upper = 150  #Upper threshold 
    
    #Canny Edge filter 
    edge = cv2.Canny(frame, lower, upper) 

    #Get pixel coordinates where edges exist
    y_index, x_index = np.nonzero(edge) 
    edge_points = np.column_stack((x_index, y_index)) 

    #Find the best line
    #0. Set d(threshold), N(iterations) and initialize variables
    N = 150 #Iterations
    d = 5 #Threshold
    bestline = 0
    mostinliers = 0
    for i in range(N):
        #1. Get 2 random points from edgepoints
        randomvalues = np.random.choice(edge_points.shape[0], 2, replace=False)
        randompoint1,randompoint2 = edge_points[randomvalues]
        #2. Set a line across with 2 points
        a = randompoint2[1] - randompoint1[1]  # y2 - y1
        b = randompoint1[0] - randompoint2[0]  # x1 - x2
        c = randompoint2[0] * randompoint1[1] - randompoint1[0] * randompoint2[1]  # x2*y1 - x1*y2
        #3. Check how many other points in edgepoints are within the threshold
        distances = np.abs(a * edge_points[:, 0] + b * edge_points[:, 1] + c)
        inliers = distances < d
        num_inliers = np.sum(inliers)
        #4. If there are enough points within the threshold select the line as bestline
        if num_inliers > mostinliers:
            mostinliers = num_inliers
            bestpoint1 = randompoint1
            bestpoint2 = randompoint2
        bestline = a,b,c

    #5. Draw the best line
    if bestline is not None:
        a, b, c = bestline
        height, width = frame.shape[:2]

        # Calculate intersection points with the frame boundaries
        points = []
                
        if b != 0:
            #Left edge (x = 0)
            lefty = int(-c / b)
            if 0 <= lefty < height:
                points.append((0, lefty))
            #Right edge (x = width - 1)
            righty = int(-(a * (width - 1) + c) / b)
            if 0 <= righty < height:
                points.append((width - 1, righty))

        if a != 0:
            #Top edge (y = 0)
            topx = int(-c / a)
            if 0 <= topx < width:
                points.append((topx, 0))
            #Bottom edge (y = height - 1)
            botx = int(-(b * (height - 1) + c) / a)
            if 0 <=botx < width:
                points.append((botx, height - 1))

        #Draw the line if at least two valid points are found
        if len(points) >= 2:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)

    #Old 5. Draw the best line between points Old
    # pt1 = tuple(bestpoint1)  # Convert to tuple for OpenCV
    # pt2 = tuple(bestpoint2)
    # cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Draw line in green
    #frame = cv2.line(frame,(x_start,y_start),(x_end,y_end),(0,255,0),3)

    end = time.time()

    second = end - start #time stamp
    fps = 1 / second if second > 0 else 0 #fps calculation

    #Printing    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    txt ="FPS: {fps:.2f}"
    cv2.putText(frame, txt.format(fps = fps),  (200, 440), font, 1, (255, 255, 255), 2, cv2.LINE_4) 

    cv2.imshow('frame',frame)
    cv2.imshow('edge',edge)

    #print(f",{second:.4f}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()