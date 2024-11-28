import cv2
import numpy as np
import time

def line_to_homogeneous(line):
    """Convert a line in pixel coordinates to its homogeneous representation."""
    x1, y1, x2, y2 = line
    return np.cross([x1, y1, 1], [x2, y2, 1])

def find_intersection(line1, line2):
    """Find the intersection of two lines in homogeneous coordinates."""
    intersection = np.cross(line1, line2)
    if intersection[2] == 0:
        return None  # Lines are parallel
    x, y = intersection[0] / intersection[2], intersection[1] / intersection[2]
    return int(x), int(y)

def is_crossed_quadrangle(points):
    """Check if a quadrangle is crossed by detecting self-intersecting edges."""
    def do_intersect(p1, q1, p2, q2):
        """Check if line segments p1q1 and p2q2 intersect."""
        def orientation(a, b, c):
            """Find orientation of ordered triplet (a, b, c)."""
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        return False

    # Check all pairs of edges
    for i in range(4):
        p1, q1 = points[i], points[(i + 1) % 4]
        for j in range(i + 2, 4):
            p2, q2 = points[j], points[(j + 1) % 4]
            if (i == 0 and j == 3):  # Avoid checking adjacent edges
                continue
            if do_intersect(p1, q1, p2, q2):
                return True
    return False

cap = cv2.VideoCapture(0)  # 0 for computer, 1 for phone

# Resize scale factor
resize_factor = 1  # Resize as needed (1 equals 100%)

while True:
    start = time.time()
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 100, 500, apertureSize=3)

    # Apply HoughLinesP method to obtain line endpoints
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=50,  # Min number of votes for a valid line
        minLineLength=100,  # Min allowed length of a line
        maxLineGap=5  # Max allowed gap between line for joining them
    )

    intersections = []
    if lines is not None:
        for points in lines:
            x1, y1, x2, y2 = points[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            lines_list.append(((x1, y1, x2, y2), length))

        # Sort lines by length (descending) and select the top 4
        lines_list.sort(key=lambda x: x[1], reverse=True)
        bestlines = [line for line, _ in lines_list[:4]]

        # Draw the selected lines on the frame
        for line in bestlines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Find intersections of the best lines
        homogeneous_lines = [line_to_homogeneous(line) for line in bestlines]
        for i in range(len(homogeneous_lines)):
            for j in range(i + 1, len(homogeneous_lines)):
                intersection = find_intersection(homogeneous_lines[i], homogeneous_lines[j])
                if intersection:
                    intersections.append(intersection)
                    # Draw the intersection points
                    cv2.circle(frame, intersection, 5, (0, 0, 255), -1)

    # Ensure we have exactly 4 intersections for the quadrangle
    if len(intersections) >= 4:
        intersections = np.array(intersections)

        # Find contours and approximate as polygons
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Approximate contour with accuracy proportional to the contour perimeter
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:  # Four-sided shape detected
                points = approx.reshape(4, 2)

                # Check if the quadrangle is self-intersecting
                if is_crossed_quadrangle(points):
                    continue  # Skip crossed quadrangles

                # Draw the quadrangle
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

                # Perspective transformation
                src_pts = points.astype("float32")
                dst_pts = np.array([[0, 0], [400 - 1, 0], [400 - 1, 300 - 1], [0, 300 - 1]], dtype="float32")
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # Warp the image
                warped = cv2.warpPerspective(frame, matrix, (400, 300))
                if warped is not None and warped.size > 0:
                    cv2.imshow("quadrangle", warped)

    # FPS calculation and display
    end = time.time()
    elapsed_time = end - start
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = "FPS: {fps:.2f}"
    cv2.putText(frame, txt.format(fps=fps), (10, 30), font, 1, (255, 255, 255), 2)

    # Show intermediate results
    cv2.imshow('frame', frame)
    cv2.imshow('edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
