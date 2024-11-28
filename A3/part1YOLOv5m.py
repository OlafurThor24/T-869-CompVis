import cv2
import time
import numpy as np
import onnxruntime as ort

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK  = (0, 0, 0)
BLUE   = (255, 178, 50)
YELLOW = (0, 255, 255)

# Load class names.
classesFile = "YOLOv5\\data\\coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the ONNX model with onnxruntime.
modelWeights = "YOLOv5\\yolov5m.onnx"
session = ort.InferenceSession(modelWeights)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, session):
    """Pre-process input image for ONNX model."""
    resized_image = cv2.resize(input_image, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = resized_image.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # Change from HWC to CHW
    blob = np.expand_dims(blob, axis=0)  # Add batch dimension
    input_name = session.get_inputs()[0].name
    return blob.astype(np.float16), input_name

def post_process(input_image, outputs):
    """Post-process the outputs from ONNX model."""
    class_ids = []
    confidences = []
    boxes = []
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    output = outputs[0]  # Assuming one output layer
    rows = output.shape[1]

    for r in range(rows):
        row = output[0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non-maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Check if 'indices' is empty or not in expected format.
    if indices is None or len(indices) == 0:
        return input_image  # If no valid indices, return the original image

    # Check if 'indices' is a tuple and handle it.
    if isinstance(indices, tuple):
        indices = indices[0]  # Extract the first element if it's a tuple

    # Ensure that indices is not empty.
    if len(indices) == 0:
        return input_image

    for i in indices.flatten():
        box = boxes[i]
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image

# Start capturing video from webcam.
cap = cv2.VideoCapture(0)  # 0 for the computer's default webcam

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process frame for the ONNX model.
    blob, input_name = pre_process(frame, session)

    # Run inference with ONNX runtime.
    outputs = session.run(None, {input_name: blob})

    # Process the outputs and draw the bounding boxes.
    processed_frame = post_process(frame.copy(), outputs)

    # Calculate FPS.
    end = time.time()
    second = end - start
    fps = 1 / second if second > 0 else 0

    # Display FPS on the frame.
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = "FPS: {fps:.2f}"
    cv2.putText(processed_frame, txt.format(fps=fps), (200, 440), font, 1, (255, 255, 255), 2, cv2.LINE_4)

    # Show the processed frame.
    cv2.imshow('frame', processed_frame)

    # Press 'q' to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
