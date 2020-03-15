import re
import time
import numpy as np
import cv2

from absl import app, flags, logging
from tflite_runtime.interpreter import Interpreter

"""
Example usage:
python detect.py --model=model/detect.tflite --labels=labels/label.txt
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "File path of .tflite file.")
flags.DEFINE_string("labels", None, "File path of labels file.")
flags.DEFINE_float("threshold", 0.4, "Score threshold for detected objects.")

# Required flags.
flags.mark_flag_as_required("model")
flags.mark_flag_as_required("labels")


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as labels_file:
        lines = labels_file.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def annotate_objects(frame, results, labels, elapsed_time):
    """Draws the bounding box and label for each object in the results."""

    img_height, img_width, _ = frame.shape

    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution

        ymin, xmin, ymax, xmax = obj['bounding_box']

        ymin = int(max(1, (ymin * img_height)))
        xmin = int(max(1, (xmin * img_width)))
        ymax = int(min(img_height, (ymax * img_height)))
        xmax = int(min(img_width, (xmax * img_width)))

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

        # Draw label
        # Look up object name from "labels" array using class index
        object_name = labels[int(obj['class_id'])]
        label = '%s: %d%%, %dms' % (object_name, int(
            obj['score']*100), elapsed_time)  # Example: 'license_plate: 72%, 15ms'
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
        # Make sure not to draw label too close to top of window
        label_ymin = max(ymin, label_size[1] + 10)
        # Draw white box to put label text in
        cv2.rectangle(frame, (xmin, label_ymin-label_size[1]-10), (
            xmin+label_size[0], label_ymin+base_line-10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text


def main(argv):
    del argv  # Unused.
    # Read labels prepare interpreter
    labels = load_labels(FLAGS.labels)
    interpreter = Interpreter(FLAGS.model)
    interpreter.allocate_tensors()

    # Get model details
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    # Check if model is quantized
    input_details = interpreter.get_input_details()
    floating_model = (input_details[0]['dtype'] == np.float32)

    # Get input source
    # cap = cv2.VideoCapture(0) #Use this for camera input as a source
    cap = cv2.VideoCapture('data/test_video.mp4') # Use this for video file as an input

    while(cap.isOpened()):

        # Prepare frame, preview window
        cv2.namedWindow('K9-Preview', cv2.WINDOW_NORMAL)
        ret, frame = cap.read()

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        image_resized = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_mean = 127.5
            input_std = 127.5
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Start detection, timer
        start_time = time.monotonic()
        results = detect_objects(interpreter, input_data, FLAGS.threshold)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Annotate objects - drawing bounding boxes
        annotate_objects(frame, results, labels, elapsed_ms)

        # Schow frame in preview
        cv2.imshow('K9-Preview', frame)

        logging.info(results)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(main)
