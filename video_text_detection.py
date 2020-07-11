from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
from text_detection import box_extractor


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', type=str,
                    help='path to optional video file')
    ap.add_argument('-east', '--east', type=str, required=True,
                    help='path to EAST text detection model')
    ap.add_argument('-c', '--min_confidence', type=float, default=0.5,
                    help='minimum confidence to process a region')
    ap.add_argument('-w', '--width', type=int, default=320,
                    help='resized image width (multiple of 32)')
    ap.add_argument('-e', '--height', type=int, default=320,
                    help='resized image height (multiple of 32)')
    arguments = vars(ap.parse_args())

    return arguments


args = get_arguments()

w, h = None, None
new_w, new_h = args['width'], args['height']
ratio_w, ratio_h = None, None

layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

if not args.get('video', False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1)

else:
    vs = cv2.VideoCapture(args['video'])

fps = FPS().start()

while True:

    frame = vs.read()
    frame = frame[1] if args.get('video', False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()

    if w is None or h is None:
        h, w = frame.shape[:2]
        ratio_w = w / float(new_w)
        ratio_h = h / float(new_h)

    frame = cv2.resize(frame, (new_w, new_h))

    blob = cv2.dnn.blobFromImage(frame, 1.0, (new_w, new_h), (123.68, 116.78, 103.94),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    scores, geometry = net.forward(layer_names)

    rectangles, confidences = box_extractor(scores, geometry, min_confidence=args['min_confidence'])
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)

        cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    fps.update()

    cv2.imshow("Detection", orig)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

fps.stop()
print(f"[INFO] elapsed time {round(fps.elapsed(), 2)}")
print(f"[INFO] approx. FPS : {round(fps.fps(), 2)}")

if not args.get('video', False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
