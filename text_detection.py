from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
from utils import forward_passer, box_extractor


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', type=str,
                    help='path to image')
    ap.add_argument('-east', '--east', type=str,
                    help='path to EAST text detection model')
    ap.add_argument('-c', '--min_confidence', type=float, default=0.5,
                    help='minimum confidence to process a region')
    ap.add_argument('-w', '--width', type=int, default=320,
                    help='resized image width (multiple of 32)')
    ap.add_argument('-e', '--height', type=int, default=320,
                    help='resized image height (multiple of 32)')
    arguments = vars(ap.parse_args())

    return arguments


def resize_image(image, width, height):

    h, w = image.shape[:2]

    ratio_w = w / width
    ratio_h = h / height

    image = cv2.resize(image, (width, height))

    return image, ratio_w, ratio_h


def main(image, width, height, detector, min_confidence):

    image = cv2.imread(image)
    orig_image = image.copy()

    image, ratio_w, ratio_h = resize_image(image, width, height)

    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']

    print("[INFO] loading the detector...")
    net = cv2.dnn.readNet(detector)

    scores, geometry = forward_passer(net, image, layers=layer_names)

    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)

        cv2.rectangle(orig_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    cv2.imshow("Detection", orig_image)
    cv2.waitKey(0)


if __name__ == '__main__':

    args = get_arguments()

    main(image=args['image'], width=args['width'], height=args['height'],
         detector=args['east'], min_confidence=args['min_confidence'])
