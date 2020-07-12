# coding: utf-8
# =====================================================================
#  Filename:    text_detection.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Recognizes regions of text in a given image
#
#  Usage: python text_detection.py --image test.png --east frozen_east_text_detection.pb
#
#  Note: Requires opencv 3.4.2 or later
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

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
    """
    Re-sizes image to given width & height
    :param image: image to resize
    :param width: new width
    :param height: new height
    :return: modified image, ratio of new & old height and width
    """
    h, w = image.shape[:2]

    ratio_w = w / width
    ratio_h = h / height

    image = cv2.resize(image, (width, height))

    return image, ratio_w, ratio_h


def main(image, width, height, detector, min_confidence):

    # reading in image
    image = cv2.imread(image)
    orig_image = image.copy()

    # resizing image
    image, ratio_w, ratio_h = resize_image(image, width, height)

    # layers used for ROI recognition
    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']

    # pre-loading the frozen graph
    print("[INFO] loading the detector...")
    net = cv2.dnn.readNet(detector)

    # getting results from the model
    scores, geometry = forward_passer(net, image, layers=layer_names)

    # decoding results from the model
    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    # applying non-max suppression to get boxes depicting text regions
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    # drawing rectangles on the image
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
