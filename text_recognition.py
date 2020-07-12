# coding: utf-8
# =====================================================================
#  Filename:    text_recognition.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Recognizes regions of text in a given image
#
#  Usage: python text_recognition.py --east frozen_east_text_detection.pb --image test.png
#         or
#         python text_recognition.py --east frozen_east_text_detection.pb --image test.png --padding 0.25
#
#  Note: Requires opencv 3.4.2 or later
#        Requires tesseract 4.0 or later
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import pytesseract
import numpy as np
import argparse
import cv2
from utils import forward_passer, box_extractor
from text_detection import resize_image
from imutils.object_detection import non_max_suppression


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
    ap.add_argument('-p', '--padding', type=float, default=0.0,
                    help='padding on each ROI border')
    arguments = vars(ap.parse_args())

    return arguments


def main(image, width, height, detector, min_confidence, padding):

    # reading in image
    image = cv2.imread(image)
    orig_image = image.copy()
    orig_h, orig_w = orig_image.shape[:2]

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

    results = []

    # text recognition main loop
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)

        dx = int((end_x - start_x) * padding)
        dy = int((end_y - start_y) * padding)

        start_x = max(0, start_x - dx)
        start_y = max(0, start_y - dy)
        end_x = min(orig_w, end_x + (dx*2))
        end_y = min(orig_h, end_y + (dy*2))

        # ROI to be recognized
        roi = orig_image[start_y:end_y, start_x:end_x]

        # recognizing text
        config = '-l eng --oem 1 --psm 7'
        text = pytesseract.image_to_string(roi, config=config)

        # collating results
        results.append(((start_x, start_y, end_x, end_y), text))

    # sorting results top to bottom
    results.sort(key=lambda r: r[0][1])

    # printing OCR results & drawing them on the image
    for (start_x, start_y, end_x, end_y), text in results:

        print('OCR Result')
        print('**********')
        print(f'{text}\n')

        # stripping out ASCII characters
        text = ''.join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig_image.copy()
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(output, text, (start_x, start_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('Detection', output)
        cv2.waitKey(0)


if __name__ == '__main__':

    # setting up tesseract path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    args = get_arguments()

    main(image=args['image'], width=args['width'], height=args['height'],
         detector=args['east'], min_confidence=args['min_confidence'],
         padding=args['padding'])
