from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2


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


def forward_passer(detector, image, layers):

    print("[INFO] loading the detector...")
    net = cv2.dnn.readNet(detector)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    scores, geometry = net.forward(layers)
    end = time.time()

    print(f"[INFO] detection in {round(end - start, 2)} seconds")

    return scores, geometry


def box_extractor(scores, geometry, min_confidence):

    num_rows, num_cols = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            box_h = x_data0[x] + x_data2[x]
            box_w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y + (cos * x_data2[x]) - (sin * x_data1[x]))
            start_x = int(end_x - box_w)
            start_y = int(end_y - box_h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rectangles, confidences


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
    h, w = image.shape[:2]

    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']
    scores, geometry = forward_passer(detector, image, layers=layer_names)

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
