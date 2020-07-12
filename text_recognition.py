import pytesseract
from text_detection import *
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

    image = cv2.imread(image)
    orig_image = image.copy()
    orig_h, orig_w = orig_image.shape[:2]

    image, ratio_w, ratio_h = resize_image(image, width, height)

    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']

    print("[INFO] loading the detector...")
    net = cv2.dnn.readNet(detector)

    scores, geometry = forward_passer(net, image, layers=layer_names)

    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    results = []

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

        roi = orig_image[start_y:end_y, start_x:end_x]

        config = '-l eng --oem 1 --psm 7'
        text = pytesseract.image_to_string(roi, config=config)

        results.append(((start_x, start_y, end_x, end_y), text))

    results.sort(key=lambda r: r[0][1])

    for (start_x, start_y, end_x, end_y), text in results:

        print('OCR Result')
        print('**********')
        print(f'{text}\n')

        text = ''.join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig_image.copy()
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(output, text, (start_x, start_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('Detection', output)
        cv2.waitKey(0)


if __name__ == '__main__':

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    args = get_arguments()

    main(image=args['image'], width=args['width'], height=args['height'],
         detector=args['east'], min_confidence=args['min_confidence'],
         padding=args['padding'])
