# OCR & Real-time Text Detection
Real-time OCR with openCV EAST & Tesseract.<br/>

_Requires openCV 3.4.2 or above._<br/>
_Requires Tesseract 4.0 or above._

## Real Time OCR
For text recognition on a live web-cam feed:
```commandline
python real_time_ocr.py --east frozen_east_text_detection.pb
```

For text recognition on a video file:
```commandline
python real_time_ocr.py --east frozen_east_text_detection.pb --video test.avi
```


## Text Detection
For text detection on image:
```commandline
python text_detection.py --image test.png --east frozen_east_text_detection.pb
```

For text detection on video or web-cam:
```commandline
python video_text_detection.py --east frozen_east_text_detection.pb
python video_text_detection.py --east frozen_east_text_detection.pb --video test.avi
```

## Text Recognition
For text recognition on image:
```commandline
python text_recognition.py --east frozen_east_text_detection.pb --image test.png
python text_recognition.py --east frozen_east_text_detection.pb --image test.png --padding 0.25
```
