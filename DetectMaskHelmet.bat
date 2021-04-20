
REM #Detector.py outputs the output.png based on the prediction made from the trained model

REM #note : multiple detection can be made by implementing face detection.

python Detector.py --image "testImages//testImage2.jpg" --m "output//maskdetector.model" --hmodel "output//helmetdetector.model" --c 0.4