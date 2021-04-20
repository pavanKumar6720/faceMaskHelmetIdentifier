#Detector.py outputs the output.png based on the prediction made from the trained model
#note : multiple detection can be made by implementing face detection.


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-hd", "--hmodel", type=str,
	default="helmet_detector.model",
	help="path to trained face mask detector model")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


model = load_model(args["model"])
hmodel = load_model(args["hmodel"])

image = cv2.imread(args["image"])
orig = image.copy()
image = cv2.resize(image, (224, 224))
image = img_to_array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)
# pass the face through the model to determine if the face
# has a mask or not
(mask, withoutMask) = model.predict(image)[0]
(helmet, withoutHelmet) = hmodel.predict(image)[0]


label = "Mask" if mask > withoutMask else "No Mask"
color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
# include the probability in the label
label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

hlabel = "Helmet" if helmet > withoutHelmet else "No Helmet"
color = (255, 0, 0) if hlabel == "Helmet" else (0, 0, 255)
# include the probability in the label
hlabel = "{}: {:.2f}%".format(hlabel, max(helmet, withoutHelmet) * 100)

# display the label and bounding box rectangle on the output
# frame


cv2.putText(orig, label, (20, 40 ),
cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.putText(orig, hlabel, (20, 70 ),
cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
# show the output image
cv2.imshow("Output", orig)
cv2.waitKey(0)
cv2.imwrite("output.jpg",orig)