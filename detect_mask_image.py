# python detect_mask_image.py --image images/pic1.jpeg

import argparse
import logging as log
import os

import cv2
import numpy as np

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

FACE_DETECTOR_PATH = "face_detector"
PROTOTXT_FILE = "deploy.prototxt"
WEIGHTS_FILE = "res10_300x300_ssd_iter_140000.caffemodel"


def detect_mask_in_image(args):
    # load our serialized face detector model from disk
    log.info("Loading face detector model")
    face_detector_net = cv2.dnn.readNet(
        os.path.sep.join([FACE_DETECTOR_PATH, PROTOTXT_FILE]),
        os.path.sep.join([FACE_DETECTOR_PATH, WEIGHTS_FILE]),
    )

    # load the face mask detector model from disk
    log.info("Loading face mask detector model")
    model = load_model(args["model"])

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(args["image"])
    (height, width) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    log.info("Computing face detections")
    face_detector_net.setInput(blob)
    detections = face_detector_net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the object
            bounding_box = detections[0, 0, i, 3:7] * np.array(
                [width, height, width, height]
            )
            (startX, startY, endX, endY) = bounding_box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(
                image,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    log.root.setLevel(log.INFO)

    # construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--image", required=True, help="path to input image")
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model",
    )
    arg_parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    args = vars(arg_parser.parse_args())

    detect_mask_in_image(args)
