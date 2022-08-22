# python detect_mask_video.py

import argparse
import logging as log
import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

FACE_DETECTOR_PATH = "face_detector"
PROTOTXT_FILE = "deploy.prototxt"
WEIGHTS_FILE = "res10_300x300_ssd_iter_140000.caffemodel"


def detect_and_predict_mask(frame, face_net, mask_net, args):
    # grab the dimensions of the frame and then construct a blob from it
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
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
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


def detect_mask_in_video(args):
    # load our serialized face detector model from disk
    log.info("Loading face detector model")
    face_detector_net = cv2.dnn.readNet(
        os.path.sep.join([FACE_DETECTOR_PATH, PROTOTXT_FILE]),
        os.path.sep.join([FACE_DETECTOR_PATH, WEIGHTS_FILE]),
    )

    # load the face mask detector model from disk
    log.info("Loading face mask detector model")
    mask_net = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    log.info("Starting video stream")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a face mask or not
        (locs, preds) = detect_and_predict_mask(
            frame, face_detector_net, mask_net, args
        )

        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    log.root.setLevel(log.INFO)

    # construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
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

    detect_mask_in_video(args)
