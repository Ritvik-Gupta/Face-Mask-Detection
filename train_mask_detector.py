# python train_mask_detector.py --dataset dataset

import argparse
import logging as log
import os

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# import the necessary packages
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from tensorflow.keras.utils import to_categorical

# initialize the initial learning rate, number of epochs to train for, and batch size
INITIAL_LEARNING = 1e-4
EPOCHS = 10
BATCH_SIZE = 32


def train_model():
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    log.info("Loading images")
    image_paths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []

    # loop over the image paths
    for image_path in image_paths:
        # extract the class label from the filename
        label = image_path.split(os.path.sep)[-2]

        # load the input image (224x224) and preprocess it
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # load the MobileNetV2 network, ensuring the head FC layer sets are left off
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
    )

    # construct the head of the model that will be placed on top of the
    # the base model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they will
    # not be updated during the first training process
    for layer in base_model.layers:
        layer.trainable = False

    # compile our model
    log.info("Compiling model")
    opt = Adam(lr=INITIAL_LEARNING, decay=INITIAL_LEARNING / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the head of the network
    log.info("Training head")
    head_net = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BATCH_SIZE,
        epochs=EPOCHS,
    )

    # make predictions on the testing set
    log.info("Evaluating network")
    pred_idxs = model.predict(testX, batch_size=BATCH_SIZE)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    pred_idxs = np.argmax(pred_idxs, axis=1)

    # show a nicely formatted classification report
    print(
        classification_report(testY.argmax(axis=1), pred_idxs, target_names=lb.classes_)
    )

    # serialize the model to disk
    log.info("Saving mask detector model")

    return model, head_net


if __name__ == "__main__":
    log.root.setLevel(log.INFO)

    # construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-d", "--dataset", type=str, default="dataset", help="path to input dataset"
    )
    arg_parser.add_argument(
        "-p",
        "--plot",
        type=str,
        default="precision_metrics.png",
        help="path to output loss/accuracy plot",
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mask_detector.model",
        help="path to output face mask detector model",
    )
    args = vars(arg_parser.parse_args())

    model, head_net = train_model()

    model.save(args["model"], save_format="h5")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), head_net.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), head_net.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), head_net.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), head_net.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    plt.show()
