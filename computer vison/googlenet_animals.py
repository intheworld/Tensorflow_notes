import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from base.nn.conv.minigooglenet import MiniGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from base.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from base.preprocessing.simplepreprocessor import SimplePreprocessor
from base.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import numpy as np
import argparse
import os
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX),
                        epochs=70, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])

# 23个epoch之后，valid集上准确率达到80%
