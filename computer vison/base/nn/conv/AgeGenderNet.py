from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras import backend as K


class AgeGenderNet:

    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)
        # Block #1: first CONV => RELU => POOL layer set
        conv1_1 = Conv2D(96, kernel_size=(7, 7),
                         strides=(4, 4))(inputs)
        act1_1 = Activation("relu")(conv1_1)
        bn1_1 = BatchNormalization(axis=chanDim)(act1_1)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bn1_1)
        do1 = Dropout(rate=0.25)(pool1)

        # Block #2: second CONV => RELU => POOL layer set
        conv2_1 = Conv2D(filters=256, kernel_size=(5, 5), padding=(2, 2))(do1)
        act2_1 = Activation("relu")(conv2_1)
        bn2_1 = BatchNormalization(axis=chanDim)(act2_1)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bn2_1)
        do2 = Dropout(rate=0.25)(pool2)

        # Block #3: third CONV => RELU => POOL layer set
        conv3_1 = Conv2D(filters=256, kernel_size=(5, 5), padding=(2, 2))(do2)
        act3_1 = Activation("relu")(conv3_1)
        bn3_1 = BatchNormalization(axis=chanDim)(act3_1)
        pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bn3_1)
        do3 = Dropout(rate=0.25)(pool3)

        # Block #4: first set of FC => RELU layers
        flatten = Flatten()(do3)
        fc1 = Dense(units=512)(flatten)
        act4_1 = Activation("relu")(fc1)
        bn4_1 = BatchNormalization(axis=chanDim)(act4_1)
        do4 = Dropout(rate=0.5)(bn4_1)

        # Block #5 second set of FC => RELU layers
        fc2 = Dense(units=512)(do4)
        act5_1 = Activation("relu")(fc2)
        bn5_1 = BatchNormalization(axis=chanDim)(act5_1)
        do5 = Dropout(rate=0.5)(bn5_1)

        # softmax classifier
        fc3 = Dense(units=classes)(do5)
        outputs = Activation("softmax")(fc3)
        model = Model(inputs=inputs, outputs=outputs)

        return model


