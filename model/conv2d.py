from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.compat.v1 import ConfigProto, Session
from numpy import floor
import matplotlib.pyplot as plt

from data_utils.pandas_creator import generate_image_data_generators

from settings import material_prop, cbfv, epochs

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

data = generate_image_data_generators(material_prop=material_prop, cbfv=cbfv)

def Alex_Net(Input: Input) -> Sequential:
    """
        keras implementation of AlexNet
        :param weights: path to preloaded weights. If none starts with random weights
        :return:
        """
    model = Sequential()

    model.add(Input)

    # 1ST layer
    model.add(Conv2D(filters=96, strides=2, kernel_size=4, activation='relu'))
    #model.add(MaxPooling2D(pool_size=3, strides=2))

    # 2nd layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    # 3rd layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=5, activation='relu'))

    # 4th layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=3, activation='relu'))

    # 5th layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())

    # FC Layers
    model.add(Dense(9216, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu'))

    # output layer
    model.add(Dense(1))

    return model


model = Alex_Net(Input=data['input'])

optimizer = SGD(lr=.001)
METRICS = [
    'accuracy',
    Precision(name='precision'),
    Recall(name='recall')
]

CALLBACKS = [
    ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=.1, min_lr=.000000001)
]

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=METRICS)
model.summary()

history = model.fit(
    data['train'],
    epochs=epochs,
    steps_per_epoch= floor(data['train'].n / data['train'].batch_size),
    validation_data=data['val'],
    callbacks=CALLBACKS
)

# evaluating the model
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()