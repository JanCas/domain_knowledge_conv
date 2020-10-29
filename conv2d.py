from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.compat.v1 import ConfigProto, Session
import tensorflow as tf

from numpy import floor
from numpy.random import seed
import matplotlib.pyplot as plt

from data_utils.pandas_creator import generate_image_data_generators
from settings import material_prop, cbfv, epochs, seed_setting

# setting the seed
seed(seed_setting)
tf.random.set_seed(seed_setting)

# GPU config
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = Session(config=config)

# data preprocessing -> turning the cbfv into a matrix
data = generate_image_data_generators(material_prop=material_prop, cbfv=cbfv)


def Alex_Net(input: Input) -> Sequential:
    """
        Performance (atom2vec):
        |--------------------------------------------------|
        | material_prop                      | MAE   | MSE |
        |------------------------------------|-------|-----|
        | ael_bulk_modulus_vrh               | 19.9  | 893.7|
        | ael_debye_temperature              | 125.12| 37164|
        | ael_shear_modulus_vrh              | 14.7  | 549 |
        | agl_log10_thermal_expansion_300K   | .1740 | .0487|
        | agl_thermal_conductivity_300K      | 3.22  | 51.7|
        | Egap                               | .5286 | .8363|
        | energy_atom                        | .3116 | .7044|
        |--------------------------------------------------|
        """
    model = Sequential()

    model.add(input)

    # 1ST layer
    model.add(Conv2D(filters=96, strides=2, kernel_size=5, activation='relu'))
    #model.add(Conv2D(filters=128, padding='same', strides=2, kernel_size=5, activation='relu'))

    # 2nd layer
    model.add(Conv2D(filters=128, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    # 3rd layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=5, activation='relu'))

    # 4th layer
    #model.add(Conv2D(filters=384, padding='same', kernel_size=3, activation='relu'))

    # 5th layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())

    # FC Layers
    model.add(Dense(8192, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', kernel_regularizer='l2'))
    #model.add(Dropout(.5))
    #model.add(Dense(4096, activation='relu', kernel_regularizer='l2'))
    #model.add(Dropout(.2))
    #model.add(Dense(2048, activation='relu'))

    # output layer
    model.add(Dense(1))

    return model


model = Alex_Net(input=data['input'])

optimizer = SGD(lr=.01)
METRICS = [
    'mean_squared_error',
    'mean_absolute_error'
]

CALLBACKS = [
    ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, factor=.1, min_lr=.0000001)
]

model.compile(loss='huber_loss', optimizer=optimizer, metrics=METRICS)
model.summary()

history = model.fit(
    data['train'],
    epochs=epochs,
    steps_per_epoch=floor(data['train'].n / data['train'].batch_size),
    validation_data=data['val'],
    callbacks=CALLBACKS
)



# evaluating the model
model.evaluate(data['test'])

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['mean_squared_error'], color='b', label="mean squared error")
ax[1].plot(history.history['val_mean_squared_error'], color='r', label="Validation MSE")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()
