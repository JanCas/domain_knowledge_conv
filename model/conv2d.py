from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import Input

def Alex_Net(input: Input, big_input: bool, extra: bool) -> Sequential:
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
    if extra:
        model.add(MaxPooling2D(pool_size=3, strides=2))

    # 2nd layer
    model.add(Conv2D(filters=128, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    # 3rd layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=5, activation='relu'))
    if big_input:
        model.add(MaxPooling2D(pool_size=3, strides=2))

    # 4th layer
    #model.add(Conv2D(filters=384, padding='same', kernel_size=3, activation='relu'))

    # 5th layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())

    # FC Layers
    model.add(Dense(4096, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', kernel_regularizer='l2'))
    #model.add(Dropout(.5))
    #model.add(Dense(4096, activation='relu', kernel_regularizer='l2'))
    #model.add(Dropout(.2))
    #model.add(Dense(2048, activation='relu'))

    # output layer
    model.add(Dense(1))

    return model
