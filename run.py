from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.compat.v1 import ConfigProto, Session
import tensorflow as tf
from matplotlib.pyplot import subplots, savefig

from numpy import floor
from numpy.random import seed
from pandas import DataFrame

from data_utils.pandas_creator import generate_image_data_generators
from settings import material_prop_list, epochs, seed_setting, cbfv_list, BASE_DIR
from model.conv2d import Alex_Net
from os.path import join
from time import localtime, strftime


# setting the seed
seed(seed_setting)
tf.random.set_seed(seed_setting)

# GPU config
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = Session(config=config)

#model config
optimizer = SGD(lr=.01)
METRICS = [
    'mean_squared_error',
    'mean_absolute_error',
]

CALLBACKS = [
    ReduceLROnPlateau(monitor='val_mean_absolute_error', patience=3, verbose=1, factor=.1, min_lr=.0000001),
    EarlyStopping(monitor='mean_absolute_error', patience=7),
    EarlyStopping(monitor='val_absolute_error', patience=7)
]

history_storage = {
    'cbfv':[],
    'material_prop': [],
    'test_loss':[],
    'test_MAE':[],
    'test_MSE':[],
    'val_loss':[],
    'val_MAE':[],
    'val_MSE':[],
    'train_loss':[],
    'train_MAE':[],
    'train_MSE':[]
}
for cbfv in cbfv_list:
    for material_prop in material_prop_list:
        history_storage['cbfv'].append(cbfv)
        history_storage['material_prop'].append(material_prop)

        data = generate_image_data_generators(material_prop=material_prop, cbfv=cbfv)
        print('----------------------------------------------------------------------')
        print('|   MATERIAL_PROP: {}, CBFV: {}'.format(material_prop, cbfv))
        print('|   NUM DATAPOINTS -> TRAIN: {}, VAL: {}, TEST: {}'.format(data['train'].n, data['val'].n, data['test'].n))
        print('|   DATA DIMENSIONS: {}'.format(data['train'].x.shape))
        print('----------------------------------------------------------------------')

        #hacky solution will be rewritten later
        is_big = cbfv not in ['atom2vec', 'magpie', 'oliynyk']
        extra = cbfv in ['jarvis', 'jarvis_shuffled', 'random_400']
        model = Alex_Net(input=data['input'], big_input=is_big, extra=extra)

        model.compile(loss='huber_loss', optimizer=optimizer, metrics=METRICS)
        model.summary()
        history = model.fit(
            data['train'],
            epochs=epochs,
            steps_per_epoch=floor(data['train'].n / data['train'].batch_size),
            validation_data=data['val'],
            callbacks=CALLBACKS
        )

        history_storage['train_loss'].append(history.history['loss'])
        history_storage['train_MAE'].append(history.history['mean_absolute_error'])
        history_storage['train_MSE'].append(history.history['mean_squared_error'])

        history_storage['val_loss'].append(history.history['val_loss'])
        history_storage['val_MAE'].append(history.history['val_mean_absolute_error'])
        history_storage['val_MSE'].append(history.history['val_mean_squared_error'])

        eval = model.evaluate(data['test'])
        history_storage['test_loss'].append(eval[0])
        history_storage['test_MSE'].append(eval[1])
        history_storage['test_MAE'].append(eval[2])

        #saving the training plots
        fig, ax = subplots(3, 1)
        ax[0].plot(history.history['loss'][3:], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'][3:], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['mean_squared_error'][3:], color='b', label="mean squared error")
        ax[1].plot(history.history['val_mean_squared_error'][3:], color='r', label="Validation MSE")
        legend = ax[1].legend(loc='best', shadow=True)

        ax[2].plot(history.history['mean_absolute_error'][3:], color='b', label='mean absolute error')
        ax[2].plot(history.history['val_mean_absolute_error'][3:], color='r', label='val MAE')
        legend = ax[2].legend(loc='best', shadow=True)
        filename = f'{material_prop}_{strftime("%m/%d/%Y_%H:%M:%S", localtime())}.jpg'

        file_path = join(BASE_DIR, 'Plots', cbfv, filename)
        savefig(file_path)

        #hopfully this will clear the GPU and prevent memory errors
        tf.keras.backend.clear_session()

        #clearing memory from numpy? fix to the memory error hopefully
        del data['train'].x, data['train'].y, data['test'].x, data['test'].y, data['val'].x, data['val'].y

#constructing the Dataframe and the csv file
df = DataFrame(history_storage)
df.to_csv('history_storage.csv')