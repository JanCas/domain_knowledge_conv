import string
from pandas import read_csv, DataFrame
from settings import BASE_DIR, batch_size
from os.path import join
from numpy import array, sum, zeros, pad, floor, ceil
from re import findall
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_test_val(material_prop: string) -> dict:
    """
    gets the train/test/val data and puts them into a dataframe
    adds 2 coloumns to the dataframe 'chemical_form'
                                     'num_elem' which contains the number of elements added up
    :param material_prop: name of the folder in 'material_properties
    :return: {'train': train_df, 'test': test_df, 'val': val_df}
    """

    main_dir = join(BASE_DIR, 'data/material_properties', material_prop)

    test_df = read_csv(join(main_dir, 'test.csv'))
    test_df['chemical_form'] = test_df.apply(lambda x: x[0].split('_')[0], axis=1)
    test_df['num_elem'] = test_df.apply(lambda x: sum(ceil(array(findall('[0-9]*\.?[0-9]+', x[2]), dtype='float'))),
                                        axis=1)

    train_df = read_csv(join(main_dir, 'train.csv'))
    train_df['chemical_form'] = train_df.apply(lambda x: x[0].split('_')[0], axis=1)
    train_df['num_elem'] = train_df.apply(lambda x: sum(ceil(array(findall('[0-9]*\.?[0-9]+', x[2]), dtype='float'))),
                                          axis=1)

    val_df = read_csv(join(main_dir, 'val.csv'))
    val_df['chemical_form'] = val_df.apply(lambda x: x[0].split('_')[0], axis=1)
    val_df['num_elem'] = val_df.apply(lambda x: sum(ceil(array(findall('[0-9]*\.?[0-9]+', x[2]), dtype='float'))),
                                      axis=1)

    return {'train': train_df, 'test': test_df, 'val': val_df}


def find_input_size(x: dict) -> int:
    """
    finds the dimensions of the biggest matrix in any of the datasets
    This makes sure the right size is used when the model is compiled
    :param x:
    :return:
    """

    y_init = 0
    for elem in x.values():
        y_init = elem['num_elem'].max() if elem['num_elem'].max() > y_init else y_init

    return y_init


def get_train_test_val_X_vector(cbfv: string, y: dict) -> array:
    """
    reads the desired feature vector and pads it to the needed size to make it a trainable rank 2 tensor
    :param cbfv:
    :return:
    """
    dir = join(BASE_DIR, 'data/element_properties', f'{cbfv}.csv')
    train, test, val = y.values()
    chemical_form_train_list = train['chemical_form']
    chemical_form_test_list = test['chemical_form']
    chemical_form_val_list = val['chemical_form']

    X_df = read_csv(dir).T
    X_df.columns = X_df.iloc[0]
    X_df = X_df[1:]

    final_size = int(find_input_size(y))

    index = 0
    train_x_vector = zeros((len(train), final_size, len(X_df)))
    for form_train in chemical_form_train_list:
        try:
            train_x_vector[index] = get_x_with_chemical_formula(x_df=X_df, final_size=(final_size, len(X_df)),
                                                            chem_form=form_train)
        except KeyError:
            print('the feature vector is missing a element in the formula {}'.format(form_train))
        index += 1
    print('forming the train x vector done')

    index = 0
    test_x_vector = zeros((len(test), final_size, len(X_df)))
    for form_test in chemical_form_test_list:
        try:
            test_x_vector[index] = get_x_with_chemical_formula(x_df=X_df, final_size=(final_size, len(X_df)),
                                                           chem_form=form_test)
        except KeyError:
            print('the feature vector is missing a element in the formula {}'.format(form_test))
        index += 1
    print('forming the test x vector done')

    index = 0
    val_x_vector = zeros((len(val), final_size, len(X_df)))
    for form_val in chemical_form_val_list:
        try:
            val_x_vector[index] = get_x_with_chemical_formula(x_df=X_df, final_size=(final_size, len(X_df)),
                                                          chem_form=form_val)
        except KeyError:
            print('the feature vector is missing a element in the formula {}'.format(form_val))
        index += 1
    print('forming the val x vector done')

    # reshaping the vectors into rank 4 vectors (samples, height, width, channels)
    train_x_vector = train_x_vector.reshape(train_x_vector.shape[0], train_x_vector.shape[1],
                                            train_x_vector.shape[2], 1)
    test_x_vector = test_x_vector.reshape(test_x_vector.shape[0], test_x_vector.shape[1],
                                          test_x_vector.shape[2], 1)
    val_x_vector = val_x_vector.reshape(val_x_vector.shape[0], val_x_vector.shape[1],
                                        val_x_vector.shape[2], 1)

    return {'train': train_x_vector, 'test': test_x_vector, 'val': val_x_vector}


def get_x_with_chemical_formula(x_df: DataFrame, final_size: tuple, chem_form: string) -> array:
    """
    constructing one x matrix based on a chemical formulas and padding it in to the desired size
    :param x_df:
    :param final_size:
    :param chem_form:
    :return:
    """
    # creatin dictionary for the chemical formula
    form = dict(findall('([A-Z][a-z]?)([0-9]*\.?[0-9]+)', chem_form))
    form.update((k, int(ceil(float(v)))) for k, v in form.items())

    chem_form_array = zeros((int(sum(list(form.values()))), final_size[1]))

    row_index = 0
    for elem, amount in form.items():
        elem_np = x_df[elem].to_numpy()
        for i in range(amount):
            # create the x_vector
            chem_form_array[row_index] = elem_np
            row_index += 1

    # calculate the padding dimensions
    top_pad = int(floor((final_size[0] - chem_form_array.shape[0]) / 2.0))
    bottom_pad = int(ceil((final_size[0] - chem_form_array.shape[0]) / 2.0))

    chem_form_array = pad(chem_form_array, ((top_pad, bottom_pad), (0, 0)), constant_values=(0, 0))

    return chem_form_array


def generate_image_data_generators(material_prop: string, cbfv: string) -> dict:
    """
    generate the keras image preprocessing whcih will be used in the fit function
    :return:
    """
    y = get_train_test_val(material_prop)
    x = get_train_test_val_X_vector(cbfv=cbfv, y=y)

    # creating y_labels
    train_label = y['train']['target'].to_numpy()
    test_label = y['test']['target'].to_numpy()
    val_label = y['val']['target'].to_numpy()

    image_gen = ImageDataGenerator()

    train_gen = image_gen.flow(x=x['train'], y=train_label, batch_size=batch_size)
    test_gen = image_gen.flow(x=x['test'], y=test_label, batch_size=batch_size)
    val_gen = image_gen.flow(x=x['val'], y=val_label, batch_size=batch_size)

    keras_input = Input(shape=(x['train'].shape[1], x['train'].shape[2], x['train'].shape[3]))
    return {'train': train_gen, 'test': test_gen, 'val': val_gen, 'input': keras_input}
