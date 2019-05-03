#coding=utf-8

from __future__ import print_function

try:
    import numpy as np
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

from __future__ import print_function

try:
    import numpy as np
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

"""
Data providing function:

This function is separated from create_model() so that hyperopt
won't reload data for each evaluation run.
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
nb_classes = 10
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


def keras_fmin_fnct(space):

    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(space['Dropout']))
    model.add(Dense(space['Dense']))
    model.add(Activation(space['Activation']))
    model.add(Dropout(space['Dropout_1']))

    # If we choose 'four', add an additional fourth layer
    if space['Dropout_2'] == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add(space['add'])
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=space['optimizer'])

    result = model.fit(x_train, y_train,
              batch_size=space['batch_size'],
              epochs=2,
              verbose=2,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'Dropout': hp.uniform('Dropout', 0, 1),
        'Dense': hp.choice('Dense', [256, 512, 1024]),
        'Activation': hp.choice('Activation', ['relu', 'sigmoid']),
        'Dropout_1': hp.uniform('Dropout_1', 0, 1),
        'Dropout_2': hp.choice('Dropout_2', ['three', 'four']),
        'add': hp.choice('add', [Dropout(0.5), Activation('linear')]),
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'batch_size': hp.choice('batch_size', [64, 128]),
    }
