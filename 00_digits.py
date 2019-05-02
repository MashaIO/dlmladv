from sklearn.datasets import load_digits
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def gen_train_test_digits():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_sc = X / 16.0
    y_cat = to_categorical(y, 10)
    X_train, X_test, y_train, y_test =     train_test_split(X_sc, y_cat, test_size=0.3,
                         random_state=0, stratify=y)
    return (X_train, y_train, X_test, y_test)

def repeat_train(model_fn, repeats=3, epochs=40,
                 verbose=0, batch_size=256):
    """
    Repeatedly train a model on (X_train, y_train),
    averaging the histories.
    
    Parameters
    ----------
    model_fn : a function with no parameters
        Function that returns a Keras model

    repeats : int, (default=3)
        Number of times the training is repeated
    
    epochs : int, (default=40)
        Number of epochs for each training run
    
    verbose : int, (default=0)
        Verbose option for the `model.fit` function
    
    batch_size : int, (default=256)
        Batch size for the `model.fit` function
        
    Returns
    -------
    mean, std : np.array, shape: (epochs, 2)
        mean : array contains the accuracy
        and validation accuracy history averaged
        over the different training runs
        std : array contains the standard deviation
        over the different training runs of
        accuracy and validation accuracy history
    """
    histories = []
    
    # repeat model definition and training
    for repeat in range(repeats):
        K.clear_session()
        model = model_fn()
        
        # train model on training data
        h = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      verbose=verbose,
                      batch_size=batch_size,
                      epochs=epochs)
        
        # append accuracy and val accuracy to list
        histories.append([h.history['acc'],
                          h.history['val_acc']])
        print(repeat, end=" ")
        
    histories = np.array(histories)
    print()
    
    # calculate mean and std across repeats:
    mean = histories.mean(axis=0)
    std = histories.std(axis=0)
    return mean, std

def base_model():
    """
    Return a fully connected model with 3 inner layers
    with 1024 nodes each and relu activation function
    """
    model = Sequential()
    model.add(Dense(1024, input_shape=(64,),
                    activation='relu'))
    model.add(Dense(1024,
                    activation='relu'))
    model.add(Dense(1024,
                    activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
def plot_mean_std(m, s):
    """
    Plot the average history as a line
    and add a colored area around it corresponding
    to +/- 1 standard deviation
    """
    plt.plot(m)
    plt.fill_between(range(len(m)), m-s, m+s, alpha=0.2)