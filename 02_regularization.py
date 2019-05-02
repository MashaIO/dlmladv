#%% [markdown]
# Regularization

#%%
#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%%
import keras.backend as K

#%% [markdown]
# let's define the `repeat_train` helper function. This function expects an already created `model_fn` as input, i.e. a function that returns a model, and it repeats the following process a number of times specified by the input `repeats`:
# 
# 1. clear the session
# ```python
# K.clear_session()
# ```
# - create a model using the `model_fn`
# ```python
# model = model_fn()
# ```
# - train the model using the training data
# ```python
# h = model.fit(X_train, y_train,
#               validation_data=(X_test, y_test),
#               verbose=verbose,
#               batch_size=batch_size,
#               epochs=epochs)
# ```
# - retrieve the accuracy of the model on training data (`acc`) and test data (`val_acc`) and append the results to the `histories` array
# ```python
# histories.append([h.history['acc'], h.history['val_acc']])
# ```
# 
# Finally, the `repeat_train` function calculates the average history along with its standard deviation and returns them.

#%%
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

#%% [markdown]
# The `repeat_train` function expects an already created `model_fn` as input. Hence, let's define a new function that will create a fully connected Neural Network with 3 inner layers. We'll call this function `base_model`, since we will use this basic model for further comparison: 

#%%
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

#%% [markdown]
# >NOTE: The base model is quite big for the problem we are trying to solve. Purposefully make the model big, so that there are lots of parameters and it can overfit easily.
#%% [markdown]
# Now we repeat 5 times the training of the base (non-regularized) model using the `repeat_train` helper function: 

#%%
((m_train_base, m_test_base),
 (s_train_base, s_test_base)) = \
    repeat_train(base_model, repeats=5)

#%% [markdown]
# We can plot the histories for training and test. First, let's define an additional helper function `plot_mean_std()`,    which plots the average history as a line and add a colored area around it corresponding to +/- 1 standard deviation:

#%%
def plot_mean_std(m, s):
    """
    Plot the average history as a line
    and add a colored area around it corresponding
    to +/- 1 standard deviation
    """
    plt.plot(m)
    plt.fill_between(range(len(m)), m-s, m+s, alpha=0.2)

#%% [markdown]
# Then, let's plot the results obtained training 5 times the base model:

#%%
plot_mean_std(m_train_base, s_train_base)
plot_mean_std(m_test_base, s_test_base)

plt.title("Base Model Accuracy")
plt.legend(['Train', 'Test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.05);

#%% [markdown]
# Overfitting in this case is evident, with the test score saturating at a lower value than the training score.

#%% [markdown]
# ##Regularization
# Regularization is a common procedure in Machine Learning and it has been used to improve the performance of complex models with many parameters.

#%% [markdown]
# We start by defining a helper function that creates a model with weight regularization: we start from the function `base_model`, and we create the function `regularized_model`, adding the `kernel_regularizer` option to each layer. First of all let's import keras's `l2` regularizer function:

#%%
from keras.regularizers import l2

#%%
def regularized_model():
    """
    Return an l2-weight-regularized, fully connected
    model with 3 inner layers with 1024 nodes each
    and relu activation function.
    """
    reg = l2(0.005)
    
    model = Sequential()
    model.add(Dense(1024,
                    input_shape=(64,),
                    activation='relu',
                    kernel_regularizer=reg))
    model.add(Dense(1024,
                    activation='relu',
                    kernel_regularizer=reg))
    model.add(Dense(1024,
                    activation='relu',
                    kernel_regularizer=reg))
    model.add(Dense(10, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#%% [markdown]
# Now we compare the results of no regularization and l2-regularization. Let's repeat the training 3 times.

#%%
(m_train_reg, m_test_reg), (s_train_reg, s_test_reg) =  repeat_train(regularized_model)

#%%
plot_mean_std(m_train_base, s_train_base)
plot_mean_std(m_test_base, s_test_base)

plot_mean_std(m_train_reg, s_train_reg)
plot_mean_std(m_test_reg, s_test_reg)

plt.axhline(m_test_base.max(),
            linestyle='dashed',
            color='black')

plt.title("Regularized Model Accuracy")
plt.legend(['Base - Train', 'Base - Test',
            'l2 - Train', 'l2 - Test',
            'Max Base Test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.05)

#%% [markdown]
# With this particular dataset, weight regularization does not seem to improve the model performance.
#
# Let us use another modern technique called **Dropout**

#%% [markdown]
# ## Dropout]
# 
# ![Dropout](./assets/dropout_network.png)

#%%
from keras.layers import Dropout

#%%
def dropout_model():
    """
    Return a fully connected model
    with 3 inner layers with 1024 nodes each
    and relu activation function. Dropout can
    be applied by selecting the rate of dropout
    """

    # dropout rate of 10% at the input and 50% in the inner layers
    input_rate = 0.1
    rate = 0.5

    model = Sequential()
    model.add(Dropout(input_rate, input_shape=(64,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate))
    model.add(Dense(10, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#%% [markdown]
# Let's train 3 times our network using the `dropout_model`:

#%%
(m_train_dro, m_test_dro), (s_train_dro, s_test_dro) =     repeat_train(dropout_model)

#%%
plot_mean_std(m_train_base, s_train_base)
plot_mean_std(m_test_base, s_test_base)

plot_mean_std(m_train_dro, s_train_dro)
plot_mean_std(m_test_dro, s_test_dro)

plt.axhline(m_test_base.max(),
            linestyle='dashed',
            color='black')

plt.title("Dropout Model Accuracy")
plt.legend(['Base - Train', 'Base - Test',
            'Dropout - Train', 'Dropout - Test',
            'Max Base Test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.05);

#%% [markdown]
# Nice! Adding Dropout to our model pushed our test score above the base model for the first time (although not by much)! This is great because we didn't have to add more data. Also, notice how the training score is lower than the test score, which indicates the model is not overfitting and also there seem to be even more room for improvement if we run the training for more epochs!

#%% [markdown]
# The Dropout also suggest to use of a global constraint to further improve the behavior of a Dropout network. Constraints can be added in Keras through the `kernel_constraint` parameter available in the definition of a layer.
# 
# Let's load the `max_norm` constraint first.  It is equivalent to the sum of the square of the weights cannot be higher than certain constant.

#%%
from keras.constraints import max_norm

#%% [markdown]
# Let's define a new model function `dropout_max_norm`, that has both `dropout` and the `max_norm` constraint:

#%%
def dropout_max_norm():
    """
    Return a fully connected model with Dropout
    and Max Norm constraint.
    """
    input_rate = 0.1
    rate = 0.5
    c = 2.0
    
    model = Sequential()
    model.add(Dropout(input_rate, input_shape=(64,)))
    model.add(Dense(1024, activation='relu',
                    kernel_constraint=max_norm(c)))
    model.add(Dropout(rate))
    model.add(Dense(1024, activation='relu',
                    kernel_constraint=max_norm(c)))
    model.add(Dropout(rate))
    model.add(Dense(1024, activation='relu',
                    kernel_constraint=max_norm(c)))
    model.add(Dropout(rate))
    model.add(Dense(10, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#%% [markdown]
# As usual we can run 3 repeated trainings and average the results:

#%%
(m_train_dmn, m_test_dmn), (s_train_dmn, s_test_dmn) =     repeat_train(dropout_max_norm)

#%% [markdown]
# And plot the comparison with the base model:

#%%
plot_mean_std(m_train_base, s_train_base)
plot_mean_std(m_test_base, s_test_base)

plot_mean_std(m_train_dmn, s_train_dmn)
plot_mean_std(m_test_dmn, s_test_dmn)

plt.axhline(m_test_base.max(),
            linestyle='dashed',
            color='black')

plt.title("Dropout & Max Norm Model Accuracy")
plt.legend(['Base - Train', 'Base - Test',
            'Dropout & Max Norm - Train', 'Dropout & Max Norm - Test',
            'Max Base Test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.05);

#%% [markdown]
# In this particular case the Max Norm constraint does not seem to produce results that are qualitatively different from the simple Dropout, but there may be datasets where this constraint helps make the network converge to a better result.

#%% [markdown]
# ##Batch Normalization

#%%
from keras.layers import BatchNormalization, Activation

#%% [markdown]
# Then we define again a new model function `batch_norm_model` that adds Batch Normalization to our fully connected network defined in the `base_model`:

#%%
def batch_norm_model():
    """
    Return a fully connected model with
    Batch Normalization.

    Returns
    -------
    model : a compiled keras model
    """
    model = Sequential()
    
    model.add(Dense(1024, input_shape=(64,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#%% [markdown]
# Batch Normalization seems to work better with smaller batches, so we will run the `repeat_train` function with a smaller `batch_size`. 
# 
# Since smaller batches mean more weight updates at each epoch we will also run the training for less epochs. 
# 
# Let's do a quick back of the envelope calculation. 
# 
# We have 1257 points in the training set. Previously, we used batches of 256 points, which gives 5 weight updates per epoch, and a total of 200 updates in 40 epochs. If we reduce the batch size to 32, we will have 40 updates at each epoch, so we should run the training for only 5 epochs. 
# 
# We will actually run it a bit longer in order to see the effectiveness of Batch Normalization. 10-15 epochs will suffice to bring the model accuracy to a much higher value on the test set.

#%%
(m_train_bn, m_test_bn), (s_train_bn, s_test_bn) =     repeat_train(batch_norm_model,
                 batch_size=32,
                 epochs=15)

#%%
plot_mean_std(m_train_base, s_train_base)
plot_mean_std(m_test_base, s_test_base)

plot_mean_std(m_train_bn, s_train_bn)
plot_mean_std(m_test_bn, s_test_bn)

plt.axhline(m_test_base.max(),
            linestyle='dashed',
            color='black')

plt.title("Batch Norm Model Accuracy")
plt.legend(['Base - Train', 'Base - Test',
            'Batch Norm - Train', 'Batch Norm - Test',
            'Max Base Test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.05)
plt.xlim(0, 15);

#%% [markdown]
# Awesome! With the addition of Batch Normalization, the model converged to a solution that is better able to generalized on the Test set, i.e. it is overfitting a lot less than the base solution.