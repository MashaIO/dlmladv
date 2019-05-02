#%% [markdown]
# # Data Augmentation
#

#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

#%%
with open('00_digits.py') as fin:
    exec(fin.read())

#%% [markdown]
# This class creates a generator that can apply all sorts of variations to an input image. Let's initialize it with a few parameters:
# - We'll set the `rescale` factor to `1/255` to normalize pixel values to the interval [0-1]
# - We'll set the `width_shift_range` and `height_shift_range` to ±10% of the total range
# - We'll set the `rotation_range` to ±20 degrees
# - We'll set the `shear_range` to ±0.3 degrees
# - We'll set the `zoom_range` to ±30%
# - We'll allow for `horizontal_flip` of the image
#
#%%
idg = ImageDataGenerator(rescale = 1./255,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         rotation_range = 20,
                         shear_range = 0.3,
                         zoom_range = 0.3,
                         horizontal_flip = True)
#%% [markdown]
# The next step is to create an iterator that will generate images with the image data generator. We basically need to tell where our training data are. Here we use the method `flow_from_directory`, which is useful when we have images stored in a directory, and we tell it to produce target images of size 128x128. The input folder structure need to be:
# 
#     top/
#         class_0/
#         class_1/
#         ...
# 
# Where `top` is the folder we will flow from, and the images are organized into one subfolder for each class.

#%%
train_gen = idg.flow_from_directory(
    './data/generator',
    target_size = (128, 128),
    class_mode = 'binary')

#%% [markdown]
# Let's generate a few images and display them:

#%%
plt.figure(figsize=(12, 12))

for i in range(16):
    img, label = train_gen.next()
    plt.subplot(4, 4, i+1)
    plt.imshow(img[0])

#%% [markdown]
# Great! In all of the images the squirrel is still visible and from a single image we have generated 16 different images that we can use for training!
# 
# Let's apply this technique to our digits and see if we can improve the score on the test set. We will use slightly less dramatic transformations and also fill the empty space with zeros along the border.

#%%
X_train, y_train, X_test, y_test = gen_train_test_digits()

#%%
digit_idg = ImageDataGenerator(width_shift_range=0.1,
                               height_shift_range=0.1,
                               rotation_range = 10,
                               shear_range = 0.1,
                               zoom_range = 0.1,
                               fill_mode='constant')

#%% [markdown]
# We will need to reshape our data into tensors with 4 axes, in order to use it with the `ImageDataGenerator`, so let's do it:

#%%
X_train_t = X_train.reshape(-1, 8, 8, 1)
X_test_t = X_test.reshape(-1, 8, 8, 1)

#%% [markdown]
# We can use the method `.flow` to flow directly from a dataset. We will need to provide the labels as well.

#%%
train_gen = digit_idg.flow(X_train_t, y=y_train)

#%% [markdown]
# Notice that by default the `.flow` method generates a batch of 32 images with corresponding labels:

#%%
imgs, labels = train_gen.next()


#%%
imgs.shape

#%% [markdown]
# Let's display a few of them:

#%%
plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(imgs[i,:,:,0], cmap='gray')
    plt.title(np.argmax(labels[i]))

#%% [markdown]
# As you can see the digits are deformed, due to the very low resolution of the images. Will this help our network or confuse it? Let's find out!
# 
# We will need a model that is able to deal with a tensor input, since the images are now tensors of order 4. Luckily, it's very simple to adapt our base model to have a `Flatten` layer as input:

#%%
def tensor_model():
    model = Sequential()
    model.add(Flatten(input_shape=(8, 8, 1)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#%% [markdown]
# We also need to define a new `repeat_train_generator` function that allows to train a model from a generator. We can take the original `repeat_train` function and modify it. We will follow the same procedure used in []() with 2 difference:
# 
# 
# 1. We'll define a generator that yields batches from `X_train_t` using the image data generator
# 
# 
# 2. We'll replace the `.fit` function:
# ```python
# h = model.fit(X_train, y_train,
#               validation_data=(X_test, y_test),
#               verbose=verbose,
#               batch_size=batch_size,
#               epochs=epochs)
# ```
# with the `.fit_generator` function:
# ```python
# h = model.fit_generator(train_gen,
#         steps_per_epoch=steps_per_epoch,
#         epochs=epochs,
#         validation_data=(X_test_t, y_test),
#         verbose=verbose)
# ```
# 
# 
# 
# Notice that, since we are now feeding variations of the data in the training set the concept of an _epoch_ becomes blurry. When does an epoch terminate if we flow random variations of the training data? The `model.fit_generator` function allows us to define how many `steps_per_epoch` we want. We will use the value of 5, with a `batch_size` of 256 like in most of the examples above.

#%%
def repeat_train_generator(model_fn, repeats=3,
                           epochs=40, verbose=0,
                           steps_per_epoch=5,
                           batch_size=256):
    """
    Repeatedly train a model on (X_train, y_train),
    averaging the histories using a generator.
    
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
        
    steps_per_epoch : int, (default=5)
        Steps_per_epoch for the `model.fit` function 
    
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
    # generator that flows batches from X_train_t
    train_gen = digit_idg.flow(X_train_t, y=y_train,
                               batch_size=batch_size)
    
    histories = []
    
    # repeat model definition and training
    for repeat in range(repeats):
        K.clear_session()
        model = model_fn()
        
        # to train with a generator use .fit_generator()
        h = model.fit_generator(train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(X_test_t, y_test),
            verbose=verbose)

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
# Once the function is defined, we can train it as usual:

#%%
((m_train_base, m_test_base),
 (s_train_base, s_test_base)) = \
    repeat_train(base_model, repeats=5)
#%%
(m_train_gen, m_test_gen), (s_train_gen, s_test_gen) =     repeat_train_generator(tensor_model)

#%% [markdown]
# And compare the results with our base model:

#%%
plot_mean_std(m_train_base, s_train_base)
plot_mean_std(m_test_base, s_test_base)

plot_mean_std(m_train_gen, s_train_gen)
plot_mean_std(m_test_gen, s_test_gen)

plt.axhline(m_test_base.max(),
            linestyle='dashed',
            color='black')

plt.title("Image Generator Model Accuracy")
plt.legend(['Base - Train', 'Base - Test',
            'Generator - Train', 'Generator - Test',
            'Max Base Test'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.05);

#%% [markdown]
# As you can see, the Data Augmentation process improved the performance of the model on our test set. This makes a lot of sense, because by feeding variations of the input data as training we have made the model more resilient to changes in the input features.