#%% [markdown]
# # Learning Curve
#
# A learning curve plots the behavior of the training and validation scores as a function of how much training data we fed to the model.
# 
# Let's load a simple dataset and explore how to build a learning curve. We will use the digits dataset from Scikit Learn, which is quite small. First of all we import the `load_digits` function and use it:

#%%
with open('common.py') as fin:
    exec(fin.read())

with open('matplotlibconf.py') as fin:
    exec(fin.read())

#%%
from sklearn.datasets import load_digits

#%% [markdown]
# Now let's create a variable called `digits` we'll fill as the result of calling `load_digits()`:

#%%
digits = load_digits()

#%% [markdown]
# Then we assign `digits.data` and `digits.target` to `X` and `y` respectively:

#%%
X, y = digits.data, digits.target

#%%
X.shape

#%% [markdown]
# `X` is an array of 1797 images that have been unrolled as feature vectors of length 64.

#%%
y.shape

#%% [markdown]
# In order to see the images we can always reshape them to the original 8x8 format. Let's plot a few digits:

#%%
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X.reshape(-1, 8, 8)[i], cmap='gray')
    plt.title(y[i])
plt.tight_layout()

#%%
X.min()
#%%
X.max()

#%% [markdown]
# Let's also check the data type of `X`:

#%%
X.dtype

#%% [markdown]
# it's a good practice to rescale the input so that it's close to 1. Let's do this by dividing by the maximum possible value (`16.0`):

#%%
X_sc = X / 16.0

#%% [markdown]
# `y` contains the labels as a list of digits:

#%%
y[:20]

#%% [markdown]
# Although it could appear that the digits are sorted, actually they are not:

#%%
plt.plot(y[:80], 'o-');

#%% [markdown]
# let's convert them to 1-hot encoding, to substitute the categorical column with a set of boolean columns, one for each category. First, let's import the `to_categorical` method from `keras.utils`:

#%%
from keras.utils import to_categorical

#%% [markdown]
# Then let's set the variable of `y_cat` to these categories:

#%%
y_cat = to_categorical(y, 10)

#%% [markdown]
# Now we can split the data into a training and a test set. Let's import the `train_test_split` function and call it against our data and the target categories:

#%%
from sklearn.model_selection import train_test_split

#%% [markdown]
# We will split the data with a 70/30 ratio and we will use a `random_state` here, so that we all get the exact same train/test split. We will also use the option `stratify`, to require the ratio of classes be balanced, i.e. about 10% for each class

#%%
X_train, X_test, y_train, y_test =     train_test_split(X_sc, y_cat, test_size=0.3,
                     random_state=0, stratify=y)

#%% [markdown]
# Let's double check that we have balanced the classes correctly. Since `y_test` is now a 1-hot encoded vector, we need first to recover the corresponding digits. We can do this using the function `argmax`:

#%%
y_test_classes = np.argmax(y_test, axis=1)

#%% [markdown]
# `y_test_classes` is an array of digits:

#%%
y_test_classes

#%% [markdown]
# There are many ways to count the number of each digit, the simplest is to temporarily wrap the array in a Pandas Series and use the `.value_counts()` method:

#%%
pd.Series(y_test_classes).value_counts()

#%% [markdown]
# Great! Our classes are balanced, with around 54 samples per class. Let's quickly train a model to classify these digits. First we load the necessary libraries:

#%%
from keras.models import Sequential
from keras.layers import Dense

#%% [markdown]
# We create a small, fully connected network with 64 inputs, a single inner layer with 16 nodes and 10 outputs with a Softmax activation function:

#%%
model = Sequential()
model.add(Dense(16, input_shape=(64,),
                activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%% [markdown]
# Let's also save the initial weights so that we can always re-start from the same initial configuration:

#%%
initial_weights = model.get_weights()

#%% [markdown]
# Now we fit the model on the training data for 100 epochs:

#%%
model.fit(X_train, y_train, epochs=100, verbose=0)

#%% [markdown]
# The model converged and we can evaluate the final training performance and test accuracies:

#%%
_, train_acc = model.evaluate(X_train, y_train,
                              verbose=0)
_, test_acc = model.evaluate(X_test, y_test,
                             verbose=0)

#%%
print("Train accuracy: {:0.4}".format(train_acc))
print("Test accuracy: {:0.4}".format(test_acc))

#%% [markdown]
# The performance on the test set is lower than the performance on the training set, which indicates the model is _overfitting_.
# 
# Before we start playing with different techniques to reduce overfitting, it is legitimate to ask if we simply don't have enough data to solve the problem.
#

#%%
# increasing fraction of the training data
fracs = np.linspace(0.1, 0.90, 5)
fracs

#%%
a = len(X_train) * fracs
train_sizes = list(a.astype(int))
train_sizes

#%% [markdown]
# loop over the train sizes, and for each train_size, do the following:
# * take exactly train_size data from X_train
# * reset the model to the intial weights
# * train the model using only the fraction of training data
# * evaluate the model on the fraction of training data
# * evaluate the model on the test data
# * append both scores to an arrays for plotting

#%%
train_scores = []
test_scores = []

#%%
for train_size in train_sizes:
    X_train_frac, _, y_train_frac, _ = \
        train_test_split(X_train, y_train, train_size=train_size, test_size=None, random_state = 0, stratify=y_train)
    
    model.set_weights(initial_weights)

    h = model.fit(X_train_frac, y_train_frac, verbose=0, epochs=100)

    r = model.evaluate(X_train_frac, y_train_frac, verbose=0)
    train_scores.append(r[-1])
    e = model.evaluate(X_test, y_test, verbose=0)
    test_scores.append(e[-1])
    print("Done size: ", train_size)

#%% [markdown]
# Plot the scores

#%%
plt.plot(train_sizes, train_scores, 'o-', label="Training score")
plt.plot(train_sizes, test_scores, 'o-', label="Test score")
plt.legend(loc="best")

#%% [markdown]
# It appears the test score would keep improving if we added more data.  If the plot does not like this, then you should improve the model.