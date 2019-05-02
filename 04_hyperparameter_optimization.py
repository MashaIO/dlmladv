#%% [markdown]
# # Hyperparameter optimization
#%% [markdown]
# Neural Network models have a lot of hyper-parameters.:
# - model architecture
#     - number of layers
#     - type of layers
#     - number of nodes
#     - activation functions
#     - ...
# - optimizer parameters
#     - optimizer type
#     - learning rate
#     - momentum
#     - ...
# - training parameters
#     - batch size
#     - learning rate scheduling
#     - number of epochs
#     - ...
#
# These parameters are called **Hyper**-parameters because they define the training experiment and the model is not allowed to change them while training. That said, they turn out to be really important in determining the success of a model in solving a particular problem.

#%% [markdown]
# ## Hyperopt and Hyperas
#%% [markdown]
# [Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library that can perform generalized hyper-parameter tuning using a technique called Bayesian Optimization.
# 
# [Hyperas](https://github.com/maxpumperla/hyperas) is a library that connects Hyperopt and Keras, making it easy to run parallel trainings of a keras model with variations in the values of the hyper-parameters.

#%% [markdown]
### Cloud based tools
#%% [markdown]
# [SigOpt](https://sigopt.com/) is a cloud based implementation of Bayesian hyperparameter search.
# 
# [AWS SageMaker](https://aws.amazon.com/sagemaker/) and [Google Cloud ML](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview) offer options for spawning parallel training experiments with different hyper-parameter combinations.
# 
# [Determined.ai](https://determined.ai/) and [Pipeline.ai](https://pipeline.ai/) also offer this feature as part of their cloud training platform.