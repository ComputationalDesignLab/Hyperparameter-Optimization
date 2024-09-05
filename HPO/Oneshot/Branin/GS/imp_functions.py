import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED']=str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import GlorotUniform

import tensorflow as tf
import random

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_loss = logs["loss"]
        if val_loss <= self.threshold:
            self.model.stop_training = True

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------------------
#                           Normalization Methods
# ----------------------------------------------------------------------------

def std_scalar(data):
    """
        Method for creating std scalar object.
    """
    
    trans = StandardScaler()
    trans.fit(data)
    
    return trans

def transform(x, x_transform):
    """
        Method for transforming the input
        based on the std scalar object.
    """
    
    x = x_transform.transform(x)
    
    return x

# ----------------------------------------------------------------------------
#                             Analytical Function
# ----------------------------------------------------------------------------

def branin(x):
    """
        Method for calculating branin function.
    """

    x1 = x[:, 0].reshape(-1,1)
    x2 = x[:, 1].reshape(-1,1)
    
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)

    func = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s

    return func

# ----------------------------------------------------------------------------
#                        Training and prediction Methods
# ----------------------------------------------------------------------------

def train(x, y, parameters):
    """
        Method for training the NN for given hyperparameters.
    """

    # Getting the standard scalar object
    x_transform = std_scalar(x)
    y_transform = std_scalar(y)

    # Normalize training and testing data to zero mean and unit variance
    x = transform(x, x_transform)
    y = transform(y, y_transform)

    # Get learning rate and set the optimizer

    lr = 0.001

    opt = Adam(learning_rate=lr)

    activations = {
       1: "relu",
       2: "elu",
       3: "tanh",
       4: "sigmoid"
        }

    activation_get = activations[parameters["activation"]]
    num_hidden_layers = parameters["num_hidden_layers"]
    activation_list = [activation_get] * num_hidden_layers

    layers = [parameters["neurons"]]
    for idx in range(num_hidden_layers):
        if idx != 0:
            layers.append(layers[idx-1])

    # Predefined hyperparameters
    regularizer = None
    
    epochs = parameters["epochs"]
    
    initializer = GlorotUniform(seed=10)

    tolerance = 0.0005  # Define your tolerance
    callbacks = MyThresholdCallback(threshold=tolerance)

    # Build the NN structure 
    model = Sequential()

    # Input layer - doesn't have any activation function
    model.add(Input(shape=(x.shape[1],)))

    # Hidden layers
    for i in range(len(layers)):
        model.add(Dense(layers[i], activation=activation_list[i], activity_regularizer=regularizer, kernel_initializer=initializer))

    # Output layer
    model.add(Dense(y.shape[1]))

    # Complile the model
    model.compile(optimizer=opt, loss='mean_squared_error')

    model.fit(x, y, epochs=epochs, verbose=0, callbacks=[callbacks], use_multiprocessing=True)

    return model, x_transform, y_transform


def predict(x, x_transform, y_transform, model):
    """
        Common method for prediction from either NN.
        Input model dictates what model will be used.
    """
    # Reshaping x
    dim = x.ndim
    if dim == 1:
        x = x.reshape(1,-1)

    # Scaling, Prediction, and Rescaling
    x = transform(x, x_transform)
    y = model(x, training=False)
    y = y_transform.inverse_transform(y)

    # Reshaping the y
    if dim == 1:
        y = y.reshape(-1,)

    return y


def objective(x, y, x_cv, y_cv, parameters):
    """
        Method for computing the loss for given hyperparameters.
    """
    # Train the model
    model, x_transform, y_transform = train(x, y, parameters)

    # Transform the data
    x = transform(x, x_transform)
    x_cv = transform(x_cv, x_transform)

    # Predict at training data
    y_pred = model(x, training=False)
    y_cv_pred = model(x_cv, training=False)

    # Transform back to original scale
    y_pred = y_transform.inverse_transform(y_pred)
    y_cv_pred = y_transform.inverse_transform(y_cv_pred)

    # Calculate the rmse
    loss = np.sqrt(mean_squared_error(y, y_pred))
    loss_cv = np.sqrt(mean_squared_error(y_cv, y_cv_pred))

    return loss + loss_cv