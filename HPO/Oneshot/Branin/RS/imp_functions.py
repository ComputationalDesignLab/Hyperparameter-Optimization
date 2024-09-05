import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['PYTHONHASHSEED']=str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.special import erf
from scipy.stats import norm

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

def reset_random_seeds():
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)

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

alpha = np.zeros([4,1])
alpha[0] = 1; alpha[1] =1.2; alpha[2] = 3; alpha[3]=3.2;

A = np.zeros([4,3])
A[0,0] = 3; A[0,1] = 10; A[0,2] = 30; 
A[1,0] = 0.1; A[1,1] = 10; A[1,2] = 35;
A[2,0] = 3; A[2,1] = 10; A[2,2] = 30;
A[3,0] = 0.1; A[3,1] = 10; A[3,2] = 35;

P = np.zeros([4,3])
P[0,0] = 0.3689; P[0,1] = 0.1170; P[0,2] = 0.2673; 
P[1,0] = 0.4699; P[1,1] = 0.4387; P[1,2] = 0.7470;
P[2,0] = 0.1091; P[2,1] = 0.8732; P[2,2] = 0.5547;
P[3,0] = 0.0381; P[3,1] = 0.5743; P[3,2] = 0.8828;

def HARTMANN3(x):
    outer = 0;
    for i in range(4):
       inner = 0;
       for j in range(3):
          inner = inner + A[i,j]*(x[j]-P[i,j])**2;
       outer = outer + alpha[i] * np.exp(-inner);
    f = -outer
    return f

# ----------------------------------------------------------------------------
#                        Training and prediction Methods
# ----------------------------------------------------------------------------

def train(x, y, parameters):
    """
        Method for training the NN for given hyperparameters.
    """

    # reset_random_seeds()


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


    # Train the model
    #history = model.fit(x, y, epochs=epochs, verbose=0, callbacks=[callbacks], use_multiprocessing=True)
    #print(len(history.history['loss']))
    #print(history.history['loss'][-1])

    model.fit(x, y, epochs=epochs, verbose=0, callbacks=[callbacks], use_multiprocessing=True)

    return model, x_transform, y_transform


def predict(x, x_transform, y_transform, model):
    """
        Common method for prediction from either NN.
        Input model dictates what model will be used.
    """
    
    # Reshaping x
    # Req for scipy minimize
    dim = x.ndim
    if dim == 1:
        x = x.reshape(1,-1)

    # Scaling, Prediction, and Rescaling
    x = transform(x, x_transform)
    y = model(x, training=False)
    y = y_transform.inverse_transform(y)

    # Reshaping the y
    # Req for scipy minimize
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