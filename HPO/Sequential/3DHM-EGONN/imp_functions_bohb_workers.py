import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import pickle
import numpy as np
from hpbandster.core.worker import Worker
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import HyperBand
from hpbandster.optimizers import BOHB
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import L2
import tensorflow as tf
from scipy.special import erf
from scipy.stats import norm

os.environ['PYTHONHASHSEED']=str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs["loss"]
        if val_loss <= self.threshold:
            self.model.stop_training = True

def std_scalar(data):
    trans = StandardScaler()
    trans.fit(data)
    return trans

def transform(x, x_transform):
    return x_transform.transform(x)

# ----------------------------------------------------------------------------
#                             Analytical Function
# ----------------------------------------------------------------------------

# Constants used in the function
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
    x_transform = std_scalar(x)
    y_transform = std_scalar(y)
    x = transform(x, x_transform)
    y = transform(y, y_transform)
    lr = 0.001
    opt = Adam(learning_rate=lr)
    activations = {1: "relu", 2: "elu", 3: "tanh", 4: "sigmoid"}
    activation_get = activations[parameters["activation"]]
    num_hidden_layers = parameters["num_hidden_layers"]
    activation_list = [activation_get] * num_hidden_layers
    layers = [parameters["num_neurons"]]
    for idx in range(num_hidden_layers):
        if idx != 0:
            layers.append(layers[idx - 1])
    regularizer = None
    epochs = parameters["num_epochs"]
    initializer = GlorotUniform(seed=10)
    tolerance = 0.0005
    callbacks = MyThresholdCallback(threshold=tolerance)
    model = Sequential()
    model.add(Input(shape=(x.shape[1],)))
    for i in range(len(layers)):
        model.add(Dense(layers[i], activation=activation_list[i], activity_regularizer=regularizer, kernel_initializer=initializer))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.fit(x, y, epochs=epochs, verbose=0, callbacks=[callbacks])
    return model, x_transform, y_transform

def predict(x, x_transform, y_transform, model):
    dim = x.ndim
    if dim == 1:
        x = x.reshape(1, -1)
    x = transform(x, x_transform)
    y = model(x, training=False)
    y = y_transform.inverse_transform(y)
    if dim == 1:
        y = y.reshape(-1, )
    return y

def objective(x, y, x_cv, y_cv, parameters):
    model, x_transform, y_transform = train(x, y, parameters)
    x = transform(x, x_transform)
    x_cv = transform(x_cv, x_transform)
    y_pred = model(x, training=False)
    y_cv_pred = model(x_cv, training=False)
    y_pred = y_transform.inverse_transform(y_pred)
    y_cv_pred = y_transform.inverse_transform(y_cv_pred)
    loss = np.sqrt(mean_squared_error(y, y_pred))
    loss_cv = np.sqrt(mean_squared_error(y_cv, y_cv_pred))
    return loss + loss_cv

# ----------------------------------------------------------------------------
#                               Methods for BOHB
# ----------------------------------------------------------------------------

class MyWorker(Worker):
    def __init__(self, x_train, y_train, x_cv, y_cv, **kwargs):
        super().__init__(**kwargs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_cv = x_cv
        self.y_cv = y_cv

    def compute(self, config, budget, **kwargs):
        opt_params = {
            "num_epochs": config["epoch"],
            "activation": config["act"],
            "num_hidden_layers": config["layer"],
            "num_neurons": config["neuron"]
        }
        loss = objective(self.x_train, self.y_train, self.x_cv, self.y_cv, opt_params)
        return {
            'loss': loss,
            'info': {}
        }

def get_configspace():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter('layer', lower=1, upper=4, log=False))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter('neuron', lower=4, upper=16, log=True))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter('act', lower=1, upper=4, log=False))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter('epoch', lower=1000, upper=10000, log=False))
    return cs

def run_hpbandster(x_train, y_train, x_cv, y_cv, iteration, min_b, max_b):

    cs = get_configspace()
    bohb = BOHB(configspace=cs, run_id='bohb', nameserver='127.0.0.1', min_budget=min_b, max_budget=max_b)
    res = bohb.run(n_iterations=iteration)
    bohb.shutdown(shutdown_workers=True)
    return res

# ----------------------------------------------------------------------------
#                           Expected Improvement
# ----------------------------------------------------------------------------

def expectedImprovement(x, x_transform, y_transform, x_s2_transform, s2_transform, ymin, mean_model, var_model):
    """
        Method to calculate the expected improvement.
        Returns negative of EI so that it can be direclty used in minimization.
    """

    # mean
    yhat = predict(x, x_transform, y_transform, mean_model)
    # Standard Deviation
    sigma = np.sqrt(predict(x, x_s2_transform, s2_transform, var_model)**2)
    # Calculating standard variable
    z = (ymin - yhat)/sigma
    # Calculating standard normal cdf
    cdf = 0.5*(1 + erf(z))
    # Calculating standard normal pdf
    pdf = norm.pdf(z)
    # Calculating expected improvement
    ExpImp = np.where(sigma <= 0, 0, sigma * (z * cdf + pdf))
 
    return -ExpImp

if __name__ == "__main__":
    # Argument parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run_id', type=str, required=True)
    argparser.add_argument('--host', type=str, required=True)
    argparser.add_argument('--worker', type=int, required=True)
    args = argparser.parse_args()

    # Load training data
    with open('train_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    x_train = data['x_train']
    y_train = data['y_train']
    x_cv = data['x_cv']
    y_cv = data['y_cv']

    # Start a worker
    w = MyWorker(x_train=x_train, y_train=y_train, x_cv=x_cv, y_cv=y_cv, nameserver=args.host, run_id=args.run_id)
    w.run(background=False)

