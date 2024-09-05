=======================
Imported Funcion 
=======================

---------------------------------------------------------------------------------------------------------
imp_function.py for grid search, random search, Bayesian optimization
---------------------------------------------------------------------------------------------------------
The following piece of code include all the imported functions except for the functions from other libraries::

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

    from ax.service.ax_client import AxClient

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

    # ----------------------------------------------------------------------------
    #                        Importing Ax for using BO
    # ----------------------------------------------------------------------------

    def Ax(x, y, x_cv, y_cv):
        """
            Method for tuning the NN using Ax.
        """

        objective_name = "objective"
        
        # Set up Ax
        ax_client = AxClient(verbose_logging=False, torch_device=device)

        # Defining experiment
        ax_client.create_experiment(
            name='NN_tunning', 
            parameters=[
                {
                    "name": "epochs", 
                    "type": "range", 
                    "bounds": [1000, 10000]
                },
                {
                    "name": "neurons",
                    "type": "range",
                    "bounds": [4, 16]
                },
                {
                    "name": "num_hidden_layers",
                    "type": "range",
                    "bounds": [1, 4]
                },
                {
                    "name": "activation",
                    "type": "range",
                    "bounds": [1, 4],
                }
            ],
            objective_name=objective_name, 
            minimize=True
        )
        
        # Number of trials
        num_trials = 30

        # Perform the experiment
        for _ in range(num_trials):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=objective(x, y, x_cv, y_cv, parameters))

        # Get the best parameters
        best, res = ax_client.get_best_parameters(use_model_predictions=False)

        return best, res[0][objective_name]

--------------------------------------------------------------------------------
imp_function.py for Hyperband and Bayesian optimization Hyperband
--------------------------------------------------------------------------------
The following piece of code include all the imported functions except for the functions from other libraries::

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import argparse
    import pickle
    import numpy as np
    from hpbandster.core.worker import Worker
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import ConfigSpace as CS
    from hpbandster.optimizers import HyperBand
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.initializers import GlorotUniform
    import tensorflow as tf
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

    def branin(x):
        x1 = x[:, 0].reshape(-1, 1)
        x2 = x[:, 1].reshape(-1, 1)
        a = 1
        b = 5.1/(4*np.pi**2)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)
        func = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s
        return func

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
        
        hb = HyperBand(configspace=cs, run_id='hb', nameserver='127.0.0.1', min_budget=min_b, max_budget=max_b)
        # bohb = BOHB(configspace=cs, run_id='bohb', nameserver='127.0.0.1', min_budget=min_b, max_budget=max_b)

        res = hb.run(n_iterations=iteration)
        # res = bohb.run(n_iterations=iteration)

        hb.shutdown(shutdown_workers=True)
        # bohb.shutdown(shutdown_workers=True)
        return res

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

.. note::

	The difference between imp_function.py for HB and BOHB is under ``def run_hpbandster``. You need to specify which algorithm you want to use.
    
----------------------
Run script file
----------------------

.. note::

	You can use runscript.py to run the code multiple times and save the data. It is recommended to use to plot the box plot afterward
