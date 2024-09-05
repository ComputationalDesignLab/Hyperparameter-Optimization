# Importing the libraries
import argparse
import numpy as np
from scipy.io import savemat
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from imp_functions import *
from scipy.optimize import differential_evolution
import time

# Setting the CML arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--train_size", type=int, default=100)
argparser.add_argument("--test_size", type=int, default=50)
argparser.add_argument("--runs", type=int, default=1)
args = argparser.parse_args()

# Setting CML arguments
train_size = args.train_size
test_size = args.test_size
runs = args.runs

print("###############################################", flush=True)
print("Training size: {}".format(train_size), flush=True)
print("###############################################", flush=True)

########################### Create architectures for random search

# 4*4*4*4=256 which is the architectures for grid search
pool = 256 

architectures = np.zeros((pool, 4))

for i in np.arange(pool):
    w = np.round(np.random.uniform(low=1000, high=10000, size=1))
    x = np.round(np.random.uniform(low=4, high=16, size=1))
    y = np.round(np.random.uniform(low=1, high=4, size=1))
    z = np.round(np.random.uniform(low=1, high=4, size=1))
    architectures[i] = np.hstack((w, x, y, z))
    
architectures = architectures.astype(int)

########################### Create the dataset
# Bounds
ub = np.array([8.0, 8.0])
lb = np.array([0.0, 0.0])

bounds = []
for i in np.arange(len(lb)):
    bounds.append((lb[i], ub[i]))

u_bounds = [8.0, 8.0]
l_bounds = [0.0, 0.0]

num_dim = 2
samples = train_size

sampler_x_train  = qmc.Halton(d=num_dim, scramble=False)
sample_x_train  = sampler_x_train.random(n=samples)
x_mean = qmc.scale(sample_x_train, l_bounds, u_bounds)
y_mean = branin(x_mean)

x_train = x_mean
y_train = y_mean

x_train, x_cv, y_train, y_cv = train_test_split(x_mean, y_mean, test_size=0.2)

x_opt = np.zeros((runs, num_dim))
f_opt = np.zeros((runs, 1))

epoch = np.zeros((runs, 1))
activation = np.zeros((runs, 1))
layer = np.zeros((runs, 1))
neuron = np.zeros((runs, 1))
training_nrmse = np.zeros((runs, 1))
loss_training_cv = np.zeros((runs, 1))

store_times = []

########################### Running grid search multiple times

for idx in range(runs):
    
    print("\nIteration: {}".format(idx+1), flush=True)

    tic = time.time()

    loss = 10000
    
    for architecture in architectures:

        parameters = {}
        parameters['epochs'] = architecture[0]
        parameters['neurons'] = architecture[1]
        parameters['num_hidden_layers'] = architecture[2]
        parameters['activation'] = architecture[3]

        # Optimize the hyperparameters
        obj = objective(x_train, y_train, x_cv, y_cv, parameters)

        # Check if the objective is less than the previous objective
        if obj < loss:
            loss = obj
            opt_params = parameters
            
    best_loss = np.array(loss)

    print(f"Best objective (loss for training + CV): {best_loss}", flush=True)
    print("Optimal hyperparameters: {}".format(opt_params), flush=True)

    # Get the model
    model, x_transform, y_transform = train(x_train, y_train, opt_params)

    # Transform the data
    x_train = x_transform.transform(x_train)
    
    # Predict at testing data
    y_pred = model(x_train)

    # Transform back to original scale
    x_train = x_transform.inverse_transform(x_train)
    y_pred = y_transform.inverse_transform(y_pred)

    training_loss = np.sqrt(mean_squared_error(y_train, y_pred)/np.ptp(y_train))

    print("Training NRMSE: {}".format(training_loss), flush=True)

    ########################### Minimize the NN model

    # Minimum of the NN model
    result = differential_evolution(predict, bounds, mutation=0.5, recombination=0.9,
                    args=(x_transform, y_transform, model), polish=True, disp=False)
    
    print("Optimal x: {}".format(result.x), flush=True)
    print("Optimal f: {}".format(result.fun), flush=True)

    toc = time.time()
    print(f"Elapsed time : {toc-tic} seconds")

    epoch[idx, 0] =  parameters['epochs']
    activation[idx, 0] = parameters['activation']
    layer[idx, 0] = parameters['num_hidden_layers']
    neuron[idx, 0] = parameters['neurons']
    x_opt[idx, 0] = result.x[0]
    x_opt[idx, 1] = result.x[1]
    f_opt[idx, 0] = result.fun
    training_nrmse[idx, 0] = training_loss
    loss_training_cv[idx, 0] = best_loss

    times_rs = toc-tic
    
    if idx == 0:
        store_times.append(times_rs)
    else:
        times_rs += store_times[idx-1]
        store_times.append(times_rs)

    data = {
        "x": x_opt,
        "fun": f_opt,
        "training_nrmse": training_nrmse,
        "time": store_times,
        "activation": activation,
        "epoch": epoch,
        "layer": layer,
        "neuron": neuron,
        "loss" : loss_training_cv,
    }

    savemat("result.mat".format(idx), data)
