========================================
Bayesian Optimization
========================================

Bayesian optimization is a popular method for optimizing black-box models, using a surrogate model (often a Gaussian process, GP) 
to predict performance and an acquisition function (e.g., expected improvement, EI) to select new HP configurations. 
Bayesian optimization updates the surrogate model with each new evaluation, balancing exploration and exploitation. 
While it converges efficiently, GP's sequential nature and slow performance can be limitations.

The following piece of code offers an example::

    import argparse
    from scipy.stats import qmc
    from scipy.io import savemat
    from sklearn.model_selection import train_test_split
    from imp_functions import *
    from scipy.optimize import differential_evolution
    import time
    import numpy as np

    # Setting the CML arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_size", type=int, default=100)
    argparser.add_argument("--test_size", type=int, default=50)
    argparser.add_argument("--runs", type=int)
    args = argparser.parse_args()

    # Setting CML arguments
    train_size = args.train_size
    test_size = args.test_size
    num_dim = 2

    print("###############################################", flush=True)
    print("Training size: {}".format(train_size), flush=True)
    print("###############################################", flush=True)

    # Bounds
    ub = np.array([8.0, 8.0])
    lb = np.array([0.0, 0.0])
    bounds = []
    for i in np.arange(len(lb)):
        bounds.append((lb[i], ub[i]))

    u_bounds = [8.0, 8.0]
    l_bounds = [0.0, 0.0]

    num_dim = 2
    runs = args.runs

    samples = train_size

    sampler_x_train  = qmc.Halton(d=num_dim, scramble=False)
    sample_x_train  = sampler_x_train.random(n=samples)
    x_mean = qmc.scale(sample_x_train, l_bounds, u_bounds)
    y_mean = branin(x_mean)

    x_train, x_cv, y_train, y_cv = train_test_split(x_mean, y_mean, test_size=0.2)

    x_opt = np.zeros((runs, num_dim))
    f_opt = np.zeros((runs, 1))

    epoch = np.zeros((runs, 1))
    activation = np.zeros((runs, 1))
    layer = np.zeros((runs, 1))
    neuron = np.zeros((runs, 1))
    training_nrmse = np.zeros((runs, 1))
    loss_training_cv = np.zeros((args.runs, 1))

    store_times = []

    # Running Ax multiple times
    for idx in range(args.runs):

        print("\nIteration: {}".format(idx+1), flush=True)
        
        tic = time.time()

        # Optimize the hyperparameters
        opt_params, obj = Ax(x_train, y_train, x_cv, y_cv)

        print("Optimal hyperparameters: {}".format(opt_params), flush=True)

        model, x_transform, y_transform = train(x_train, y_train, opt_params)

        # Transform the data
        x_train = x_transform.transform(x_train)
        
        # Predict at testing data
        y_pred = model(x_train)

        # Transform back to original scale
        x_train = x_transform.inverse_transform(x_train)

        y_pred = y_transform.inverse_transform(y_pred)

        best_loss = np.array(obj)
        training_loss = np.sqrt(mean_squared_error(y_train, y_pred)/np.ptp(y_train))

        print(f"Best objective (loss for training + CV): {best_loss}", flush=True)
        print("Training NRMSE: {}".format(training_loss), flush=True)

        ########################### Minimize the NN model
        # Minimum of the NN model
        result = differential_evolution(predict, bounds, mutation=0.5, recombination=0.9,
                        args=(x_transform, y_transform, model), polish=True, disp=False)

        print("Optimal x: {}".format(result.x), flush=True)
        print("Optimal f: {}".format(result.fun), flush=True)

        toc = time.time()
        print(f"Elapsed time : {toc-tic} seconds")

        epoch[idx, 0] =  opt_params['epochs']
        activation[idx, 0] = opt_params['activation']
        layer[idx, 0] = opt_params['num_hidden_layers']
        neuron[idx, 0] = opt_params['neurons']

        x_opt[idx, 0] = result.x[0]
        x_opt[idx, 1] = result.x[1]
        f_opt[idx, 0] = result.fun
        training_nrmse[idx, 0] = training_loss
        loss_training_cv[idx, 0] = best_loss

        times_grid = toc-tic
        
        if idx == 0:
            store_times.append(times_grid)
        else:
            times_grid += store_times[idx-1]
            store_times.append(times_grid)

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
