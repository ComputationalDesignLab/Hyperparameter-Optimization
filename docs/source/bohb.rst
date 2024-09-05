========================================
Bayesian Optimization Hyperband
========================================

Bayesian optimization Hyperband combines Bayesian optimization's guided search with Hyperband's efficient resource allocation. 
It replaces Bayesian optimization's GP with a tree-structured Parzen estimator (TPE), 
which improves performance by using kernel density estimation to compute probability densities for high-performing HP configurations. 
Bayesian optimization Hyperband optimizes multiple HP configurations in parallel, making it highly efficient in navigating complex HP spaces.

The following piece of code offers an example::

	from imp_functions_bohb_workers import *
    from scipy.optimize import differential_evolution

    import argparse
    import os
    import subprocess
    import time
    import pickle
    from scipy.io import savemat
    from scipy.stats import qmc
    from sklearn.model_selection import train_test_split
    import numpy as np
    import logging
    import hpbandster.core.nameserver as hpns
    from hpbandster.optimizers import BOHB

    # logging.WARNING has less log
    #logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('hpbandster').setLevel(logging.WARNING)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_size", type=int, default=10)
    argparser.add_argument("--runs", type=int, default=1)
    argparser.add_argument('--n_workers', type=int, default=10)
    args = argparser.parse_args()

    # Setting bohb parameters
    iteration = 40
    min_b = 5
    max_b = 30

    print("###############################################", flush=True)
    print("Training size: {}".format(args.train_size), flush=True)
    print("Parallel worker size: {}".format(args.n_workers), flush=True)
    print("###############################################", flush=True)

    # Bounds
    ub = np.array([8.0, 8.0])
    lb = np.array([0.0, 0.0])

    bounds = [(lb[i], ub[i]) for i in np.arange(len(lb))]

    num_dim = 2
    samples = args.train_size
    sampler_x_train = qmc.Halton(d=num_dim, scramble=False)
    sample_x_train = sampler_x_train.random(n=samples)
    x = qmc.scale(sample_x_train, lb, ub)
    y = branin(x)

    # Split the training data into training and cross-validation data
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.2)

    x_opt = np.zeros((args.runs, num_dim))
    f_opt = np.zeros((args.runs, 1))
    epoch = np.zeros((args.runs, 1))
    activation = np.zeros((args.runs, 1))
    layer = np.zeros((args.runs, 1))
    neuron = np.zeros((args.runs, 1))
    training_nrmse = np.zeros((args.runs, 1))
    loss = np.zeros((args.runs, 1))

    store_times = []

    # Serialize training data to pass to workers
    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_cv': x_cv,
        'y_cv': y_cv
    }
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    # Running the optimization loop
    for idx in range(args.runs):
        print("\nIteration: {}".format(idx + 1), flush=True)

        # Start a nameserver
        NS = hpns.NameServer(run_id='bohb', host='127.0.0.1', port=None)
        NS.start()

        # Start worker processes using subprocess
        worker_processes = []
        for i in range(args.n_workers):
            # Open imp_function located in two level up from the current directory
            worker_cmd = f'python ../../imp_functions_bohb_workers.py --run_id bohb --host 127.0.0.1 --worker {i}'
            worker_processes.append(subprocess.Popen(worker_cmd, shell=True))

        # Give workers some time to start up and register
        time.sleep(5)

        tic = time.time()

        # Run hpbandster
        res = run_hpbandster(x_train, y_train, x_cv, y_cv, iteration, min_b, max_b)

        # Extract the best configuration
        best_config = res.get_id2config_mapping()[res.get_incumbent_id()]['config']
        best_loss = res.get_runs_by_id(res.get_incumbent_id())[-1]['loss']
        best_loss = np.array(best_loss)
        print(f"Best objective (loss for training + CV): {best_loss}", flush=True)
        print(f"Best hyperparameters: {best_config}", flush=True)

        opt_params = {
            "num_epochs": best_config["epoch"],
            "activation": best_config["act"],
            "num_hidden_layers": best_config["layer"],
            "num_neurons": best_config["neuron"]
        }

        # Get the model
        model, x_transform, y_transform = train(x_train, y_train, opt_params)

        # Transform the data
        x_train = x_transform.transform(x_train)
        # Predict at training data
        y_pred = model(x_train)
        # Transform back to original scale
        x_train = x_transform.inverse_transform(x_train)
        y_pred = y_transform.inverse_transform(y_pred)

        training_loss = np.sqrt(mean_squared_error(y_train, y_pred) / np.ptp(y_train))

        print("Training NRMSE: {}".format(training_loss), flush=True)

        # Minimize the NN model
        result = differential_evolution(predict, bounds, mutation=0.5, recombination=0.9,
                                        args=(x_transform, y_transform, model), polish=True, disp=False)

        print("Optimal x: {}".format(result.x), flush=True)
        print("Optimal f: {}".format(result.fun), flush=True)

        toc = time.time()
        print(f"Elapsed time : {toc - tic} seconds")

        epoch[idx, 0] = opt_params['num_epochs']
        activation[idx, 0] = opt_params['activation']
        layer[idx, 0] = opt_params['num_hidden_layers']
        neuron[idx, 0] = opt_params['num_neurons']

        x_opt[idx, 0] = result.x[0]
        x_opt[idx, 1] = result.x[1]
        f_opt[idx, 0] = result.fun
        training_nrmse[idx, 0] = training_loss
        loss[idx, 0] = best_loss

        times_bohb = toc - tic

        if idx == 0:
            store_times.append(times_bohb)
        else:
            times_grid = store_times[idx - 1]
            store_times.append(times_bohb)

        data = {
            "x": x_opt,
            "fun": f_opt,
            "training_nrmse": training_nrmse,
            "time": store_times,
            "activation": activation,
            "epoch": epoch,
            "layer": layer,
            "neuron": neuron,
            "loss" : loss,
        }

        savemat("result.mat", data)

        # Shutdown the optimizer and nameserver
        NS.shutdown()

        # Terminate worker processes
        for p in worker_processes:
            p.terminate()
            p.wait()
