================================================================================
Bayesian Optimization Hyperband (Sequential)
================================================================================

Within the EGONN framework, different HP tuning frequencies are explored: maintaining initially tuned HPs (``HPO-static``), tuning every five iterations (``HPO-5itr``), and tuning every iteration (``HPO-1itr``). These strategies aim to balance dynamic HP adjustments with time efficiency. HPO is conducted using **Bayesian optimization Hyperband**, building on the successful application of **Bayesian optimization Hyperband** in the one-shot method with the two-dimensional Branin function.

The EGONN method starts with an initial sample size of 10 and adds 45 infill points, using a Halton sequence to ensure consistent starting points. 
The study compares various HPO strategies within EGONN against the traditional EGO method as a benchmark. 
Initially, all HPs are optimized, followed by a focused optimization of a subset based on preliminary findings. 
The evaluation highlights convergence performance and time efficiency across the methods

The following piece of code offers an example::

    from imp_functions_bohb_workers import *
    from scipy import optimize

    import argparse
    import subprocess
    import time
    import pickle
    from scipy.io import savemat
    from scipy.stats import qmc
    from sklearn.model_selection import train_test_split
    import numpy as np
    import logging
    import hpbandster.core.nameserver as hpns

    #logging.WARNING has less log
    # logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('hpbandster').setLevel(logging.WARNING)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_size", type=int, default=10)
    argparser.add_argument("--budget", default=50, type=int)
    argparser.add_argument("--runs", type=int, default=1)
    argparser.add_argument('--n_workers', type=int, default=10)
    args = argparser.parse_args()

    # Setting bohb parameters
    iteration = 40
    min_b = 5
    max_b = 30

    # Number of points/dim
    num_dim = 3 # no of dimensions
    num_mean_var = 30 
    num_mean = 10
    num_var = num_mean_var - num_mean
    budget = args.budget # Maximum number of iterations/infill points

    print("###############################################", flush=True)
    print("Mean training size: {}".format(num_mean), flush=True)
    print("Variance training size: {}".format(num_var), flush=True)
    print("Budget: {}".format(budget), flush=True)
    print("Parallel worker size: {}".format(args.n_workers), flush=True)
    print("###############################################", flush=True)

    # Bounds
    ub = np.array([1, 1, 1]) # upper bounds for both dimensions
    lb = np.array([0, 0, 0]) # lower bounds for both dimensions

    bounds = []
    for i in np.arange(len(lb)):
        bounds.append((lb[i], ub[i]))

    u_bounds = [1, 1, 1]
    l_bounds = [0, 0, 0]

    sampler_mean_var  = qmc.Halton(d=num_dim, scramble=False)
    sample_mean_var = sampler_mean_var.random(n=num_mean_var)
    x_mean_var = qmc.scale(sample_mean_var, l_bounds, u_bounds)

    x_mean = x_mean_var[:num_mean] 
    y_mean = np.array([HARTMANN3(x) for x in x_mean])

    x_var = x_mean_var[num_mean:]
    y_var = np.array([HARTMANN3(x) for x in x_var])

    # Storing the data
    data = {
        'x': x_mean,
        'y': y_mean
    }

    # Saving the data
    savemat('doe.mat', data)

    training_nrmse = np.zeros((budget, 1))
    training_nrmse_mean = np.zeros((budget, 1))
    training_nrmse_var = np.zeros((budget, 1))
    store_times = []

    activation = np.zeros((budget, 1))
    epoch = np.zeros((budget, 1))
    layer = np.zeros((budget, 1))
    neuron = np.zeros((budget, 1))

    var_activation = np.zeros((budget, 1))
    var_epoch = np.zeros((budget, 1))
    var_layer = np.zeros((budget, 1))
    var_neuron = np.zeros((budget, 1))

    var_learning_rate = np.zeros((budget, 1))
    var_activation = np.zeros((budget, 1))
    var_epoch = np.zeros((budget, 1))

    loss = np.zeros((budget, 1))
    var_loss = np.zeros((budget, 1))

    # ----------------------------------------------------------------------------
    #                               EGONN Loop
    # ----------------------------------------------------------------------------

    # n = 1
    n = 5

    for itr in range(budget):

        print("\nInfill iteration: {}".format(itr + 1))

        print("Optimizing hyperparameters and training mean NN", flush=True)

        tic = time.time()

        ######### Train the mean NN

        if itr == 0:
        # if itr % n == 0:

            # Split the training data into training and cross-validation data
            x_train, x_cv, y_train, y_cv = train_test_split(x_mean, y_mean, test_size=0.2)
            # Serialize training data to pass to workers
            data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_cv': x_cv,
                'y_cv': y_cv
            }
            with open('train_data.pkl', 'wb') as f:
                pickle.dump(data, f)

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
            mean_NN, x_transform, y_transform = train(x_train, y_train, opt_params)

        else:
            mean_NN, x_transform, y_transform = train(x_mean, y_mean, opt_params)


        ########## Train the var NN

        print("Optimizing hyperparameters and training var NN", flush=True)
        # Predict value at x_mean
        y_mean_pred = predict(x_mean, x_transform, y_transform, mean_NN)
        # Calculate the error (s2) for first dataset
        error_mean = y_mean - y_mean_pred
        # Rescaling the predicted y at x_var for calculating the error
        y_var_pred = predict(x_var, x_transform, y_transform, mean_NN)
        # Calculate the error (s2) for second dataset
        error_var = y_var - y_var_pred
        # Stack the error data
        s2_var = np.vstack((error_mean, error_var))

        if itr == 0:
        # if itr % n == 0:

            # Split the training data into training and cross-validation data
            x_train, x_cv, y_train, y_cv = train_test_split(np.vstack((x_mean, x_var)), s2_var, test_size=0.2)
            # Serialize training data to pass to workers
            data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_cv': x_cv,
                'y_cv': y_cv
            }
            with open('train_data.pkl', 'wb') as f:
                pickle.dump(data, f)

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

            # Run hpbandster
            res = run_hpbandster(x_train, y_train, x_cv, y_cv, iteration, min_b, max_b)

            # Extract the best configuration
            best_config_var = res.get_id2config_mapping()[res.get_incumbent_id()]['config']
            best_loss_var = res.get_runs_by_id(res.get_incumbent_id())[-1]['loss']
            best_loss_var = np.array(best_loss_var)
            print(f"Best objective (loss for training + CV): {best_loss_var}", flush=True)
            print(f"Best hyperparameters: {best_config_var}", flush=True)

            opt_params_var = {
                "num_epochs": best_config_var["epoch"],
                "activation": best_config_var["act"],
                "num_hidden_layers": best_config_var["layer"],
                "num_neurons": best_config_var["neuron"]
            }

            # Get the model
            var_NN, x_s2_transform, s2_transform = train(np.vstack((x_mean, x_var)), s2_var, opt_params_var)

        else:
            var_NN, x_s2_transform, s2_transform = train(np.vstack((x_mean, x_var)), s2_var, opt_params_var)

        toc = time.time()
        times_bohb = toc - tic
        store_times.append(times_bohb)
        print(f"Elapsed time for HPO : {times_bohb} seconds")

        # Transform the data
        x_train = x_transform.transform(x_train)
        # Predict at training data
        y_pred = mean_NN(x_train)
        # Transform back to original scale
        x_train = x_transform.inverse_transform(x_train)
        y_pred = y_transform.inverse_transform(y_pred)

        training_loss = np.sqrt(mean_squared_error(y_train, y_pred) / np.ptp(y_train))

        s2_var_pred = predict(np.vstack((x_mean, x_var)), x_s2_transform, s2_transform, var_NN)
        training_loss_mean = np.sqrt(mean_squared_error(y_mean, y_mean_pred)/np.ptp(y_mean))
        training_loss_var = np.sqrt(mean_squared_error(s2_var, s2_var_pred)/np.ptp(s2_var))

        print("Training NRMSE for mean NN: {}".format(training_loss), flush=True)
        print("Training NRMSE2 for mean NN: {}".format(training_loss_mean), flush=True)
        print("Training NRMSE for var NN: {}".format(training_loss_var), flush=True)

        # ----------------------------------------------------------------------------
        #                           Minimizing the EI
        # ----------------------------------------------------------------------------

        print("######################## Adding Infill {} of {}".format(itr + 1, budget), flush=True)

        # Finding feasible y_min req for EI calculation
        ymin = np.min(y_mean)
        index = np.where(y_mean == ymin)[0][0]
        xmin = x_mean[index,:]

        print("Best objective found: {}".format(ymin), flush=True)
        print("Best objective found at: {}".format(xmin), flush=True)

        ######### Optimize the expected improvement

        print("Maximizing EI", flush=True)

        # Maximize the EI
        ei_opt = optimize.differential_evolution(expectedImprovement, bounds, popsize=50, mutation=0.5, recombination=0.9,
                    args=(x_transform, y_transform, x_s2_transform, s2_transform, ymin, mean_NN, var_NN), polish=False)

        # Computing function value at new x
        x_new = ei_opt.x.reshape(1,-1)
        #f_new = HARTMANN3(x_new).reshape(1,-1)
        f_new = np.array([HARTMANN3(x) for x in x_new]).reshape(1,-1)

        print("New infill point: {}".format(ei_opt.x), flush=True)
        print("max EI: {}".format(-ei_opt.fun), flush=True)

        # ----------------------------------------------------------------------------
        #                               Save the data
        # ----------------------------------------------------------------------------

        # Storing the ei_opt and corresponding x
        if itr == 0:
            x_ei_optimums = ei_opt.x.reshape(1,-1)
            y_ei_optimums = -ei_opt.fun.reshape(-1,)
        else:
            x_ei_optimums = np.vstack((x_ei_optimums, ei_opt.x.reshape(1,-1)))
            y_ei_optimums = np.append(y_ei_optimums, -ei_opt.fun.reshape(-1,))

        if itr == 0:
            x_min_data = xmin.reshape(1,-1)
            y_min_data = ymin.reshape(-1,)
        else:
            x_min_data = np.vstack((x_min_data, xmin.reshape(1,-1)))
            y_min_data = np.append(y_min_data, ymin.reshape(-1,))

        # Appending the data for next iteration
        x_mean = np.vstack((x_mean, x_new))
        y_mean = np.vstack((y_mean, f_new))

        if itr == 0:
        # if itr % n == 0:

            epoch[itr, 0] = opt_params['num_epochs']
            activation[itr, 0] = opt_params['activation']
            layer[itr, 0] = opt_params['num_hidden_layers']
            neuron[itr, 0] = opt_params['num_neurons']

            var_epoch[itr, 0] = opt_params_var['num_epochs']
            var_activation[itr, 0] = opt_params_var['activation']
            var_layer[itr, 0] = opt_params_var['num_hidden_layers']
            var_neuron[itr, 0] = opt_params_var['num_neurons']

            training_nrmse[itr, 0] = training_loss
            training_nrmse_mean[itr, 0] = training_loss_mean
            training_nrmse_var[itr, 0] = training_loss_var

            loss[itr, 0] = best_loss
            var_loss[itr, 0] = best_loss_var


        data = {
            'x_data_min': x_min_data,
            'f_data_min': y_min_data,
            'xei': x_ei_optimums,
            'yei': y_ei_optimums,
            'y': y_mean,
            "activation": activation,
            "epoch": epoch,
            "layer": layer,
            "neuron": neuron,
            "var_activation": var_activation,
            "var_epoch": var_epoch,
            "var_layer": var_layer,
            "var_neuron": var_neuron,
            "training_nrmse_mean": training_nrmse,
            "training_nrmse_var": training_nrmse_var,
            "training_nrmse_mean2": training_nrmse_mean,
            "time": store_times,
            "var_loss" : var_loss,
            "loss" : loss,
        }

        savemat("result.mat", data)

        # Shutdown the optimizer and nameserver
        NS.shutdown()

        # Terminate worker processes
        for p in worker_processes:
            p.terminate()
            p.wait()



.. note::

	If you want to test HPO-static, you should set **if loop** as ``if itr % n == 0:``. If you want to test HPO-1itr and HPO-5itr, you should set **n** in line 101 and set **if loop** as ``if itr == 0:``.

    