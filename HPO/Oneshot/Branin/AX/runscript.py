# Importing the necessary libraries
import os
import numpy as np
import time

# Variables
min_sample_size = 10
max_sample_size = 50
num_samples = 5
samples = np.linspace(min_sample_size, max_sample_size, num_samples, dtype=int)
num_runs = 10
directory = "output_Branin_ax_new"

# Create a directory to store the results
if directory == None:
    print("Please provide a directory name")
    exit()
elif os.path.exists(directory):
    os.system("rm -rf {}".format(directory))
    os.mkdir(directory)
else:
    os.mkdir(directory)

times = []
# Optimize the hyperparameters for different sample sizes
for sample in samples:
#    tic = time.time()
    os.mkdir("{}/{}".format(directory, sample))
    os.chdir("{}/{}".format(directory, sample))
    #os.system("python ../../hp_opt.py --train_size {} --test_size 50 --runs {} >> log.txt".format(sample, num_runs))
    os.system("python ../../hp_opt.py --train_size {} --test_size 50 --runs {} >> log.txt".format(sample, num_runs))
    os.chdir("../..")

#    toc = time.time()
#    times.append(toc-tic)
    
#    print(f"Elapsed time for {sample} using Ax : {toc-tic} seconds")

#print(f"Elapsed time using Ax : {times} seconds")

