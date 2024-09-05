# Importing the necessary libraries
import os
import numpy as np
import time

# Variables
num_runs = 20
directory = "output_3DHM_BOHB_static"

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
for run in range(num_runs):
    tic = time.time()
    os.mkdir("{}/{}".format(directory, run+1))
    os.chdir("{}/{}".format(directory, run+1))

    os.system("python ../../bohb_workers.py --budget 45 >> log.txt")
    os.chdir("../..")

    toc = time.time()
    times.append(toc-tic)
    
    print(f"Elapsed time for egonn for {num_runs} runs : {toc-tic} seconds")

print(f"Elapsed time using BOHB for 1itr: {times} seconds")

