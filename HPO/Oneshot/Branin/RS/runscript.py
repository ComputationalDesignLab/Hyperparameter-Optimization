# Importing the necessary libraries
import os
import numpy as np
import time



'''
# RS: Oneshot-Branin
'''
# Variables
min_sample_size = 10
max_sample_size = 50
num_samples = 5
samples = np.linspace(min_sample_size, max_sample_size, num_samples, dtype=int)
num_runs = 10
directory_gs = "output_Branin_rs_new"

# Create a directory to store the results
if directory_gs == None:
    print("Please provide a directory name")
    exit()
elif os.path.exists(directory_gs):
    os.system("rm -rf {}".format(directory_gs))
    os.mkdir(directory_gs)
else:
    os.mkdir(directory_gs)

times_rs = []
# Optimize the hyperparameters for different sample sizes
for sample in samples:
    tic = time.time()
    os.mkdir("{}/{}".format(directory_gs, sample))
    os.chdir("{}/{}".format(directory_gs, sample))
    os.system("python ../../rs.py --train_size {} --test_size 50 --runs {} >> log.txt".format(sample, num_runs))
    os.chdir("../..")

    toc = time.time()
    times_rs.append(toc-tic)
    
    print(f"Elapsed time for {sample} using RS : {toc-tic} seconds")

print(f"Elapsed time using RS : {times_rs} seconds")


