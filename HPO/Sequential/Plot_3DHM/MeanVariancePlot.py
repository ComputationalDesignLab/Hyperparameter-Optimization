from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.lines as mlines

'''
Load data for kriging
'''
data = {}  # Initialize an empty dictionary

# Loop over the number of files
for i in range(1, 11):
    # Formulate the file path
    file_path = f'output_3DHM_KRG/{i}/result.mat'
    
    # Load the data and store it in the dictionary
    mat_data = loadmat(file_path)
    
    # Store the components in a dictionary
    data[f'data{i}'] = {'x1_opt': mat_data['x_data_min'][:, 0], 'x2_opt': mat_data['x_data_min'][:, 1], 'x3_opt': mat_data['x_data_min'][:, 2], 
                        'f_opt': mat_data['f_data_min'].reshape(-1,1), 'time': mat_data['time'].reshape(-1,1), 'loss': mat_data['error']
                        }
# Access the list by infill_x1_opt['infill0_x1_opt'], infill_x1_opt['infill1_x1_opt'], and so on.
infill_x1_opt = {}
infill_x2_opt = {}
infill_x3_opt = {}
infill_f_opt = {}
infill_time = {}
infill_loss = {}

for i in range(45):
    infill_x1_opt[f'infill{i}_x1_opt'] = []
    infill_x2_opt[f'infill{i}_x2_opt'] = []
    infill_x3_opt[f'infill{i}_x3_opt'] = []
    infill_f_opt[f'infill{i}_f_opt'] = []
    infill_time[f'infill{i}_time'] = []
    infill_loss[f'infill{i}_loss'] = []
    for j in range(1,11):
        infill_x1_opt[f'infill{i}_x1_opt'].append(data[f'data{j}']['x1_opt'][i])
        infill_x2_opt[f'infill{i}_x2_opt'].append(data[f'data{j}']['x2_opt'][i])
        infill_x3_opt[f'infill{i}_x3_opt'].append(data[f'data{j}']['x3_opt'][i])
        infill_f_opt[f'infill{i}_f_opt'].append(data[f'data{j}']['f_opt'][i][0])
        infill_time[f'infill{i}_time'].append(data[f'data{j}']['time'][i][0])
        infill_loss[f'infill{i}_loss'].append(data[f'data{j}']['loss'][i][0])

x1_opts = []
x2_opts = []
x3_opts = []
f_opts = []
times = []
losses = []

for i in range(45):
    x1_opts.append(np.array(infill_x1_opt[f'infill{i}_x1_opt']).reshape(-1,1))
    x2_opts.append(np.array(infill_x2_opt[f'infill{i}_x2_opt']).reshape(-1,1))
    x3_opts.append(np.array(infill_x3_opt[f'infill{i}_x3_opt']).reshape(-1,1))
    f_opts.append(np.array(infill_f_opt[f'infill{i}_f_opt']).reshape(-1,1))
    times.append(np.array(infill_time[f'infill{i}_time']).reshape(-1,1))
    losses.append(np.array(infill_loss[f'infill{i}_loss']).reshape(-1,1))

x1_opts = np.concatenate(x1_opts)
x2_opts = np.concatenate(x2_opts)
x3_opts = np.concatenate(x3_opts)
f_opts = np.concatenate(f_opts, axis=1)
times = np.concatenate(times, axis=1)
losses = np.concatenate(losses, axis=1)

# Compute means and standard deviations for each number of samples
mean_x1_opts = np.mean(x1_opts, axis=0)
std_x1_opts = np.std(x1_opts, axis=0)
mean_x2_opts = np.mean(x2_opts, axis=0)
std_x2_opts = np.std(x2_opts, axis=0)
mean_x3_opts = np.mean(x3_opts, axis=0)
std_x3_opts = np.std(x3_opts, axis=0)

mean_f_opts_krg = np.mean(f_opts, axis=0)
std_f_opts_krg = np.std(f_opts, axis=0)
mean_times_krg = np.mean(times, axis=0)
std_times_krg= np.std(times, axis=0)
mean_losses_krg = np.mean(losses, axis=0)
std_losses_krg = np.std(losses, axis=0)

'''
Load data for BOHB-1itr
'''
data = {}  # Initialize an empty dictionary

# Loop over the number of files
for i in range(1, 11):
    # Formulate the file path
    file_path = f'output_3DHM_BOHB_1itr/{i}/result.mat'
    
    # Load the data and store it in the dictionary
    mat_data = loadmat(file_path)
    
    # Store the components in a dictionary
    data[f'data{i}'] = {
                        'f_opt': mat_data['f_data_min'].reshape(-1,1), 'time': mat_data['time'].reshape(-1,1), 'loss': mat_data['loss'],
                        'layer': mat_data['layer'], 'neuron': mat_data['neuron'], 'epoch': mat_data['epoch'], 'activation': mat_data['activation']
                        }

# Access the list by infill_x1_opt['infill0_x1_opt'], infill_x1_opt['infill1_x1_opt'], and so on.
infill_f_opt = {}
infill_time = {}
infill_loss = {}
infill_layer = {}
infill_neuron = {}
infill_epoch = {}
infill_activation = {}

for i in range(45):
    infill_f_opt[f'infill{i}_f_opt'] = []
    infill_time[f'infill{i}_time'] = []
    infill_loss[f'infill{i}_loss'] = []
    infill_layer[f'infill{i}_layer'] = []
    infill_neuron[f'infill{i}_neuron'] = []
    infill_epoch[f'infill{i}_epoch'] = []
    infill_activation[f'infill{i}_activation'] = []

    for j in range(1,11):
        infill_f_opt[f'infill{i}_f_opt'].append(data[f'data{j}']['f_opt'][i][0])
        infill_time[f'infill{i}_time'].append(data[f'data{j}']['time'][i][0])
        infill_loss[f'infill{i}_loss'].append(data[f'data{j}']['loss'][i][0])
        infill_layer[f'infill{i}_layer'].append(data[f'data{j}']['layer'][i][0])
        infill_neuron[f'infill{i}_neuron'].append(data[f'data{j}']['neuron'][i][0])
        infill_epoch[f'infill{i}_epoch'].append(data[f'data{j}']['epoch'][i][0])
        infill_activation[f'infill{i}_activation'].append(data[f'data{j}']['activation'][i][0])

f_opts = []
times = []
losses = []
layers = []
neurons = []
epochs = []
activations = []

for i in range(45):
    f_opts.append(np.array(infill_f_opt[f'infill{i}_f_opt']).reshape(-1,1))
    times.append(np.array(infill_time[f'infill{i}_time']).reshape(-1,1))
    losses.append(np.array(infill_loss[f'infill{i}_loss']).reshape(-1,1))
    layers.append(np.array(infill_layer[f'infill{i}_layer']).reshape(-1,1))
    epochs.append(np.array(infill_epoch[f'infill{i}_epoch']).reshape(-1,1))
    neurons.append(np.array(infill_neuron[f'infill{i}_neuron']).reshape(-1,1))
    activations.append(np.array(infill_activation[f'infill{i}_activation']).reshape(-1,1))

f_opts = np.concatenate(f_opts, axis=1)
times = np.concatenate(times, axis=1)
losses = np.concatenate(losses, axis=1)
layers = np.concatenate(layers, axis=1)
neurons = np.concatenate(neurons, axis=1)
epochs = np.concatenate(epochs, axis=1)
activations = np.concatenate(activations, axis=1)

# Compute means and standard deviations for each number of samples
mean_f_opts_1itr = np.mean(f_opts, axis=0)
std_f_opts_1itr = np.std(f_opts, axis=0)
mean_times_1itr = np.mean(times, axis=0)
std_times_1itr= np.std(times, axis=0)
mean_losses_1itr = np.mean(losses, axis=0)
std_losses_1itr = np.std(losses, axis=0)
mean_layers_1itr = np.mean(layers, axis=0)
std_layers_1itr = np.std(layers, axis=0)
mean_neurons_1itr = np.mean(neurons, axis=0)
std_neurons_1itr = np.std(neurons, axis=0)
mean_epochs_1itr = np.mean(epochs, axis=0)
std_epochs_1itr = np.std(epochs, axis=0)
mean_activations_1itr = np.mean(activations, axis=0)
std_activations_1itr = np.std(activations, axis=0)

'''
Load data for BOHB-5itr
'''
data = {}  # Initialize an empty dictionary

# Loop over the number of files
for i in range(1, 11):
    # Formulate the file path
    file_path = f'output_3DHM_BOHB_5itr/{i}/result.mat'
    
    # Load the data and store it in the dictionary
    mat_data = loadmat(file_path)
    
    # Store the components in a dictionary
    data[f'data{i}'] = {
                        'f_opt': mat_data['f_data_min'].reshape(-1,1), 'time': mat_data['time'].reshape(-1,1), 'loss': mat_data['loss'],
                        'layer': mat_data['layer'], 'neuron': mat_data['neuron'], 'epoch': mat_data['epoch'], 'activation': mat_data['activation']
                        }

# Access the list by infill_x1_opt['infill0_x1_opt'], infill_x1_opt['infill1_x1_opt'], and so on.
infill_f_opt = {}
infill_time = {}
infill_loss = {}
infill_layer = {}
infill_neuron = {}
infill_epoch = {}
infill_activation = {}

for i in range(45):
    infill_f_opt[f'infill{i}_f_opt'] = []
    infill_time[f'infill{i}_time'] = []
    infill_loss[f'infill{i}_loss'] = []
    infill_layer[f'infill{i}_layer'] = []
    infill_neuron[f'infill{i}_neuron'] = []
    infill_epoch[f'infill{i}_epoch'] = []
    infill_activation[f'infill{i}_activation'] = []

    for j in range(1,11):
        infill_f_opt[f'infill{i}_f_opt'].append(data[f'data{j}']['f_opt'][i][0])
        infill_time[f'infill{i}_time'].append(data[f'data{j}']['time'][i][0])
        infill_loss[f'infill{i}_loss'].append(data[f'data{j}']['loss'][i][0])
        infill_layer[f'infill{i}_layer'].append(data[f'data{j}']['layer'][i][0])
        infill_neuron[f'infill{i}_neuron'].append(data[f'data{j}']['neuron'][i][0])
        infill_epoch[f'infill{i}_epoch'].append(data[f'data{j}']['epoch'][i][0])
        infill_activation[f'infill{i}_activation'].append(data[f'data{j}']['activation'][i][0])

f_opts = []
times = []
losses = []
layers = []
neurons = []
epochs = []
activations = []

for i in range(45):
    f_opts.append(np.array(infill_f_opt[f'infill{i}_f_opt']).reshape(-1,1))
    times.append(np.array(infill_time[f'infill{i}_time']).reshape(-1,1))
    losses.append(np.array(infill_loss[f'infill{i}_loss']).reshape(-1,1))
    layers.append(np.array(infill_layer[f'infill{i}_layer']).reshape(-1,1))
    epochs.append(np.array(infill_epoch[f'infill{i}_epoch']).reshape(-1,1))
    neurons.append(np.array(infill_neuron[f'infill{i}_neuron']).reshape(-1,1))
    activations.append(np.array(infill_activation[f'infill{i}_activation']).reshape(-1,1))

f_opts = np.concatenate(f_opts, axis=1)
times = np.concatenate(times, axis=1)
losses = np.concatenate(losses, axis=1)
layers = np.concatenate(layers, axis=1)
neurons = np.concatenate(neurons, axis=1)
epochs = np.concatenate(epochs, axis=1)
activations = np.concatenate(activations, axis=1)

def update_zeros(data):
    last_value = data[0]
    for i in range(1, len(data)):
        if data[i] == 0:
            data[i] = last_value
        else:
            last_value = data[i]
    return data

# Update zeros for each data set
f_opts = np.array([update_zeros(data) for data in f_opts])
times = np.array([update_zeros(data) for data in times])
losses = np.array([update_zeros(data) for data in losses])
layers = np.array([update_zeros(data) for data in layers])
neurons = np.array([update_zeros(data) for data in neurons])
epochs = np.array([update_zeros(data) for data in epochs])
activations = np.array([update_zeros(data) for data in activations])

# Compute means and standard deviations for each number of samples
mean_f_opts_5itr = np.mean(f_opts, axis=0)
std_f_opts_5itr = np.std(f_opts, axis=0)
mean_times_5itr = np.mean(times, axis=0)
std_times_5itr= np.std(times, axis=0)
mean_losses_5itr = np.mean(losses, axis=0)
std_losses_5itr = np.std(losses, axis=0)
mean_layers_5itr = np.mean(layers, axis=0)
std_layers_5itr = np.std(layers, axis=0)
mean_neurons_5itr = np.mean(neurons, axis=0)
std_neurons_5itr = np.std(neurons, axis=0)
mean_epochs_5itr = np.mean(epochs, axis=0)
std_epochs_5itr = np.std(epochs, axis=0)
mean_activations_5itr = np.mean(activations, axis=0)
std_activations_5itr = np.std(activations, axis=0)

'''
Load data for BOHB-static
'''
data = {}  # Initialize an empty dictionary

# Loop over the number of files
for i in range(1, 11):
    # Formulate the file path
    file_path = f'output_3DHM_BOHB_static/{i}/result.mat'
    
    # Load the data and store it in the dictionary
    mat_data = loadmat(file_path)
    
    # Store the components in a dictionary
    data[f'data{i}'] = {
                        'f_opt': mat_data['f_data_min'].reshape(-1,1), 'time': mat_data['time'].reshape(-1,1), 'loss': mat_data['loss'],
                        'layer': mat_data['layer'], 'neuron': mat_data['neuron'], 'epoch': mat_data['epoch'], 'activation': mat_data['activation']
                        }

# Access the list by infill_x1_opt['infill0_x1_opt'], infill_x1_opt['infill1_x1_opt'], and so on.
infill_f_opt = {}
infill_time = {}
infill_loss = {}
infill_layer = {}
infill_neuron = {}
infill_epoch = {}
infill_activation = {}

for i in range(45):
    infill_f_opt[f'infill{i}_f_opt'] = []
    infill_time[f'infill{i}_time'] = []
    infill_loss[f'infill{i}_loss'] = []
    infill_layer[f'infill{i}_layer'] = []
    infill_neuron[f'infill{i}_neuron'] = []
    infill_epoch[f'infill{i}_epoch'] = []
    infill_activation[f'infill{i}_activation'] = []

    for j in range(1,11):
        infill_f_opt[f'infill{i}_f_opt'].append(data[f'data{j}']['f_opt'][i][0])
        infill_time[f'infill{i}_time'].append(data[f'data{j}']['time'][i][0])
        infill_loss[f'infill{i}_loss'].append(data[f'data{j}']['loss'][i][0])
        infill_layer[f'infill{i}_layer'].append(data[f'data{j}']['layer'][i][0])
        infill_neuron[f'infill{i}_neuron'].append(data[f'data{j}']['neuron'][i][0])
        infill_epoch[f'infill{i}_epoch'].append(data[f'data{j}']['epoch'][i][0])
        infill_activation[f'infill{i}_activation'].append(data[f'data{j}']['activation'][i][0])

f_opts = []
times = []
losses = []
layers = []
neurons = []
epochs = []
activations = []

for i in range(45):
    f_opts.append(np.array(infill_f_opt[f'infill{i}_f_opt']).reshape(-1,1))
    times.append(np.array(infill_time[f'infill{i}_time']).reshape(-1,1))
    losses.append(np.array(infill_loss[f'infill{i}_loss']).reshape(-1,1))
    layers.append(np.array(infill_layer[f'infill{i}_layer']).reshape(-1,1))
    epochs.append(np.array(infill_epoch[f'infill{i}_epoch']).reshape(-1,1))
    neurons.append(np.array(infill_neuron[f'infill{i}_neuron']).reshape(-1,1))
    activations.append(np.array(infill_activation[f'infill{i}_activation']).reshape(-1,1))

f_opts = np.concatenate(f_opts, axis=1)
times = np.concatenate(times, axis=1)
losses = np.concatenate(losses, axis=1)
layers = np.concatenate(layers, axis=1)
neurons = np.concatenate(neurons, axis=1)
epochs = np.concatenate(epochs, axis=1)
activations = np.concatenate(activations, axis=1)

# Update zeros for each data set
f_opts = np.array([update_zeros(data) for data in f_opts])
times = np.array([update_zeros(data) for data in times])
losses = np.array([update_zeros(data) for data in losses])
layers = np.array([update_zeros(data) for data in layers])
neurons = np.array([update_zeros(data) for data in neurons])
epochs = np.array([update_zeros(data) for data in epochs])
activations = np.array([update_zeros(data) for data in activations])

# Compute means and standard deviations for each number of samples
mean_f_opts_static = np.mean(f_opts, axis=0)
std_f_opts_static = np.std(f_opts, axis=0)
mean_times_static = np.mean(times, axis=0)
std_times_static= np.std(times, axis=0)
mean_losses_static = np.mean(losses, axis=0)
std_losses_static = np.std(losses, axis=0)
mean_layers_static = np.mean(layers, axis=0)
std_layers_static = np.std(layers, axis=0)
mean_neurons_static = np.mean(neurons, axis=0)
std_neurons_static = np.std(neurons, axis=0)
mean_epochs_static = np.mean(epochs, axis=0)
std_epochs_static = np.std(epochs, axis=0)
mean_activations_static = np.mean(activations, axis=0)
std_activations_static = np.std(activations, axis=0)


'''
'''
'''
Plot
'''
'''
'''
x = np.arange(10, 55)  
alpha = 0.15
alpha2 = 0.1
linewidth = 2
tableau_palette = sns.color_palette("tab10")
blue = tableau_palette[0]  
orange = tableau_palette[3] # this is red
green = tableau_palette[2] 
red = tableau_palette[1]   # this is orange

fontsize=20
plot_y = 4
'''
Plot for optimum vs iteration
'''
fig, ax = plt.subplots(figsize=(12, plot_y))
plt.plot(x, mean_f_opts_krg, color=orange, linewidth = linewidth)
plt.scatter(x, mean_f_opts_krg, color=orange, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_f_opts_krg, std_f_opts_krg), np.add(mean_f_opts_krg, std_f_opts_krg), color=orange, alpha=alpha)

plt.plot(x, mean_f_opts_1itr, color=red, linewidth = linewidth)
plt.scatter(x, mean_f_opts_1itr, color=red, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_f_opts_1itr, std_f_opts_1itr), np.add(mean_f_opts_1itr, std_f_opts_1itr), color=red, alpha=alpha)

plt.plot(x, mean_f_opts_5itr, color=blue, linewidth = linewidth)
plt.scatter(x, mean_f_opts_5itr, color=blue, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_f_opts_5itr, std_f_opts_5itr), np.add(mean_f_opts_5itr, std_f_opts_5itr), color=blue, alpha=alpha)

plt.plot(x, mean_f_opts_static, color=green, linewidth = linewidth)
plt.scatter(x, mean_f_opts_static, color=green, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_f_opts_static, std_f_opts_static), np.add(mean_f_opts_static, std_f_opts_static), color=green, alpha=alpha2)

plt.axvspan(0, 10, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
plt.plot([10, 55], [-3.86, -3.86], color='black', linestyle='--')  # Horizontal line from x=6 to x=25 at y=0
plt.text(5.5, -3.51, 'DoE', fontsize=fontsize, ha='center')  # Insert text

# ax.set_ylim([-4, -3])
ax.set_yticks([-4, -3.75, -3.5, -3.25, -3])
ax.set_xticks(range(10,60,5)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of samples', fontsize=fontsize)
ax.set_ylabel(r'$\hat{y}(\mathbf{x}_d)$', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)
plt.xlim(1, 54)  # Set x limit from 1 to 30, adding extra space

plt.legend(handles=[mlines.Line2D([], [], color=orange, label='EGO'),
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')], loc='upper right', fontsize=fontsize)
plt.tight_layout()
plt.savefig('optimumVSitr.png', format='png', dpi=300)
plt.show()

from matplotlib.ticker import MultipleLocator

'''
Plot for loss vs iteration
'''
x = np.arange(1, 46)  

fig, ax = plt.subplots(figsize=(12, plot_y))

plt.plot(x, mean_losses_1itr, color=red, linewidth = linewidth)
plt.scatter(x, mean_losses_1itr, color=red, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_losses_1itr, std_losses_1itr), np.add(mean_losses_1itr, std_losses_1itr), color=red, alpha=alpha)

plt.plot(x, mean_losses_5itr, color=blue, linewidth = linewidth)
plt.scatter(x, mean_losses_5itr, color=blue, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_losses_5itr, std_losses_5itr), np.add(mean_losses_5itr, std_losses_5itr), color=blue, alpha=alpha)

plt.plot(x, mean_losses_static, color=green, linewidth = linewidth)
plt.scatter(x, mean_losses_static, color=green, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_losses_static, std_losses_static), np.add(mean_losses_static, std_losses_static), color=green, alpha=alpha2)

# plt.axvspan(0, 10, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
# plt.text(5.5, 0.68, 'DoE', fontsize=fontsize, ha='center')  # Insert text

ax.set_ylim([0, 1.4])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xticks(range(1,45,5)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of infills', fontsize=fontsize)
ax.set_ylabel('Loss', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)

plt.xlim(1, 45)  # Set x limit from 1 to 30, adding extra space

plt.legend(handles=[
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')], loc='upper right', fontsize=fontsize)
plt.tight_layout()
plt.savefig('lossVSitr.png', format='png', dpi=300)
plt.show()


'''
Plot for time vs iteration

fig, ax = plt.subplots(figsize=(12, plot_y))
plt.plot(x, mean_times_krg, color=orange, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_times_krg, std_times_krg), np.add(mean_times_krg, std_times_krg), color=orange, alpha=alpha)

plt.plot(x, mean_times_1itr, color=red, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_times_1itr, std_times_1itr), np.add(mean_times_1itr, std_times_1itr), color=red, alpha=alpha)

plt.plot(x, mean_times_5itr, color=blue, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_times_5itr, std_times_5itr), np.add(mean_times_5itr, std_times_5itr), color=blue, alpha=alpha)

plt.plot(x, mean_times_static, color=green, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_times_static, std_times_static), np.add(mean_times_static, std_times_static), color=green, alpha=alpha2)

plt.axvspan(0, 10, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
plt.text(5.5, 385, 'DoE', fontsize=fontsize, ha='center')  # Insert text

# ax.set_ylim([-4, -3])
ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
ax.set_xticks(range(10,55,2)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of samples', fontsize=fontsize)
ax.set_ylabel('Time (s)', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)
plt.xlim(1, 54)  # Set x limit from 1 to 30, adding extra space

plt.legend(handles=[mlines.Line2D([], [], color=orange, label='Kriging'),
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')])
plt.tight_layout()
plt.savefig('timeVSitr.png', format='png', dpi=300)
plt.show()
'''





'''
Plot for optimum vs time
'''
mean_times_krg = np.cumsum(mean_times_krg)
mean_times_static = np.cumsum(mean_times_static)
mean_times_1itr = np.cumsum(mean_times_1itr)
mean_times_5itr = np.cumsum(mean_times_5itr)

mean_times_krg = np.insert(mean_times_krg, 0, 0)
mean_times_static = np.insert(mean_times_static, 0, 0)
mean_times_1itr = np.insert(mean_times_1itr, 0, 0)
mean_times_5itr = np.insert(mean_times_5itr, 0, 0)

mean_f_opts_krg = np.insert(mean_f_opts_krg, 0, mean_f_opts_krg[0])
mean_f_opts_1itr = np.insert(mean_f_opts_1itr, 0, mean_f_opts_1itr[0])
mean_f_opts_5itr = np.insert(mean_f_opts_5itr, 0, mean_f_opts_5itr[0])
mean_f_opts_static = np.insert(mean_f_opts_static, 0, mean_f_opts_static[0])

std_f_opts_krg = np.insert(std_f_opts_krg, 0, std_f_opts_krg[0])
std_f_opts_1itr = np.insert(std_f_opts_1itr, 0, std_f_opts_5itr[0])
std_f_opts_5itr = np.insert(std_f_opts_5itr, 0, std_f_opts_5itr[0])
std_f_opts_static = np.insert(std_f_opts_static, 0, std_f_opts_static[0])

fig, ax = plt.subplots(figsize=(12, plot_y))

plt.plot(mean_times_krg, mean_f_opts_krg, color=orange, linewidth = linewidth)
plt.scatter(mean_times_krg, mean_f_opts_krg, color=orange, linewidth = linewidth)
plt.fill_between(mean_times_krg, np.subtract(mean_f_opts_krg, std_f_opts_krg), np.add(mean_f_opts_krg, std_f_opts_krg), color=orange, alpha=alpha)

plt.plot(mean_times_1itr, mean_f_opts_1itr, color=red, linewidth = linewidth)
plt.scatter(mean_times_1itr, mean_f_opts_1itr, color=red, linewidth = linewidth)
plt.fill_between(mean_times_1itr, np.subtract(mean_f_opts_1itr, std_f_opts_1itr), np.add(mean_f_opts_1itr, std_f_opts_1itr), color=red, alpha=alpha)

plt.plot(mean_times_5itr, mean_f_opts_5itr, color=blue, linewidth = linewidth)
plt.scatter(mean_times_5itr, mean_f_opts_5itr, color=blue, linewidth = linewidth)
plt.fill_between(mean_times_5itr, np.subtract(mean_f_opts_5itr, std_f_opts_5itr), np.add(mean_f_opts_5itr, std_f_opts_5itr), color=blue, alpha=alpha)

plt.plot(mean_times_static, mean_f_opts_static, color=green, linewidth = linewidth)
plt.scatter(mean_times_static, mean_f_opts_static, color=green, linewidth = linewidth)
plt.fill_between(mean_times_static, np.subtract(mean_f_opts_static, std_f_opts_static), np.add(mean_f_opts_static, std_f_opts_static), color=green, alpha=alpha2)

plt.plot([0, 12000], [-3.86, -3.86], color='black', linestyle='--')  # Horizontal line from x=6 to x=25 at y=0

ax.set_ylim([-3.9, -3.2])
ax.set_yticks([-3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2])
ax.set_xticks(range(0,12001,2000)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xscale("log")

plt.xlabel('Time (s)', fontsize=fontsize)
ax.set_ylabel(r'$\hat{y}(\mathbf{x}_d)$', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)

plt.legend(handles=[
                    mlines.Line2D([], [], color=orange, label='EGO'),
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')], loc='best', fontsize=fontsize)

# mlines.Line2D([], [], color='black', linestyle='--', label='Minimum')
plt.tight_layout()
plt.savefig('optimumVStime.png', format='png', dpi=300)
plt.show()

'''
Plot for loss vs time
'''















'''
# Create figures for x1_opts and x2_opts and x3_opts


fig, ax = plt.subplots(figsize=(12, 5))
plt.plot(x, mean_x1_opts, color=palette[3])
plt.fill_between(x, np.subtract(mean_x1_opts, std_x1_opts), np.add(mean_x1_opts, std_x1_opts), color=palette[3], alpha=alpha)
plt.plot(x, mean_x2_opts, color=palette[7])
plt.fill_between(x, np.subtract(mean_x2_opts, std_x2_opts), np.add(mean_x2_opts, std_x2_opts), color=palette[7], alpha=alpha)
plt.plot(x, mean_x3_opts, color=palette[9])
plt.fill_between(x, np.subtract(mean_x3_opts, std_x3_opts), np.add(mean_x3_opts, std_x3_opts), color=palette[9], alpha=alpha)

plt.axvspan(0, 15, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
plt.plot([15, 45], [0.115, 0.115], color='gray', linestyle='--')  # Horizontal line from x=6 to x=25 at y=0
plt.plot([15, 45], [0.555, 0.555], color='gray', linestyle='--')  # Horizontal line from x=6 to x=25 at y=0
plt.plot([15, 45], [0.853, 0.853], color='gray', linestyle='--')  # Horizontal line from x=6 to x=25 at y=0
plt.text(7.5, 0.75/2, 'DoE', fontsize=fontsize, ha='center')  # Insert text

ax.set_ylim([-0.25, 1])
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks(range(15,45,3)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of Samples', fontsize=fontsize)
plt.ylabel('$x_i$', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)
plt.xlim(1, 44)  # Set x limit from 1 to 30, adding extra space

var_patch1 = Patch(color=palette[3], alpha=alpha, label='Variance of $x_1$')
var_patch2 = Patch(color=palette[7], alpha=alpha, label='Variance of $x_2$')
var_patch3 = Patch(color=palette[5], alpha=alpha, label='Variance of $x_3$')

plt.legend(handles=[mlines.Line2D([], [], color=palette[3], label='Mean of $x_1$'), 
                    mlines.Line2D([], [], color=palette[7], label='Mean of $x_2$'), 
                    mlines.Line2D([], [], color=palette[9], label='Mean of $x_3$'), 
                    mlines.Line2D([], [], color='gray', linestyle='--', label='True Optimums')], loc='upper left', fontsize=fontsize)
plt.tight_layout()
plt.savefig('EGO_KRG_ST_x.png', format='png', dpi=300)
plt.show()
'''
