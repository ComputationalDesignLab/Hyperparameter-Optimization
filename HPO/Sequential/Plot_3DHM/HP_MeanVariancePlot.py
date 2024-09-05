from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.lines as mlines


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
# x = np.arange(10, 55)  
x = np.arange(1, 46)  

alpha = 0.15
alpha2 = 0.1
linewidth = 2
tableau_palette = sns.color_palette("tab10")
# blue = tableau_palette[0]  
# orange = tableau_palette[1]
# green = tableau_palette[2] 
# red = tableau_palette[3]   
blue = tableau_palette[0]  
orange = tableau_palette[3]
green = tableau_palette[2] 
red = tableau_palette[1]  

fontsize = 27
legend_y = -0.39
figsize_x = 12
figsize_y = 7

'''
Plot for activation vs iteration
'''
fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

plt.plot(x, mean_activations_1itr, color=red, linewidth = linewidth)
plt.scatter(x, mean_activations_1itr, color=red, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_activations_1itr, std_activations_1itr), np.add(mean_activations_1itr, std_activations_1itr), color=red, alpha=alpha)

plt.plot(x, mean_activations_5itr, color=blue, linewidth = linewidth)
plt.scatter(x, mean_activations_5itr, color=blue, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_activations_5itr, std_activations_5itr), np.add(mean_activations_5itr, std_activations_5itr), color=blue, alpha=alpha)

plt.plot(x, mean_activations_static, color=green, linewidth = linewidth)
plt.scatter(x, mean_activations_static, color=green, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_activations_static, std_activations_static), np.add(mean_activations_static, std_activations_static), color=green, alpha=alpha2)

# plt.axvspan(0, 9.6, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
# plt.text(5.5, 2.35, 'DoE', fontsize=fontsize, ha='center')  # Insert text

ax.set_ylim([1, 4])
ax.set_yticks([1,2,3,4])
ax.set_yticklabels(['relu', 'elu', 'tanh', 'sigmoid'])  # Set custom labels

# ax.set_xticks(range(10,55,4)) 
ax.set_xticks(range(1,45,5)) 

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of infills', fontsize=fontsize)
ax.set_ylabel('Activation functions', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)
# plt.xlim(1, 54)  # Set x limit from 1 to 30, adding extra space
plt.xlim(1, 45)

plt.legend(handles=[
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')
                    ], loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, legend_y))

plt.tight_layout()
plt.savefig('activationVSitr.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

'''
Plot for epoch vs iteration
'''
fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

plt.plot(x, mean_epochs_1itr, color=red, linewidth = linewidth)
plt.scatter(x, mean_epochs_1itr, color=red, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_epochs_1itr, std_epochs_1itr), np.add(mean_epochs_1itr, std_epochs_1itr), color=red, alpha=alpha)

plt.plot(x, mean_epochs_5itr, color=blue, linewidth = linewidth)
plt.scatter(x, mean_epochs_5itr, color=blue, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_epochs_5itr, std_epochs_5itr), np.add(mean_epochs_5itr, std_epochs_5itr), color=blue, alpha=alpha)

plt.plot(x, mean_epochs_static, color=green, linewidth = linewidth)
plt.scatter(x, mean_epochs_static, color=green, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_epochs_static, std_epochs_static), np.add(mean_epochs_static, std_epochs_static), color=green, alpha=alpha2)

# plt.axvspan(0, 9.6, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
# plt.text(5.5, 4950, 'DoE', fontsize=fontsize, ha='center')  # Insert text

ax.set_ylim([1000,10000])
ax.set_yticks([1000, 4000, 7000, 10000])

# ax.set_xticks(range(10,55,4)) 
ax.set_xticks(range(1,45,5)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of infills', fontsize=fontsize)

ax.set_ylabel('Number of epochs', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)
# plt.xlim(1, 54)  # Set x limit from 1 to 30, adding extra space
plt.xlim(1, 45)

plt.legend(handles=[
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')
                    ], loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, legend_y))

plt.tight_layout()
plt.savefig('epochVSitr.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

'''
Plot for layer vs iteration
'''
fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

plt.plot(x, mean_layers_1itr, color=red, linewidth = linewidth)
plt.scatter(x, mean_layers_1itr, color=red, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_layers_1itr, std_layers_1itr), np.add(mean_layers_1itr, std_layers_1itr), color=red, alpha=alpha)

plt.plot(x, mean_layers_5itr, color=blue, linewidth = linewidth)
plt.scatter(x, mean_layers_5itr, color=blue, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_layers_5itr, std_layers_5itr), np.add(mean_layers_5itr, std_layers_5itr), color=blue, alpha=alpha)

plt.plot(x, mean_layers_static, color=green, linewidth = linewidth)
plt.scatter(x, mean_layers_static, color=green, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_layers_static, std_layers_static), np.add(mean_layers_static, std_layers_static), color=green, alpha=alpha2)

# plt.axvspan(0, 9.6, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
# plt.text(5.5, 2.35, 'DoE', fontsize=fontsize, ha='center')  # Insert text
ax.set_ylim([1,4])
ax.set_yticks([1,2,3,4])

# ax.set_xticks(range(10,55,4)) 
ax.set_xticks(range(1,45,5)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of infills', fontsize=fontsize)

ax.set_ylabel('Number of layers', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)
# plt.xlim(1, 54)  # Set x limit from 1 to 30, adding extra space
plt.xlim(1, 45)

plt.legend(handles=[
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')
                    ], loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, legend_y))

plt.tight_layout()
plt.savefig('layerVSitr.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

'''
Plot for neuron vs iteration
'''
fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

plt.plot(x, mean_neurons_1itr, color=red, linewidth = linewidth)
plt.scatter(x, mean_neurons_1itr, color=red, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_neurons_1itr, std_neurons_1itr), np.add(mean_neurons_1itr, std_neurons_1itr), color=red, alpha=alpha)

plt.plot(x, mean_neurons_5itr, color=blue, linewidth = linewidth)
plt.scatter(x, mean_neurons_5itr, color=blue, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_neurons_5itr, std_neurons_5itr), np.add(mean_neurons_5itr, std_neurons_5itr), color=blue, alpha=alpha)

plt.plot(x, mean_neurons_static, color=green, linewidth = linewidth)
plt.scatter(x, mean_neurons_static, color=green, linewidth = linewidth)
plt.fill_between(x, np.subtract(mean_neurons_static, std_neurons_static), np.add(mean_neurons_static, std_neurons_static), color=green, alpha=alpha2)

# plt.axvspan(0, 9.6, facecolor='gray', alpha=1)  # Adding a colored box from 0 to 5
# plt.text(5.5, 9.65, 'DoE', fontsize=fontsize, ha='center')  # Insert text

ax.set_ylim([4,16])
ax.set_yticks([4, 8, 12, 16])

# ax.set_xticks(range(10,55,4)) 
ax.set_xticks(range(1,45,5)) 
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.xlabel('Number of infills', fontsize=fontsize)

ax.set_ylabel('Number of neurons', fontsize=fontsize)
plt.margins(x=0)  # Remove x-axis margins
plt.grid(True)

# plt.xlim(1, 54)  # Set x limit from 1 to 30, adding extra space
plt.xlim(1, 45)

plt.legend(handles=[
                    mlines.Line2D([], [], color=red, label='BOHB-1itr'),
                    mlines.Line2D([], [], color=blue, label='BOHB-5itr'),
                    mlines.Line2D([], [], color=green, label='BOHB-static')
                    ], loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, legend_y))

plt.tight_layout()
plt.savefig('neuronVSitr.png', format='png', dpi=300, bbox_inches='tight')
plt.show()




        
        