import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns

'''
Data from Ax
'''
data1 = loadmat('output_Branin_ax/10/result.mat')
data2 = loadmat('output_Branin_ax/20/result.mat')
data3 = loadmat('output_Branin_ax/30/result.mat')
data4 = loadmat('output_Branin_ax/40/result.mat')
data5 = loadmat('output_Branin_ax/50/result.mat')

activation1 = data1['activation'].reshape(-1)
activation2 = data2['activation'].reshape(-1)
activation3 = data3['activation'].reshape(-1)
activation4 = data4['activation'].reshape(-1)
activation5 = data5['activation'].reshape(-1)

epoch1 = data1['epoch'].reshape(-1)
epoch2 = data2['epoch'].reshape(-1)
epoch3 = data3['epoch'].reshape(-1)
epoch4 = data4['epoch'].reshape(-1)
epoch5 = data5['epoch'].reshape(-1)

layer1 = data1['layer'].reshape(-1)
layer2 = data2['layer'].reshape(-1)
layer3 = data3['layer'].reshape(-1)
layer4 = data4['layer'].reshape(-1)
layer5 = data5['layer'].reshape(-1)

neuron1 = data1['neuron'].reshape(-1)
neuron2 = data2['neuron'].reshape(-1)
neuron3 = data3['neuron'].reshape(-1)
neuron4 = data4['neuron'].reshape(-1)
neuron5 = data5['neuron'].reshape(-1)

layer_ax = [np.array(layer1), np.array(layer2), np.array(layer3), 
             np.array(layer4), np.array(layer5)]
neuron_ax = [np.array(neuron1), np.array(neuron2), np.array(neuron3), 
             np.array(neuron4), np.array(neuron5)]
epoch_ax = [np.array(epoch1), np.array(epoch2), np.array(epoch3), 
             np.array(epoch4), np.array(epoch5)]
activation_ax = [np.array(activation1), np.array(activation2), np.array(activation3), 
             np.array(activation4), np.array(activation5)]

'''
Data from GS
'''
data1 = loadmat('output_Branin_gs/10/result.mat')
data2 = loadmat('output_Branin_gs/20/result.mat')
data3 = loadmat('output_Branin_gs/30/result.mat')
data4 = loadmat('output_Branin_gs/40/result.mat')
data5 = loadmat('output_Branin_gs/50/result.mat')

activation1 = data1['activation'].reshape(-1)
activation2 = data2['activation'].reshape(-1)
activation3 = data3['activation'].reshape(-1)
activation4 = data4['activation'].reshape(-1)
activation5 = data5['activation'].reshape(-1)

epoch1 = data1['epoch'].reshape(-1)
epoch2 = data2['epoch'].reshape(-1)
epoch3 = data3['epoch'].reshape(-1)
epoch4 = data4['epoch'].reshape(-1)
epoch5 = data5['epoch'].reshape(-1)

layer1 = data1['layer'].reshape(-1)
layer2 = data2['layer'].reshape(-1)
layer3 = data3['layer'].reshape(-1)
layer4 = data4['layer'].reshape(-1)
layer5 = data5['layer'].reshape(-1)

neuron1 = data1['neuron'].reshape(-1)
neuron2 = data2['neuron'].reshape(-1)
neuron3 = data3['neuron'].reshape(-1)
neuron4 = data4['neuron'].reshape(-1)
neuron5 = data5['neuron'].reshape(-1)

layer_gs = [np.array(layer1), np.array(layer2), np.array(layer3), 
             np.array(layer4), np.array(layer5)]
neuron_gs = [np.array(neuron1), np.array(neuron2), np.array(neuron3), 
             np.array(neuron4), np.array(neuron5)]
epoch_gs = [np.array(epoch1), np.array(epoch2), np.array(epoch3), 
             np.array(epoch4), np.array(epoch5)]
activation_gs = [np.array(activation1), np.array(activation2), np.array(activation3), 
             np.array(activation4), np.array(activation5)]

'''
Data from RS
'''
data1 = loadmat('output_Branin_rs/10/result.mat')
data2 = loadmat('output_Branin_rs/20/result.mat')
data3 = loadmat('output_Branin_rs/30/result.mat')
data4 = loadmat('output_Branin_rs/40/result.mat')
data5 = loadmat('output_Branin_rs/50/result.mat')

activation1 = data1['activation'].reshape(-1)
activation2 = data2['activation'].reshape(-1)
activation3 = data3['activation'].reshape(-1)
activation4 = data4['activation'].reshape(-1)
activation5 = data5['activation'].reshape(-1)

epoch1 = data1['epoch'].reshape(-1)
epoch2 = data2['epoch'].reshape(-1)
epoch3 = data3['epoch'].reshape(-1)
epoch4 = data4['epoch'].reshape(-1)
epoch5 = data5['epoch'].reshape(-1)

layer1 = data1['layer'].reshape(-1)
layer2 = data2['layer'].reshape(-1)
layer3 = data3['layer'].reshape(-1)
layer4 = data4['layer'].reshape(-1)
layer5 = data5['layer'].reshape(-1)

neuron1 = data1['neuron'].reshape(-1)
neuron2 = data2['neuron'].reshape(-1)
neuron3 = data3['neuron'].reshape(-1)
neuron4 = data4['neuron'].reshape(-1)
neuron5 = data5['neuron'].reshape(-1)

layer_rs = [np.array(layer1), np.array(layer2), np.array(layer3), 
             np.array(layer4), np.array(layer5)]
neuron_rs = [np.array(neuron1), np.array(neuron2), np.array(neuron3), 
             np.array(neuron4), np.array(neuron5)]
epoch_rs = [np.array(epoch1), np.array(epoch2), np.array(epoch3), 
             np.array(epoch4), np.array(epoch5)]
activation_rs = [np.array(activation1), np.array(activation2), np.array(activation3), 
             np.array(activation4), np.array(activation5)]

'''
Data from HB
'''
data1 = loadmat('output_Branin_hb/10/result.mat')
data2 = loadmat('output_Branin_hb/20/result.mat')
data3 = loadmat('output_Branin_hb/30/result.mat')
data4 = loadmat('output_Branin_hb/40/result.mat')
data5 = loadmat('output_Branin_hb/50/result.mat')

activation1 = data1['activation'].reshape(-1)
activation2 = data2['activation'].reshape(-1)
activation3 = data3['activation'].reshape(-1)
activation4 = data4['activation'].reshape(-1)
activation5 = data5['activation'].reshape(-1)

epoch1 = data1['epoch'].reshape(-1)
epoch2 = data2['epoch'].reshape(-1)
epoch3 = data3['epoch'].reshape(-1)
epoch4 = data4['epoch'].reshape(-1)
epoch5 = data5['epoch'].reshape(-1)

layer1 = data1['layer'].reshape(-1)
layer2 = data2['layer'].reshape(-1)
layer3 = data3['layer'].reshape(-1)
layer4 = data4['layer'].reshape(-1)
layer5 = data5['layer'].reshape(-1)

neuron1 = data1['neuron'].reshape(-1)
neuron2 = data2['neuron'].reshape(-1)
neuron3 = data3['neuron'].reshape(-1)
neuron4 = data4['neuron'].reshape(-1)
neuron5 = data5['neuron'].reshape(-1)

layer_hb = [np.array(layer1), np.array(layer2), np.array(layer3), 
             np.array(layer4), np.array(layer5)]
neuron_hb = [np.array(neuron1), np.array(neuron2), np.array(neuron3), 
             np.array(neuron4), np.array(neuron5)]
epoch_hb = [np.array(epoch1), np.array(epoch2), np.array(epoch3), 
             np.array(epoch4), np.array(epoch5)]
activation_hb = [np.array(activation1), np.array(activation2), np.array(activation3), 
             np.array(activation4), np.array(activation5)]

'''
Data from BOHB
'''
data1 = loadmat('output_Branin_bohb/10/result.mat')
data2 = loadmat('output_Branin_bohb/20/result.mat')
data3 = loadmat('output_Branin_bohb/30/result.mat')
data4 = loadmat('output_Branin_bohb/40/result.mat')
data5 = loadmat('output_Branin_bohb/50/result.mat')

activation1 = data1['activation'].reshape(-1)
activation2 = data2['activation'].reshape(-1)
activation3 = data3['activation'].reshape(-1)
activation4 = data4['activation'].reshape(-1)
activation5 = data5['activation'].reshape(-1)

epoch1 = data1['epoch'].reshape(-1)
epoch2 = data2['epoch'].reshape(-1)
epoch3 = data3['epoch'].reshape(-1)
epoch4 = data4['epoch'].reshape(-1)
epoch5 = data5['epoch'].reshape(-1)

layer1 = data1['layer'].reshape(-1)
layer2 = data2['layer'].reshape(-1)
layer3 = data3['layer'].reshape(-1)
layer4 = data4['layer'].reshape(-1)
layer5 = data5['layer'].reshape(-1)

neuron1 = data1['neuron'].reshape(-1)
neuron2 = data2['neuron'].reshape(-1)
neuron3 = data3['neuron'].reshape(-1)
neuron4 = data4['neuron'].reshape(-1)
neuron5 = data5['neuron'].reshape(-1)

layer_bohb = [np.array(layer1), np.array(layer2), np.array(layer3), 
             np.array(layer4), np.array(layer5)]
neuron_bohb = [np.array(neuron1), np.array(neuron2), np.array(neuron3), 
             np.array(neuron4), np.array(neuron5)]
epoch_bohb = [np.array(epoch1), np.array(epoch2), np.array(epoch3), 
             np.array(epoch4), np.array(epoch5)]
activation_bohb = [np.array(activation1), np.array(activation2), np.array(activation3), 
             np.array(activation4), np.array(activation5)]



'''
Plot
'''
fontsize = 27

data1 = [np.random.rand(15) * 10 for i in range(5)]

labels = ['10', '20', '30', '40', '50']

tableau_palette = sns.color_palette("tab10")
blue = tableau_palette[0]  
orange = tableau_palette[1]
green = tableau_palette[2] 
red = tableau_palette[3]   
violet = tableau_palette[4]


# Define a function to plot the boxplots at a specific offset
def plot_box(data, offset, color, label):
    position = np.arange(len(data)) + offset
    bp = ax.boxplot(data, positions=position, patch_artist=True, widths=0.1, medianprops={'linewidth': 2})
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set_color('black')
    return bp

def plot_box_trans(data, offset, color, label, transparency=1):
    position = np.arange(len(data)) + offset
    bp = ax.boxplot(data, positions=position, patch_artist=True, widths=0.10, 
                    medianprops={'linewidth': 2, 'alpha': transparency},
                    whiskerprops={'alpha': transparency},
                    capprops={'alpha': transparency},
                    boxprops={'facecolor': color, 'alpha': transparency},
                    flierprops={'alpha': transparency, 'markeredgecolor': color})
    
    for median in bp['medians']:
        median.set_color('black')
    return bp
'''
Plot for layer
'''
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each data set with a different offset and color
plot_box_trans(layer_gs, 0.00, 'violet', 'Data1')
plot_box_trans(layer_rs, 0.1, 'orange', 'Data2')
plot_box_trans(layer_ax, 0.2, 'green', 'Data5')
plot_box_trans(layer_hb, 0.3, 'red', 'Data3')
plot_box_trans(layer_bohb, 0.4, 'blue', 'Data4')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([1,4])
ax.set_yticks([1,2,3,4])
ax.set_ylabel('Number of layers', fontsize=fontsize)
ax.set_xlabel('Number of samples', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [
                   Patch(facecolor='violet', label='Grid Search'),
                   Patch(facecolor='orange', label='Random Search'),
                   Patch(facecolor='green', label='BO'),
                   Patch(facecolor='red', label='HB'),
                   Patch(facecolor='blue', label='BOHB'),
                   mlines.Line2D([], [], color='black', linestyle='--',  markersize=15, label='Minimum')
]

# ax.legend(handles=legend_elements, loc='lower right', fontsize=fontsize, ncol=2)
fig.legend(handles=legend_elements, loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, -0.24))
plt.subplots_adjust(bottom=0.15)  # You may need to adjust this value

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('HP_box_layer.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

'''
Plot for neuron
'''
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each data set with a different offset and color
plot_box_trans(neuron_gs, 0.00, 'violet', 'Data1')
plot_box_trans(neuron_rs, 0.1, 'orange', 'Data2')
plot_box_trans(neuron_ax, 0.2, 'green', 'Data5')
plot_box_trans(neuron_hb, 0.3, 'red', 'Data3')
plot_box_trans(neuron_bohb, 0.4, 'blue', 'Data4')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([4,16])
ax.set_yticks([4, 8, 12, 16])
ax.set_ylabel('Number of neurons', fontsize=fontsize)
ax.set_xlabel('Number of samples', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [
                   Patch(facecolor='violet', label='Grid Search'),
                   Patch(facecolor='orange', label='Random Search'),
                   Patch(facecolor='green', label='BO'),
                   Patch(facecolor='red', label='HB'),
                   Patch(facecolor='blue', label='BOHB'),
                   mlines.Line2D([], [], color='black', linestyle='--',  markersize=15, label='Minimum')
]

# ax.legend(handles=legend_elements, loc='upper right', fontsize=fontsize, ncol=2)
fig.legend(handles=legend_elements, loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, -0.24))
plt.subplots_adjust(bottom=0.15)  # You may need to adjust this value

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('HP_box_neuron.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

'''
Plot for epoch
'''
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each data set with a different offset and color
plot_box_trans(epoch_gs, 0.00, 'violet', 'Data1')
plot_box_trans(epoch_rs, 0.1, 'orange', 'Data2')
plot_box_trans(epoch_ax, 0.2, 'green', 'Data5')
plot_box_trans(epoch_hb, 0.3, 'red', 'Data3')
plot_box_trans(epoch_bohb, 0.4, 'blue', 'Data4')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([1000,10000])
ax.set_yticks([1000, 4000, 7000, 10000])
ax.set_ylabel('Number of epochs', fontsize=fontsize)
ax.set_xlabel('Number of samples', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [
                   Patch(facecolor='violet', label='Grid Search'),
                   Patch(facecolor='orange', label='Random Search'),
                   Patch(facecolor='green', label='BO'),
                   Patch(facecolor='red', label='HB'),
                   Patch(facecolor='blue', label='BOHB'),
                   mlines.Line2D([], [], color='black', linestyle='--',  markersize=15, label='Minimum')
]

# ax.legend(handles=legend_elements, loc='upper left', fontsize=fontsize, ncol=2)
fig.legend(handles=legend_elements, loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, -0.24))
plt.subplots_adjust(bottom=0.15)  # You may need to adjust this value

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('HP_box_epoch.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

'''
Plot for activation
'''
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each data set with a different offset and color
plot_box_trans(activation_gs, 0.00, 'violet', 'Data1')
plot_box_trans(activation_rs, 0.1, 'orange', 'Data2')
plot_box_trans(activation_ax, 0.2, 'green', 'Data5')
plot_box_trans(activation_hb, 0.3, 'red', 'Data3')
plot_box_trans(activation_bohb, 0.4, 'blue', 'Data4')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([1,4])
ax.set_yticks([1,2,3,4])
ax.set_yticklabels(['relu', 'elu', 'tanh', 'sigmoid'])  # Set custom labels

ax.set_ylabel('Activation function', fontsize=fontsize)
ax.set_xlabel('Number of samples', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [
                   Patch(facecolor='violet', label='Grid Search'),
                   Patch(facecolor='orange', label='Random Search'),
                   Patch(facecolor='green', label='BO'),
                   Patch(facecolor='red', label='HB'),
                   Patch(facecolor='blue', label='BOHB'),
                   mlines.Line2D([], [], color='black', linestyle='--',  markersize=15, label='Minimum')
]

# ax.legend(handles=legend_elements, loc='upper left', fontsize=fontsize, ncol=2)
fig.legend(handles=legend_elements, loc='lower center', fontsize=fontsize, ncol=3, bbox_to_anchor=(0.5, -0.24))
plt.subplots_adjust(bottom=0.15)  # You may need to adjust this value

# Show the plot
ax.grid(True)
# plt.tight_layout()
plt.savefig('HP_box_activation.png', format='png', dpi=300, bbox_inches='tight')
plt.show()






