import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns


'''
Data for x coordinates for plot 
I plan not to plot x for the paper 3

x_opt1 = data1['x']
x_opt2 = data2['x']
x_opt3 = data3['x']
x_opt4 = data4['x']
x_opt5 = data5['x']

x1_1 = x_opt1[:, 0]  
x2_1 = x_opt1[:, 1]  
x1_2 = x_opt2[:, 0]  
x2_2 = x_opt2[:, 1]  
x1_3 = x_opt3[:, 0]  
x2_3 = x_opt3[:, 1]  
x1_4 = x_opt4[:, 0]  
x2_4 = x_opt4[:, 1]  
x1_5 = x_opt5[:, 0]  
x2_5 = x_opt5[:, 1]  

x1_opt_ax = [np.array(x1_1), np.array(x1_2), np.array(x1_3), 
             np.array(x1_4), np.array(x1_5)]
x2_opt_ax = [np.array(x2_1), np.array(x2_2), np.array(x2_3), 
             np.array(x2_4), np.array(x2_5)]
'''
'''
Data from Ax
'''
data1 = loadmat('output_Branin_ax/10/result.mat')
data2 = loadmat('output_Branin_ax/20/result.mat')
data3 = loadmat('output_Branin_ax/30/result.mat')
data4 = loadmat('output_Branin_ax/40/result.mat')
data5 = loadmat('output_Branin_ax/50/result.mat')

f_opt1 = data1['fun'].reshape(-1)
f_opt2 = data2['fun'].reshape(-1)
f_opt3 = data3['fun'].reshape(-1)
f_opt4 = data4['fun'].reshape(-1)
f_opt5 = data5['fun'].reshape(-1)

loss1 = data1['loss'].reshape(-1)
loss2 = data2['loss'].reshape(-1)
loss3 = data3['loss'].reshape(-1)
loss4 = data4['loss'].reshape(-1)
loss5 = data5['loss'].reshape(-1)

time1 = data1['time'].reshape(-1)
time2 = data2['time'].reshape(-1)
time3 = data3['time'].reshape(-1)
time4 = data4['time'].reshape(-1)
time5 = data5['time'].reshape(-1)

training_nrmse1 = data1['training_nrmse'].reshape(-1)
training_nrmse2 = data2['training_nrmse'].reshape(-1)
training_nrmse3 = data3['training_nrmse'].reshape(-1)
training_nrmse4 = data4['training_nrmse'].reshape(-1)
training_nrmse5 = data5['training_nrmse'].reshape(-1)

f_opt_ax = [np.array(f_opt1), np.array(f_opt2), np.array(f_opt3), 
             np.array(f_opt4), np.array(f_opt5)]

loss_ax = [np.array(loss1), np.array(loss2), np.array(loss3), 
             np.array(loss4), np.array(loss5)]

time_ax = [np.array(time1), np.array(time2), np.array(time3), 
             np.array(time4), np.array(time5)]

training_nrmse_ax = [np.array(training_nrmse1), np.array(training_nrmse2), np.array(training_nrmse3), 
             np.array(training_nrmse4), np.array(training_nrmse5)]

'''
Data from GS
'''
data1 = loadmat('output_Branin_gs/10/result.mat')
data2 = loadmat('output_Branin_gs/20/result.mat')
data3 = loadmat('output_Branin_gs/30/result.mat')
data4 = loadmat('output_Branin_gs/40/result.mat')
data5 = loadmat('output_Branin_gs/50/result.mat')

f_opt1 = data1['fun'].reshape(-1)
f_opt2 = data2['fun'].reshape(-1)
f_opt3 = data3['fun'].reshape(-1)
f_opt4 = data4['fun'].reshape(-1)
f_opt5 = data5['fun'].reshape(-1)

loss1 = data1['loss'].reshape(-1)
loss2 = data2['loss'].reshape(-1)
loss3 = data3['loss'].reshape(-1)
loss4 = data4['loss'].reshape(-1)
loss5 = data5['loss'].reshape(-1)

time1 = data1['time'].reshape(-1)
time2 = data2['time'].reshape(-1)
time3 = data3['time'].reshape(-1)
time4 = data4['time'].reshape(-1)
time5 = data5['time'].reshape(-1)

training_nrmse1 = data1['training_nrmse'].reshape(-1)
training_nrmse2 = data2['training_nrmse'].reshape(-1)
training_nrmse3 = data3['training_nrmse'].reshape(-1)
training_nrmse4 = data4['training_nrmse'].reshape(-1)
training_nrmse5 = data5['training_nrmse'].reshape(-1)

f_opt_gs = [np.array(f_opt1), np.array(f_opt2), np.array(f_opt3), 
             np.array(f_opt4), np.array(f_opt5)]

loss_gs = [np.array(loss1), np.array(loss2), np.array(loss3), 
             np.array(loss4), np.array(loss5)]

time_gs = [np.array(time1), np.array(time2), np.array(time3), 
             np.array(time4), np.array(time5)]

training_nrmse_gs = [np.array(training_nrmse1), np.array(training_nrmse2), np.array(training_nrmse3), 
             np.array(training_nrmse4), np.array(training_nrmse5)]

'''
Data from RS
'''
data1 = loadmat('output_Branin_rs/10/result.mat')
data2 = loadmat('output_Branin_rs/20/result.mat')
data3 = loadmat('output_Branin_rs/30/result.mat')
data4 = loadmat('output_Branin_rs/40/result.mat')
data5 = loadmat('output_Branin_rs/50/result.mat')

f_opt1 = data1['fun'].reshape(-1)
f_opt2 = data2['fun'].reshape(-1)
f_opt3 = data3['fun'].reshape(-1)
f_opt4 = data4['fun'].reshape(-1)
f_opt5 = data5['fun'].reshape(-1)

loss1 = data1['loss'].reshape(-1)
loss2 = data2['loss'].reshape(-1)
loss3 = data3['loss'].reshape(-1)
loss4 = data4['loss'].reshape(-1)
loss5 = data5['loss'].reshape(-1)

time1 = data1['time'].reshape(-1)
time2 = data2['time'].reshape(-1)
time3 = data3['time'].reshape(-1)
time4 = data4['time'].reshape(-1)
time5 = data5['time'].reshape(-1)

training_nrmse1 = data1['training_nrmse'].reshape(-1)
training_nrmse2 = data2['training_nrmse'].reshape(-1)
training_nrmse3 = data3['training_nrmse'].reshape(-1)
training_nrmse4 = data4['training_nrmse'].reshape(-1)
training_nrmse5 = data5['training_nrmse'].reshape(-1)

f_opt_rs = [np.array(f_opt1), np.array(f_opt2), np.array(f_opt3), 
             np.array(f_opt4), np.array(f_opt5)]

loss_rs = [np.array(loss1), np.array(loss2), np.array(loss3), 
             np.array(loss4), np.array(loss5)]

time_rs = [np.array(time1), np.array(time2), np.array(time3), 
             np.array(time4), np.array(time5)]

training_nrmse_rs = [np.array(training_nrmse1), np.array(training_nrmse2), np.array(training_nrmse3), 
             np.array(training_nrmse4), np.array(training_nrmse5)]

'''
Data from HB
'''
data1 = loadmat('output_Branin_hb/10/result.mat')
data2 = loadmat('output_Branin_hb/20/result.mat')
data3 = loadmat('output_Branin_hb/30/result.mat')
data4 = loadmat('output_Branin_hb/40/result.mat')
data5 = loadmat('output_Branin_hb/50/result.mat')

f_opt1 = data1['fun'].reshape(-1)
f_opt2 = data2['fun'].reshape(-1)
f_opt3 = data3['fun'].reshape(-1)
f_opt4 = data4['fun'].reshape(-1)
f_opt5 = data5['fun'].reshape(-1)

loss1 = data1['loss'].reshape(-1)
loss2 = data2['loss'].reshape(-1)
loss3 = data3['loss'].reshape(-1)
loss4 = data4['loss'].reshape(-1)
loss5 = data5['loss'].reshape(-1)

time1 = data1['time'].reshape(-1)
time2 = data2['time'].reshape(-1)
time3 = data3['time'].reshape(-1)
time4 = data4['time'].reshape(-1)
time5 = data5['time'].reshape(-1)

training_nrmse1 = data1['training_nrmse'].reshape(-1)
training_nrmse2 = data2['training_nrmse'].reshape(-1)
training_nrmse3 = data3['training_nrmse'].reshape(-1)
training_nrmse4 = data4['training_nrmse'].reshape(-1)
training_nrmse5 = data5['training_nrmse'].reshape(-1)

f_opt_hb = [np.array(f_opt1), np.array(f_opt2), np.array(f_opt3), 
             np.array(f_opt4), np.array(f_opt5)]

loss_hb = [np.array(loss1), np.array(loss2), np.array(loss3), 
             np.array(loss4), np.array(loss5)]

time_hb = [np.array(time1), np.array(time2), np.array(time3), 
             np.array(time4), np.array(time5)]

training_nrmse_hb = [np.array(training_nrmse1), np.array(training_nrmse2), np.array(training_nrmse3), 
             np.array(training_nrmse4), np.array(training_nrmse5)]

'''
Data from BOHB
'''
data1 = loadmat('output_Branin_bohb/10/result.mat')
data2 = loadmat('output_Branin_bohb/20/result.mat')
data3 = loadmat('output_Branin_bohb/30/result.mat')
data4 = loadmat('output_Branin_bohb/40/result.mat')
data5 = loadmat('output_Branin_bohb/50/result.mat')

f_opt1 = data1['fun'].reshape(-1)
f_opt2 = data2['fun'].reshape(-1)
f_opt3 = data3['fun'].reshape(-1)
f_opt4 = data4['fun'].reshape(-1)
f_opt5 = data5['fun'].reshape(-1)

loss1 = data1['loss'].reshape(-1)
loss2 = data2['loss'].reshape(-1)
loss3 = data3['loss'].reshape(-1)
loss4 = data4['loss'].reshape(-1)
loss5 = data5['loss'].reshape(-1)

time1 = data1['time'].reshape(-1)
time2 = data2['time'].reshape(-1)
time3 = data3['time'].reshape(-1)
time4 = data4['time'].reshape(-1)
time5 = data5['time'].reshape(-1)

training_nrmse1 = data1['training_nrmse'].reshape(-1)
training_nrmse2 = data2['training_nrmse'].reshape(-1)
training_nrmse3 = data3['training_nrmse'].reshape(-1)
training_nrmse4 = data4['training_nrmse'].reshape(-1)
training_nrmse5 = data5['training_nrmse'].reshape(-1)

f_opt_bohb = [np.array(f_opt1), np.array(f_opt2), np.array(f_opt3), 
             np.array(f_opt4), np.array(f_opt5)]

loss_bohb = [np.array(loss1), np.array(loss2), np.array(loss3), 
             np.array(loss4), np.array(loss5)]

time_bohb = [np.array(time1), np.array(time2), np.array(time3), 
             np.array(time4), np.array(time5)]

training_nrmse_bohb = [np.array(training_nrmse1), np.array(training_nrmse2), np.array(training_nrmse3), 
             np.array(training_nrmse4), np.array(training_nrmse5)]



'''
Plot
'''
data1 = [np.random.rand(15) * 10 for i in range(5)]

labels = ['10', '20', '30', '40', '50']

tableau_palette = sns.color_palette("tab10")
blue = tableau_palette[0]  
orange = tableau_palette[1]
green = tableau_palette[2] 
red = tableau_palette[3]   
violet = tableau_palette[4]

plot_y = 4 

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
Plot for f
'''
fig, ax = plt.subplots(figsize=(12, plot_y))

# Plot each data set with a different offset and color
plot_box_trans(f_opt_gs, 0.00, 'violet', 'Data1')
plot_box_trans(f_opt_rs, 0.1, 'orange', 'Data2')
plot_box_trans(f_opt_ax, 0.2, 'green', 'Data5')
plot_box_trans(f_opt_hb, 0.3, 'red', 'Data3')
plot_box_trans(f_opt_bohb, 0.4, 'blue', 'Data4')
plt.axhline(y=0.397, color='black', linestyle='--')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([-10,6])
ax.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2, 4,6])
ax.set_ylabel(r'$\hat{y}(\mathbf{x}_d)$', fontsize=20)
ax.set_xlabel('Number of samples', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [
                   Patch(facecolor='violet', label='Grid Search'),
                   Patch(facecolor='orange', label='Random Search'),
                   Patch(facecolor='green', label='BO'),
                   Patch(facecolor='red', label='HB'),
                   Patch(facecolor='blue', label='BOHB')
]

ax.legend(handles=legend_elements, loc='lower right', fontsize=20, ncol=3)

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('Converge_box_f.png', format='png', dpi=300)
plt.show()

'''
Plot for loss
'''
fig, ax = plt.subplots(figsize=(12, plot_y))

# Plot each data set with a different offset and color
plot_box_trans(loss_gs, 0.00, 'violet', 'Data1')
plot_box_trans(loss_rs, 0.1, 'orange', 'Data2')
plot_box_trans(loss_ax, 0.2, 'green', 'Data5')
plot_box_trans(loss_hb, 0.3, 'red', 'Data3')
plot_box_trans(loss_bohb, 0.4, 'blue', 'Data4')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([0,40])
ax.set_yticks([0, 5,10,15,20,25,30,35,40])
ax.set_ylabel('Loss', fontsize=20)
ax.set_xlabel('Number of samples', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [
                   Patch(facecolor='violet', label='Grid Search'),
                   Patch(facecolor='orange', label='Random Search'),
                   Patch(facecolor='green', label='BO'),
                   Patch(facecolor='red', label='HB'),
                   Patch(facecolor='blue', label='BOHB')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=20, ncol=3)

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('Converge_box_loss.png', format='png', dpi=300)
plt.show()

'''
Plot for time
'''
fig, ax = plt.subplots(figsize=(12, plot_y))

# Plot each data set with a different offset and color
plot_box_trans(time_gs, 0.00, 'violet', 'Data1')
plot_box_trans(time_rs, 0.1, 'orange', 'Data2')
plot_box_trans(time_ax, 0.2, 'green', 'Data5')
plot_box_trans(time_hb, 0.3, 'red', 'Data3')
plot_box_trans(time_bohb, 0.4, 'blue', 'Data4')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([10**1, 10**6])
ax.set_yticks([10**1, 10**2,10**3,10**4,10**5,10**6])
ax.set_ylabel('Time (s)', fontsize=20)
ax.set_xlabel('Number of samples', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.yscale("log")

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [
                   Patch(facecolor='violet', label='Grid Search'),
                   Patch(facecolor='orange', label='Random Search'),
                   Patch(facecolor='green', label='BO'),
                   Patch(facecolor='red', label='HB'),
                   Patch(facecolor='blue', label='BOHB')
]

ax.legend(handles=legend_elements, loc='best', fontsize=20, ncol=3)

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('Converge_box_time.png', format='png', dpi=300)
plt.show()

'''
Plot for training_nrmse

fig, ax = plt.subplots(figsize=(12, plot_y))

# Plot each data set with a different offset and color
plot_box_trans(training_nrmse_gs, 0.00, 'violet', 'Data1')
plot_box_trans(training_nrmse_rs, 0.1, 'orange', 'Data2')
plot_box_trans(training_nrmse_ax, 0.2, 'green', 'Data5')
plot_box_trans(training_nrmse_hb, 0.3, 'red', 'Data3')
plot_box_trans(training_nrmse_bohb, 0.4, 'blue', 'Data4')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
# ax.set_ylim([-10,6])
# ax.set_yticks([-10, -8, -6, -4, -2, 0, 2, 4,6])
ax.set_ylabel('nRMSE', fontsize=20)
ax.set_xlabel('Number of samples', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

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

ax.legend(handles=legend_elements, loc='lower right', fontsize=20, ncol=2)

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('Converge_box_training_nrmse.png', format='png', dpi=300)
plt.show()
'''

'''
Plot for x1

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each data set with a different offset and color
plot_box_trans(x1_opt_ax, 0.00, 'blue', 'Data1')
plot_box_trans(x1_opt_gs, 0.15, 'green', 'Data2')
plot_box_trans(x1_opt_rs, 0.30, 'red', 'Data3')
plt.axhline(y=3.1415, color='black', linestyle='--')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylabel('Minimum value for $\hat{x}_{d_1}$', fontsize=20 )
ax.set_xlabel('Number of samples', fontsize=20)
ax.set_ylim([2.5, 4.5])
ax.set_yticks([2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4, 4.25, 4.5])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='BO and MABO'),
                   Patch(facecolor='green', label='Grid Search'),
                   Patch(facecolor='red', label='Random Search'),
                   mlines.Line2D([], [], color='black', linestyle='--',  markersize=15, label='Minimum')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('Converge_box_x1.png', format='png', dpi=300)
plt.show()
'''

'''
Plot for x2

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each data set with a different offset and color
plot_box_trans(x2_opt_ax, 0.00, 'blue', 'Data1')
plot_box_trans(x2_opt_gs, 0.15, 'green', 'Data2')
plot_box_trans(x2_opt_rs, 0.30, 'red', 'Data3')
plt.axhline(y=2.275, color='black', linestyle='--')

# Adjust the x-axis
ax.set_xticks(np.arange(len(data1)) + 0.15)  # Set x-ticks in the middle of the group of boxplots
ax.set_xticklabels(labels)
ax.set_ylim([0,5])
ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
ax.set_ylabel('Minimum value for $\hat{x}_{d_2}$', fontsize=20 )
ax.set_xlabel('Number of samples', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='BO and MABO'),
                   Patch(facecolor='green', label='Grid Search'),
                   Patch(facecolor='red', label='Random Search'),
                   mlines.Line2D([], [], color='black', linestyle='--',  markersize=15, label='Minimum')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

# Show the plot
ax.grid(True)
plt.tight_layout()
plt.savefig('Converge_box_x2.png', format='png', dpi=300)
plt.show()
'''