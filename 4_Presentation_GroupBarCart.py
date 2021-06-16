import matplotlib.pyplot as plt
import numpy as np

labels = ['No_cost_min', 'Isotonic_Cal', 'Cost_B_ovrSmpl', 'Cost_B_Weighting']
#labels = ['basic', 'under-sampling', 'over-sampling', 'combination']
#labels = ['No_cost_min', 'Isotonic_Cal', 'Cost_B_ovrSmpl', 'Cost_B_Weighting']
#labels = ['No_cost_min', 'No_Calibration', 'Platt_Scaling', 'Isotonic_Cal']
#labels = ['No_Sampling', 'Undersampling', 'Oversampling', 'Combination']
#labels = ['No_weighting', 'with_weights_RandForest', 'with_weights_LogRegr']

Loss = [18761, 8789, 17845, 8749]
Accuracy = [0.7 , 0.5 , 0.65, 0.7]
LossAdjusted = []
MX = np.amax(Loss)
MN = np.amin(Loss)

#shift the loss prices to the 0-1 interval for better optical comparison with the accuracy  
for i in range(4):
    a = (Loss[i] - MN) / (MX - MN)
    a = round(a, 2)
    LossAdjusted.append(a)    

#-------------------------------------------CHART 1----------------------------------------------
#---------------------------------------Group Bar Chart------------------------------------------

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, LossAdjusted, width, label='LossAdjusted' , color=[ 'blue'])
rects2 = ax.bar(x + width/2, Accuracy, width, label='Accuracy', color=[ 'orange'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Method results')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

#-------------------------------------------CHART 2----------------------------------------------

width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Loss, width, label='Loss', color=['blue'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Method results')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)

fig.tight_layout()

plt.show()