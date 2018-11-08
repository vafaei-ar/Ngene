import matplotlib as mpl
mpl.use('agg')

import os
import numpy as np
import pylab as plt

threshod = 0.1


fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1)

for mode in ['0','1','2','3','unet_16','unet_32']:


    res_file = 'results/'+mode+'_'+str(threshold)        
    pr_list = np.load(res_file+'_pr.npy')
    roc_list = np.load(res_file+'_roc.npy')         

    print pr_list.shape,roc_list.shape


    aucv = np.trapz(tpr, fpr)


exit()

ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('recall')
ax1.set_ylabel('precision')


ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
#    
plt.legend(loc='best')

plt.savefig('ROC_PR.jpg')
plt.close()



def decompose(num):
    l1,l2 = 0,0
    sqr = l1**2
    while sqr<num:
        l1 += 1
        sqr = l1**2
    ans = l1*l2
    while ans<num:
        l2 += 1
        ans = l1*l2
    return l1,l2
    
for mode in ['0','1','2','3']:
    res_file = 'results/'+mode+'_'+str(threshold) 
    weights = np.load(res_file+'_filters')
    
    for ww in weights:
        print ww.shape
    
exit()

num = 144 
l1,l2 = decompose(num)

# define the figure size and grid layout properties
figsize = (l1,l2)
cols = 3
gs = gridspec.GridSpec(l1,l2)
gs.update(hspace=0.0,wspace=0.0)
# define the data for cartesian plots

fig1 = plt.figure(num=1, figsize=figsize)
ax = []
for col in range(l1):
    for row in range(l2):
        ax.append(fig1.add_subplot(gs[row, col]))
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])



