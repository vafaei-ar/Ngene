import matplotlib as mpl
mpl.use('agg')

import os
import glob
import argparse
import numpy as np
import pylab as plt
import rficnn as rfc
from time import time
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=False, help='choose architecture', type=str, default='0')
#parser.add_argument('--trsh', required=False, help='choose threshold', type=float, default=0.1)

args = parser.parse_args()
arch = 'arch_'+args.arch+'_3class'

mode = 'one_hot'

thresholds = [1e-10, 0.1]
th_labels = [0,1,2]

rfc.the_print('Chosen architecture is: '+args.arch,bgc='green')
model_add = './models/multiclass_model_'+arch+'_'+mode

test_files = sorted(glob.glob('/home/anke/HIDE_simulations/hide_sims_test/calib_1month/*.fits'))
rfc.the_print('number of files: '+str(len(test_files)))
ws = 400

dp = rfc.DataProvider(files=test_files,label_name='RFI',
                      ny=ws,
                      one_hot=1,
                      thresholds=thresholds,
                      th_labels=th_labels,
                      a_min=0, a_max=200)

_,nx,ny,nc = dp(1)[1].shape
print(dp(1)[1].shape)

conv = rfc.ConvolutionalLayers(nx=nx,ny=ny,n_channel=1,n_class=nc,
                               restore=os.path.exists(model_add),
                               model_add=model_add,
                               arch_file_name=arch)


res_file = 'results/threeclass_'+arch+'_'+mode
rfc.ch_mkdir('results')
rfc.ch_mkdir('predictions')
pred_dir = 'predictions/'

times = []
cnf_matrix = []

for fil in test_files:

    fname = fil.split('/')[-1]
    dp = rfc.DataProvider(files=[fil],label_name='RFI',
                  one_hot=1,
                  thresholds=thresholds,
                  th_labels=th_labels,
                  a_min=0, a_max=200)
    data,mask = dp(1)
    
    s = time()
    pred = conv.conv_large_image(data,pad=10,lx=276,ly=400)
    e = time()
    times.append(e-s)        
    
    data = data[0,10:-10,10:-10,0]
    y_true = mask[0,10:-10,10:-10,:]

    pred = pred[10:-10,10:-10,:]
    y_true = np.argmax(y_true,axis=-1).astype(int).reshape(-1)
    y_score = np.argmax(pred,axis=-1).astype(int).reshape(-1)

    

    #time = np.linspace(0,24,14400)
    #frequency = np.linspace(9.904950253309999653e+02, 1.259066105108820011e+03, 276)

    time_pixels = np.arange(0,14400,1)
    timestamps = np.linspace(0,24, 14400)
    channels = np.arange(0, 276, 1)
    frequency = np.linspace(9.904950253309999653e+02, 1.259066105108820011e+03, 276)

    xticklabels = np.arange(0,24,6)
    xticks = np.arange(0,14400,3600)

    yticks = np.arange(10, 270, 50)
    yticklabels = np.array([1000, 1050, 1100, 1150, 1200, 1250])

    mpl.rc('font', size=20)
# first figure
    fig, ax = plt.subplots(figsize=(18,8))
    train = ax.imshow(data,aspect='auto', origin='lower')

    plt.xlabel('Time [h]', fontsize=20)
    plt.ylabel('Frequency [MHz]', fontsize=20)
    plt.xticks(xticks, xticklabels, fontsize=20)
    plt.yticks(yticks, yticklabels, fontsize=20)

    cbar = fig.colorbar(train)
    cbar.ax.tick_params(labelsize=20) 
    cbar.set_label('Normalized Amplitude', fontsize=20)

    plt.tight_layout()
    plt.savefig(pred_dir+fname+'_data.jpg',dpi=150)
    plt.close()

# second figure
    fig2, ax2 = plt.subplots(figsize=(18,8))
    cmap = plt.get_cmap('viridis', 3)
    mask_show = ax2.imshow(np.argmax(mask[0,:,:,:],axis=2),aspect='auto', origin='lower', cmap=cmap)

    plt.xlabel('Time [h]', fontsize=20)
    plt.ylabel('Frequency [MHz]', fontsize=20)
    plt.xticks(xticks, xticklabels, fontsize=20)
    plt.yticks(yticks, yticklabels, fontsize=20)

    cbar2 = fig2.colorbar(mask_show)
    cbar2.ax.tick_params(labelsize=20) 
    cbar2.set_ticks([0.33,1, 1.66])
    cbar2.set_ticklabels(th_labels)

    plt.tight_layout()
    plt.savefig(pred_dir+fname+'_mask.jpg',dpi=150)
    plt.close()

# third figure
    fig3, ax3 = plt.subplots(figsize=(18,8))
    cmap2 = plt.get_cmap('viridis', 3)
    pred_show = ax3.imshow(np.argmax(pred[:,:,:],axis=2),aspect='auto', origin='lower', cmap=cmap)

    yticks = np.arange(10, 250, 50)
    yticklabels = np.array([1000, 1050, 1100, 1150, 1200])

    plt.xlabel('Time [h]', fontsize=20)
    plt.ylabel('Frequency [MHz]', fontsize=20)
    plt.xticks(xticks, xticklabels, fontsize=20)
    plt.yticks(yticks, yticklabels, fontsize=20)

    cbar3 = fig3.colorbar(pred_show)
    cbar3.ax.tick_params(labelsize=20) 
    cbar3.set_ticks([0.33,1, 1.66])
    cbar3.set_ticklabels(th_labels)
    
    plt.tight_layout()
    plt.savefig(pred_dir+fname+'_pred.jpg',dpi=150)
    plt.close()


    np.save(pred_dir+'_'+fname+'_mask',mask)
    np.save(pred_dir+'_'+fname+'_pred',pred)

    cnf_matrix.append(confusion_matrix(y_true, y_score))

np.save(res_file+'_confusion_matrix', np.array(cnf_matrix))    
    
print(np.mean(times))




