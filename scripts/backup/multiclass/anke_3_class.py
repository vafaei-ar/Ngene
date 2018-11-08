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
parser.add_argument('--arch', required=False, help='choose architecture', type=str, default='1')
#parser.add_argument('--trsh', required=False, help='choose threshold', type=float, default=0.1)
parser.add_argument('--time_limit', action="store", type=int, default=60)
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--embed', action="store_true", default=False)

args = parser.parse_args()
arch = args.arch
#threshold = args.trsh
time_limit = args.time_limit
learning_rate = args.learning_rate
one_hot = not args.embed

if one_hot:
    arch = 'arch_'+args.arch+'_3class'
    mode = 'one_hot'
else:
    arch = 'arch_'+args.arch
    mode = 'embed'

thresholds = [1e-10, 0.1]
th_labels = [0,1,2]

rfc.the_print('Chosen architecture is: '+args.arch+'; normal =: '+mode+'; learning rate = '+str(learning_rate),bgc='green')
model_add = './models/multiclass_model_'+arch+'_'+mode

files_list = sorted(glob.glob('/home/anke/HIDE_simulations/hide_sims_test/calib_1month/*.fits'))

rfc.the_print('number of files: '+str(len(files_list)))
ws = 400

dp = rfc.DataProvider(files=files_list,label_name='RFI',
                      ny=ws,
                      one_hot=one_hot,
                      thresholds=thresholds,
                      th_labels=th_labels,
                      a_min=0, a_max=200)

_,nx,ny,nc = dp(1)[1].shape
print(dp(1)[1].shape)

conv = rfc.ConvolutionalLayers(nx=nx,ny=ny,n_channel=1,n_class=nc,
                               restore=os.path.exists(model_add),
                               model_add=model_add,
                               arch_file_name=arch)

n_rounds = 5
if args.train:
    for i in range(n_rounds):
        rfc.the_print('ROUND: '+str(i)+', learning rate='+str(learning_rate),bgc='blue')
        conv.train(data_provider=dp,training_epochs = 10000000,n_s = n_rounds,learning_rate = learning_rate, dropout=0.7, time_limit=time_limit//n_rounds, verbose=1)
        learning_rate = learning_rate/4.

else:
    import pickle

#    pred_dir = 'predictions/threeclass_'+arch+'_'+mode+'/'
#    rfc.ch_mkdir(pred_dir)
    res_file = 'results/threeclass_'+arch+'_'+mode
    rfc.ch_mkdir('results')

#    weights = conv.get_filters()
#    with open(res_file+'_filters', 'w') as filehandler:
#        pickle.dump(weights, filehandler)
#    np.save(res_file+'_filters',weights)

    test_files = sorted(glob.glob('../../../data/hide_sims_test/calib_1month/*.fits'))

    times = []
    pr_list = []
    roc_list = []
    auc_list = []
    cnf_matrix = []

    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,300,endpoint=1)

    for fil in test_files:

        fname = fil.split('/')[-1]
        dp = rfc.DataProvider(files=[fil],label_name='RFI',
                      one_hot=one_hot,
                      thresholds=thresholds,
                      th_labels=th_labels,
                      a_min=0, a_max=200)
        data,mask = dp(1)
        
        s = time()
        pred = conv.conv_large_image(data,pad=10,lx=276,ly=400)
        e = time()
        times.append(e-s)        
        
        y_true = mask[0,10:-10,10:-10,:]
        
        if one_hot:
            pred = pred[10:-10,10:-10,:]
            y_true = np.argmax(y_true,axis=-1).astype(int).reshape(-1)
            y_score = np.argmax(pred,axis=-1).astype(int).reshape(-1)
        else:
            pred = pred[10:-10,10:-10]
            y_true = y_true.astype(int).reshape(-1)
            y_score = pred
            y_score = y_score-y_score.min()
            y_score = nc*y_score/y_score.max()-0.5
            y_score = np.around(y_score).astype(int).reshape(-1)
        

#        time = np.linspace(0,24,14400)
#        frequency = np.linspace(9.904950253309999653e+02, 1.259066105108820011e+03, 276)

#        xticklabels = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6])
#        xticks = xticklabels/time[1]
#        yticks = [0,50,100,150,200,250]
#        yticklabels = [990, 1040, 1090, 2040, 2090,3040]

#        mpl.rc('font', size=20)

#        fig, ax = plt.subplots(figsize=(18,4))
#        plt.xlabel('Time/[s]', fontsize=20)
#        plt.ylabel('Frequency/[MHz]', fontsize=20)
#        train = ax.imshow(data[0,:,:,0],aspect='auto', origin='lower')
#        plt.xticks(xticks, xticklabels)
#        plt.yticks(yticks,yticklabels)
#        plt.colorbar(train)
#        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
#        plt.savefig(pred_dir+fname+'_data.jpg',dpi=60)
#        plt.close()


#        fig2, ax2 = plt.subplots(figsize=(18,4))
#        plt.xlabel('Time/[s]', fontsize=20)
#        plt.ylabel('Frequency/[MHz]', fontsize=20)
#        mask_show = ax2.imshow(np.argmax(mask[0,:,:,:],axis=2),aspect='auto', origin='lower')
#        plt.xticks(xticks, xticklabels)
#        plt.yticks(yticks, yticklabels)
#        plt.colorbar(mask_show)
#        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
#        plt.savefig(pred_dir+fname+'_mask.jpg',dpi=60)
#        plt.close()

#        fig3, ax3 = plt.subplots(figsize=(18,4))
#        plt.xlabel('Time/[s]', fontsize=20)
#        plt.ylabel('Frequency/[MHz]', fontsize=20)
#        pred_show = ax3.imshow(pred[0,:,:,0],aspect='auto', origin='lower')
#        plt.xticks(xticks, xticklabels)
#        plt.yticks(yticks, yticklabels)
#        plt.colorbar(pred_show)
#        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
#        plt.savefig(pred_dir+fname+'_pred.jpg',dpi=60)
#        plt.close()


#        np.save(pred_dir+fname+'_mask',mask)
#        np.save(pred_dir+fname+'_pred',pred)

        cnf_matrix.append(confusion_matrix(y_true, y_score))

    np.save(res_file+'_confusion_matrix', np.array(cnf_matrix))    
        
    print(np.mean(times))




