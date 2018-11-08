import matplotlib as mpl
mpl.use('agg')

import os
import glob
import argparse
import numpy as np
import pylab as plt
import rficnn as rfc
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=False, help='choose architecture', type=str, default='1')
parser.add_argument('--trsh', required=False, help='choose threshold', type=float, default=0.1)
parser.add_argument('--time_limit', action="store", type=int, default=60)
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--normal', action="store_true", default=False)

args = parser.parse_args()
arch = args.arch
threshold = args.trsh
time_limit = args.time_limit
learning_rate = args.learning_rate
one_hot = not args.normal

thresholds = [1e-10, 0.1]
th_labels = [0,1,2]

rfc.the_print('Chosen architecture is: '+args.arch+'; threshold =: '+str(threshold)+'; learning rate = '+str(learning_rate),bgc='green')
model_add = './models/multiclass_model_'+arch+'_'+str(threshold)

files_list = sorted(glob.glob('../../../data/hide_sims_train/calib_1year/*.fits'))

rfc.the_print('number of files: '+str(len(files_list)))
ws = 400

dp = rfc.DataProvider(files=files_list,label_name='RFI',
                      ny=ws,
                      one_hot=one_hot,
                      thresholds=thresholds,
                      th_labels=th_labels,
                      a_min=0, a_max=200)

_,nx,ny,nc = dp(1)[1].shape

conv = rfc.ConvolutionalLayers(nx=nx,ny=ny,n_channel=1,n_class=nc,
                               restore=os.path.exists(model_add),
                               model_add=model_add,arch_file_name='arch_'+args.arch)

n_rounds = 10
if args.train:
    for i in range(n_rounds):
        rfc.the_print('ROUND: '+str(i)+', learning rate='+str(learning_rate),bgc='blue')
        conv.train(data_provider=dp,training_epochs = 10000000,n_s = 10,learning_rate = learning_rate, dropout=0.7, time_limit=time_limit//n_rounds, verbose=1)
        learning_rate = learning_rate/4.

else:
    import pickle

    pred_dir = 'predictions/threeclass_'+arch+'_'+str(threshold)+'/'
    rfc.ch_mkdir(pred_dir)
    res_file = 'results/threeclass_'+arch+'_'+str(threshold)
    rfc.ch_mkdir('results')

    weights = conv.get_filters()
    with open(res_file+'_filters', 'w') as filehandler:
        pickle.dump(weights, filehandler)
#    np.save(res_file+'_filters',weights)

    test_files = sorted(glob.glob('../../../data/hide_sims_test/calib_1month/*.fits'))

    times = []
    pr_list = []
    roc_list = []
    auc_list = []

    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,300,endpoint=1)

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

        time = np.linspace(0,24,14400)
        frequency = np.linspace(9.904950253309999653e+02, 1.259066105108820011e+03, 276)

        xticklabels = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6])
        xticks = xticklabels/time[1]
        yticks = [0,50,100,150,200,250]
        yticklabels = [990, 1040, 1090, 2040, 2090,3040]

        mpl.rc('font', size=20)

        fig, ax = plt.subplots(figsize=(18,4))
        plt.xlabel('Time/[s]', fontsize=20)
        plt.ylabel('Frequency/[MHz]', fontsize=20)
        train = ax.imshow(data[0,:,:,0],aspect='auto', origin='lower')
        plt.xticks(xticks, xticklabels)
        plt.yticks(yticks,yticklabels)
        plt.colorbar(train)
        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'_data.jpg',dpi=60)
        plt.close()


        fig2, ax2 = plt.subplots(figsize=(18,4))
        plt.xlabel('Time/[s]', fontsize=20)
        plt.ylabel('Frequency/[MHz]', fontsize=20)
        mask_show = ax2.imshow(np.argmax(mask[0,:,:,:],axis=2),aspect='auto', origin='lower')
        plt.xticks(xticks, xticklabels)
        plt.yticks(yticks, yticklabels)
        plt.colorbar(mask_show)
        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'_mask.jpg',dpi=60)
        plt.close()

        fig3, ax3 = plt.subplots(figsize=(18,4))
        plt.xlabel('Time/[s]', fontsize=20)
        plt.ylabel('Frequency/[MHz]', fontsize=20)
        pred_show = ax3.imshow(pred[0,:,:,0],aspect='auto', origin='lower')
        plt.xticks(xticks, xticklabels)
        plt.yticks(yticks, yticklabels)
        plt.colorbar(pred_show)
        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'_pred.jpg',dpi=60)
        plt.close()


        np.save(pred_dir+fname+'_mask',mask)
        np.save(pred_dir+fname+'_pred',pred)

        

        y_true = mask.reshape(-1).astype(int)
        y_score = pred.reshape(-1)
        y_score /= y_score.max()

        recall,precision = rfc.prc(y_true, y_score, trsh)
        pr_list.append(np.stack([precision,recall]).T)
        
        fpr,tpr = rfc.rocc(y_true, y_score, trsh)
        roc_list.append(np.stack([fpr,tpr]).T)

# check confusion matrix
        tp,fp,tn,fn = rfc.tfpnr(y_true, y_score)
        confusion_matrix = [tp,fp,tn,fn]

    np.save(res_file+'_confusion_matrix', confusion_matrix)    
    np.save(res_file+'_pr',np.array(pr_list))
    np.save(res_file+'_roc',np.array(roc_list))         
    print(np.mean(times))




