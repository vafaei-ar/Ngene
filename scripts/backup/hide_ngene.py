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

args = parser.parse_args()
arch = args.arch
threshold = args.trsh
time_limit = args.time_limit
learning_rate = args.learning_rate

rfc.the_print('Chosen architecture is: '+args.arch+'; threshold =: '+str(threshold)+'; learning rate = '+str(learning_rate),bgc='green')
model_add = './models/model_'+arch+'_'+str(threshold)

files_list = sorted(glob.glob('../../data/hide_sims_train/calib_1year/*.fits'))
rfc.the_print('number of files: '+str(len(files_list)))
ws = 400

dp = rfc.DataProvider(ny=ws,a_min=0, a_max=200, files=files_list,label_name='RFI_MASK',threshold=threshold)
#print dp(5)[0].shape
_,nx,ny,_ = dp(1)[0].shape

conv = rfc.ConvolutionalLayers(nx=nx,ny=ny,n_channel=1,
                        restore=os.path.exists(model_add),
                        model_add=model_add,
                        arch_file_name='arch_'+args.arch)

n_rounds = 10
if args.train:
    for i in range(n_rounds):
        rfc.the_print('ROUND: '+str(i)+', learning rate='+str(learning_rate),bgc='blue')
        conv.train(data_provider=dp,training_epochs = 10000000,n_s = 100,learning_rate = learning_rate, dropout=0.7, time_limit=time_limit//n_rounds, verbose=1)
        learning_rate = learning_rate/4.

else:
    import pickle

    pred_dir = 'predictions/'+arch+'_'+str(threshold)+'/'
    rfc.ch_mkdir(pred_dir)
    res_file = 'results/'+arch+'_'+str(threshold)
    rfc.ch_mkdir('results')

    weights = conv.get_filters()
    with open(res_file+'_filters', 'w') as filehandler:
        pickle.dump(weights, filehandler)
#    np.save(res_file+'_filters',weights)

    test_files = sorted(glob.glob('../../data/hide_sims_test/calib_1month/*.fits'))

    times = []
    pr_list = []
    roc_list = []
    auc_list = []
    
    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,300,endpoint=1)

    for fil in test_files:

        fname = fil.split('/')[-1]
        dp = rfc.DataProvider(a_min=0, a_max=200, files=[fil],label_name='RFI_MASK')
        data,mask = dp(1)
        
        s = time()
        pred = conv.conv_large_image(data,pad=10,lx=276,ly=400)
        e = time()
        times.append(e-s)

        mask = mask[0,10:-10,:,0]
        pred = pred[10:-10,:]

        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
        ax1.imshow(data[0,:,:,0],aspect='auto')
        ax2.imshow(mask,aspect='auto')
        ax3.imshow(pred,aspect='auto')

        np.save(pred_dir+fname+'_mask',mask)
        np.save(pred_dir+fname+'_pred',pred)

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'.jpg',dpi=30)
        plt.close()

        y_true = mask.reshape(-1).astype(int)
        y_score = pred.reshape(-1)
        y_score /= y_score.max()

        recall,precision = rfc.prc(y_true, y_score, trsh)
        pr_list.append(np.stack([recall,recall]).T)
        
        fpr,tpr = rfc.rocc(y_true, y_score, trsh)
        roc_list.append(np.stack([fpr,tpr]).T)
        
    np.save(res_file+'_pr',np.array(pr_list))
    np.save(res_file+'_roc',np.array(roc_list))         
    print np.mean(times)     



