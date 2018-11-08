import matplotlib as mpl
mpl.use('agg')

import sys
import glob
import numpy as np
import argparse
from time import time
sys.path.insert(0,'../unet/tf_unet')
import pylab as plt
import rficnn as rfc
from tf_unet import unet
from tf_unet import util
from IPython.display import clear_output

parser = argparse.ArgumentParser()
parser.add_argument('--feat', required=False, help='choose architecture', type=int, default=16)
parser.add_argument('--trsh', required=False, help='choose threshold', type=float, default=0.1)
parser.add_argument('--test', action="store_true", default=False)
args = parser.parse_args()
threshold = args.trsh
features_root = args.feat


files_list = sorted(glob.glob('../data/data/training/*.h5'))
ws = 400

dp = rfc.DataProvider(ny=ws,a_min=0, a_max=200, files=files_list,label_name='gt_mask',n_class=2)
_,nx,ny,_ = dp(1)[0].shape

training_iters = 100
epochs = 300
model_dir = './models/unet_'+str(features_root)+'_'+str(threshold)

net = unet.Unet(channels=dp.channels, 
                n_class=dp.n_class, 
                layers=3, 
                features_root=features_root,
                cost_kwargs=dict(regularizer=0.001))

if not args.test:
    trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(dp, model_dir, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.5, 
                         display_step=1000000)

else:

    test_files = sorted(glob.glob('../data/data/test/*.h5'))

    pred_dir = 'predictions/unet_'+str(features_root)+'/'
    rfc.ch_mkdir(pred_dir)
    res_file = 'results/unet_'+str(features_root)+'_'+str(threshold)
    rfc.ch_mkdir('results')

    times = []
    pr_list = []
    roc_list = []
    auc_list = []
    
    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,300,endpoint=1)

    for fil in test_files:

        fname = fil.split('/')[-1]
        dp = rfc.DataProvider(a_min=0, a_max=200, files=[fil],label_name='gt_mask',n_class=2)
        data,mask = dp(1)
        
        pred,dt = net.predict(model_dir+'/model.cpkt', data,time_it=1)
        times.append(dt)
        mask = util.crop_to_shape(mask, pred.shape)

        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
        ax1.imshow(data[0,:,:,0],aspect='auto')
        ax2.imshow(mask[0,:,:,1],aspect='auto')
        ax3.imshow(pred[0,:,:,1],aspect='auto')

        np.save(pred_dir+fname+'_mask',mask)
        np.save(pred_dir+fname+'_pred',pred)

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'.jpg',dpi=30)
        plt.close()

        y_true = mask[0,:,:,1].reshape(-1).astype(int)
        y_score = pred[0,:,:,1].reshape(-1)
        y_score /= y_score.max()

        recall,precision = rfc.prc(y_true, y_score, trsh)
        pr_list.append(np.stack([precision,recall]).T)
        
        fpr,tpr = rfc.rocc(y_true, y_score, trsh)
        roc_list.append(np.stack([fpr,tpr]).T)
        
    np.save(res_file+'_pr',np.array(pr_list))
    np.save(res_file+'_roc',np.array(roc_list))         
    print np.mean(times)     




