import matplotlib as mpl
mpl.use('agg')

import sys
import glob
import numpy as np
import argparse
from time import time
import tensorflow as tf                  
sys.path.insert(0,'../unet/tf_unet')
import pylab as plt
import rficnn as rfc
from dataprovider import DataProvider
from tf_unet import unet
from tf_unet import util
from IPython.display import clear_output

parser = argparse.ArgumentParser()

parser.add_argument('--layers', required=False, help='number of layers', type=int, default=3)
parser.add_argument('--feat', required=False, help='choose architecture', type=int, default=16)
parser.add_argument('--nx', required=False, help='frequency window', type=int, default=300)
parser.add_argument('--ny', required=False, help='time window', type=int, default=300)
parser.add_argument('--train', action="store_true", default=False)
args = parser.parse_args()
layers = args.layers
features_root = args.feat
nx = args.nx
ny = args.ny

files_list = sorted(glob.glob('../../data/kat7/dataset2/training*.h5'))
#nx,ny = 600,400

dp = DataProvider(nx=nx,ny=ny,a_min=0, a_max=200, files=files_list,label_name='mask',n_class=2)
_,nx,ny,_ = dp(1)[0].shape

print dp(3)[1].shape
#exit()

training_iters = 100
epochs = 1000

name = str(layers)+'_'+str(features_root)+'_'+str(nx)+'x'+str(ny)

model_dir = './models/kat7_unet_'+name

net = unet.Unet(channels=dp.channels, 
                n_class=2, 
                layers=layers, 
                features_root=features_root,
                cost_kwargs=dict(regularizer=0.001))

if args.train:
    trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(dp, model_dir, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.5, 
                         display_step=1000000,
                         restore=1)
			             
    print('')
    n_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Number of trainable variables: {:d}'.format(n_variables))

else:

    test_files = sorted(glob.glob('../../data/kat7/dataset2/test*.h5'))

    pred_dir = 'predictions/kat7_unet_'+name+'/'
    rfc.ch_mkdir(pred_dir)
    res_file = 'results/kat7_unet_'+name
    rfc.ch_mkdir('results')

    times = []
    pr_list = []
    roc_list = []
    auc_list = []
    
    clrs = ['b','r','g','k']
    trsh = np.linspace(1,0,300,endpoint=1)

    for fil in test_files:

        fname = fil.split('/')[-1]
        dp = DataProvider(a_min=0, a_max=200, files=[fil],label_name='mask',n_class=2)
        
        data0,mask0 = dp(1)
        
        nt = data0.shape[2]
        chunks = np.array_split(np.arange(nt).astype(int), 10)
        
        
        for ind in chunks:
            data,mask = data0[:,:,ind,:],mask0[:,:,ind,:]
            print data.shape,mask.shape
            
            pred,dt = net.predict(model_dir+'/model.cpkt', data,time_it=1)

            times.append(dt)
            _,kk,jj,_, = pred.shape
            mask = util.crop_to_shape(mask, pred.shape)[:,:kk,:jj,:]

            print mask.shape, pred.shape

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
            
            print y_true.shape, y_score.shape
            
            y_score /= y_score.max()

            recall,precision = rfc.prc(y_true, y_score, trsh)
            pr_list.append(np.stack([recall,precision]).T)
            
            fpr,tpr = rfc.rocc(y_true, y_score, trsh)
            roc_list.append(np.stack([fpr,tpr]).T)
            
            
            
#            data,mask = dp(1)
#            if features_root==32:
#                data,mask = data[:,:,30000:,:],mask[:,:,30000:,:]
#            print data.shape,mask.shape
#            
#            pred,dt = net.predict(model_dir+'/model.cpkt', data,time_it=1)
#            print 'doe'
#            times.append(dt)
#            _,kk,_,_, = pred.shape
#            mask = util.crop_to_shape(mask, pred.shape)[:,:kk,:,:]

#            print mask.shape, pred.shape

#            fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
#            ax1.imshow(data[0,:,:,0],aspect='auto')
#            ax2.imshow(mask[0,:,:,1],aspect='auto')
#            ax3.imshow(pred[0,:,:,1],aspect='auto')

#            np.save(pred_dir+fname+'_mask',mask)
#            np.save(pred_dir+fname+'_pred',pred)

#            plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
#            plt.savefig(pred_dir+fname+'.jpg',dpi=30)
#            plt.close()

#            y_true = mask[0,:,:,1].reshape(-1).astype(int)
#            y_score = pred[0,:,:,1].reshape(-1)
#            
#            print y_true.shape, y_score.shape
#            
#            y_score /= y_score.max()

#            recall,precision = rfc.prc(y_true, y_score, trsh)
#            pr_list.append(np.stack([recall,precision]).T)
#            
#            fpr,tpr = rfc.rocc(y_true, y_score, trsh)
#            roc_list.append(np.stack([fpr,tpr]).T)        
        
        
    np.save(res_file+'_pr',np.array(pr_list))
    np.save(res_file+'_roc',np.array(roc_list))         
    print np.mean(times)     




