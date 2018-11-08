import matplotlib as mpl
mpl.use('agg')

import glob
import argparse
from time import time
import numpy as np
import pylab as plt
import rficnn as rfc

parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=False, help='choose architecture', type=str, default='1')
parser.add_argument('--trsh', required=False, help='choose threshold', type=float, default=0.1)
args = parser.parse_args()
threshold = args.trsh

rfc.the_print('Chosen architecture is: '+args.arch+' and threshod is: '+str(threshold),bgc='green')
model_add = './models/model_'+args.arch+'_'+str(threshold)

conv = ss.ConvolutionalLayers(nx=276,ny=400,n_channel=1,restore=1,
                                model_add=model_add,arch_file_name='arch_'+args.arch)

sim_files = glob.glob('../data/hide_sims_test/calib_1year/*.fits'))

times = []

for fil in sim_files:

    fname = fil.split('/')[-1]
    print fname
    data,mask = read_chunck_sdfits(fil,label_tag=RFI,threshold=0.1,verbose=0)
    data = np.clip(np.fabs(data), 0, 200)
    
    data -= data.min()
    data /= data.max()
    lnx,lny = data.shape
    
    s = time()
    pred = conv.conv_large_image(data.reshape(1,lnx,lny,1),pad=10,lx=276,ly=400)
    e = time()
    times.append(e-s)

    mask = mask[10:-10,:]
    pred = pred[10:-10,:]

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,8))
    ax1.imshow(data,aspect='auto')
    ax2.imshow(mask,aspect='auto')
    ax3.imshow(pred,aspect='auto')

    np.save('../comparison/'+fname+'_mask_'+sys.argv[1],mask)
    np.save('../comparison/'+fname+'_pred_'+sys.argv[1],pred)

    plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
    plt.savefig('../comparison/'+fname+'_'+sys.argv[1]+'.jpg',dpi=30)
    plt.close()
    
print np.mean(times)
