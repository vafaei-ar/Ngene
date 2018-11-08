import matplotlib as mpl
mpl.use('agg')

import os
import glob
import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm
import rficnn as rfc

dss = ['training','validation','test']

rfc.ch_mkdir('plots')

for ds in dss:
    test_files = sorted(glob.glob('../../../data/kat7/dataset/'+ds+'/*.h5'))

    for fil in test_files:

        fname = fil.split('/')[-1]
        dp = rfc.DataProvider(a_min=0, a_max=100, files=[fil],label_name='mask')
        data,mask = dp(1)

        mask = mask[0,:,:,0]

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8))
        ax1.imshow(data[0,:,:,0], norm=LogNorm(),aspect='auto')
        ax2.imshow(mask,aspect='auto')

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig('./plots/'+ds+'_'+fname+'.jpg',dpi=100)
        plt.close()



