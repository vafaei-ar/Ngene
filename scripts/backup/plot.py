#! /home/gf/pakages/miniconda2/bin/python2.7
import matplotlib as mpl
mpl.use('agg')

import numpy as np
import matplotlib.pylab as plt
import glob
import h5py
import sys

sim_files = glob.glob('./TEST*'+sys.argv[1]+'.h5')
#sim_files = glob.glob('../data/tiny/*')
#sim_files = [sys.argv[1]]
for i,filename in enumerate(sim_files):
	print i
	out_name = filename[:-3]

	with h5py.File(filename,"r") as fp: 
		data = fp['data'].value
		mask = fp['mask'].value
		freq = fp['frequencies'].value
		time = fp['time'].value
		ra = fp['ra'].value
		dec = fp['dec'].value
		ref_channel = fp['ref_channel'].value

	data = np.clip(np.fabs(data), 0, 200)

	print data.min(), data.max()
	fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,8))
	img = ax1.imshow(data, aspect="auto")
	img = ax2.imshow(mask, aspect="auto")
	plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
	plt.savefig(filename+'.jpg',dpi=100)
	plt.close()
