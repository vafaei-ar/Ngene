import glob 
import numpy as np
from astropy.io import fits
import h5py
from PIL import Image
from scipy.misc import toimage

'''
Created on Aug 18, 2016
original author: jakeret

Modified at: March 20, 2018
by:           yabebal fantaye

Modified on July 20, 2018
by: 		Anke van Dyk
'''

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

class BaseDataProvider(object):
    
    def __init__(self,files,nx,ny,
                     a_min=-np.inf, a_max=np.inf,
                     rgb=False,verbose=0,channels=1,n_class=1):
        
        self.a_min = a_min 
        self.a_max = a_max 
        self.verbose = verbose
        self.nx = nx
        self.ny = ny
        self.rgb = rgb       
        self.files = files
        self.channels = channels
        self.n_class = n_class

        assert len(files) > 0, "No training files"
        if verbose: print("Number of files used: %s"%len(files))

    def read_chunck(self): 
        filename = self.files[self.file_idx]
        data,label = np.load(filename)
        return data,label
           
    def _next_data(self):
        self.file_idx = np.random.choice(len(self.files))
        data, label = self.read_chunck()
        
        assert self.nx<=data.shape[0],'Error, Somthing is wrong is the given nx. Seems better to decrease it!'
        
        n_try = -1
        if self.ny>0:
            n_try += 1
            ny = data.shape[1]
            while ny < self.ny:
                print('Warning! something is wrong with {} dimensions.'.format(self.files[self.file_idx]))
                self.file_idx = np.random.choice(len(self.files))
                data, label = self.read_chunck()
                ny = data.shape[1]
                assert n_try<1000,'Error, Somthing is wrong is the given ny. Seems better to decrease it!'
            
        return data, label           

    def pre_process(self, data, label):
        
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        data = np.expand_dims(data, axis=-1)
        if self.rgb:
            data = to_rgb(data).astype(np.uint8)
            
#        label -= np.amin(label)
#        label /= np.amax(label)

        if self.n_class == 1:
            return data,np.expand_dims(label, axis=-1)
        elif self.n_class == 2:
            lx,ly = label.shape
            labels = np.zeros((lx, ly, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = 1-label
            return data,labels
        else:
            assert 0,'Error, illegal number of class. It only can be either 1 or 2!'
    
    def __call__(self, n):
        
        data, label = self._next_data()
        data, label = self.pre_process(data, label)
        nx,ny,nc = data.shape   
        assert nc==self.channels,'Error, problem with given number of channel!'
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        X[0] = data
        Y[0] = label
        for i in range(1,n):
            data, label = self._next_data()
            data, label = self.pre_process(data, label)
            X[i] = data
            Y[i] = label
    
        return X, Y
  
def get_slice(data,label,nx,ny):
    lx,ly = data.shape  
    if nx==0 or nx==lx:
        slx = slice(0, lx)                
    else:
        idx = np.random.randint(0, lx - nx)            
        slx = slice(idx, (idx+nx))       
    if ny==0 or ny==ly:
        sly = slice(0, ly)                
    else:
        idy = np.random.randint(0, ly - ny)            
        sly = slice(idy, (idy+ny))
    return data[slx, sly],label[slx, sly]
   
def threshold_mask(data,rfi,thresholds,th_labels=None):
    if not (isinstance(thresholds, list) or isinstance(thresholds, np.ndarray)):
        thresholds = [thresholds]

    n_trsh = len(thresholds)
    if th_labels is None:
        th_labels = np.arange(n_trsh)+1

    rel_rfi = np.abs(1.*rfi/data)
    mask = np.zeros(data.shape)
    for i in range(n_trsh-1):
        mask[(thresholds[i]<rel_rfi) & (thresholds[i+1]>=rel_rfi)] = th_labels[i]        
    mask[thresholds[n_trsh-1]<rel_rfi] = th_labels[n_trsh-1]
    
#    if verbose:
#        txt='percentage of rfi pixels with >{:.3e} % RFI fraction: {:.2e} %'
#        print( txt.format(threshold,
#               np.divide(100.*np.sum(label), np.product(label.shape)) ))
    return mask

def read_chunck_hdf5(filename,nx,ny,label_tag,thresholds=None,th_labels=None,verbose=0):
#    if thresholds is not None:
#        thresholds = thresholds
#    else:
#        thresholds = [0.01]
        
    with h5py.File(filename, "r") as fp:
        column_names = list(fp.keys())
        if label_tag is None or not (label_tag in column_names):
            print('You did not select any label tag for mask production, please choose!',list(column_names))
            assert label_tag in column_names, 'Error, Tag is not found!'
        
        data,label = np.array(fp['data']),np.array(fp[label_tag])
        data,label = get_slice(data,label,nx,ny)
        if label_tag=='rfi_map' and thresholds is not None:
            label = threshold_mask(data,label,thresholds,th_labels=th_labels)
            
    return data, label

def read_chunck_sdfits(filename,nx,ny,label_tag,thresholds=None,th_labels=None,verbose=0):
#    if thresholds is not None:
#        thresholds = thresholds
#    else:
#        thresholds = [0.01]
    
    f = fits.open(filename)        
    fp = f[1].data
    column_names = fp.columns.names
    if label_tag is None or not (label_tag in column_names):
        print('You did not select any label tag for mask production, please choose!',column_names)
        assert 'Error, Tag is not found!'

    data,label = fp["DATA"].squeeze().T,fp[label_tag].squeeze().T
    data,label = get_slice(data,label,nx,ny)
    if label_tag=='RFI' and thresholds is not None:
        label = threshold_mask(data,label,thresholds,th_labels=th_labels)
    f.close()
    return data, label

class DataProvider(BaseDataProvider):
    
    def __init__(self,files,nx=0,ny=0, 
                     label_name=None,thresholds=None,th_labels=None,
                     a_min=-np.inf, a_max=np.inf,
                     rgb=False,verbose=0,channels=1,n_class=1):
        
        super(DataProvider, self).__init__(files, nx, ny,
                     a_min=a_min, a_max=a_max,
                     rgb=rgb,verbose=verbose,channels=channels,n_class=n_class)
    
        self.label_tag = label_name
        self.thresholds = thresholds
        self.th_labels = th_labels

    def read_chunck(self):     
        filename = self.files[self.file_idx]
        ext = filename.split('.')[-1]
        
        nx,ny = self.nx,self.ny
        label_tag = self.label_tag
        thresholds = self.thresholds
        th_labels = self.th_labels
        verbose = self.verbose
        
        if ext == 'fits':
            data,label = read_chunck_sdfits(filename,nx,ny,label_tag,thresholds=thresholds,th_labels=th_labels,verbose=verbose)
        elif ext == 'h5':
            data,label = read_chunck_hdf5(filename,nx,ny,label_tag,thresholds=thresholds,th_labels=th_labels,verbose=verbose)
        else:
            assert 0,'Error, unsupported file format!'
            
        return data,label       
        
        
        
        
#def read_chunck_hdf5(filename,nx,ny,label_tag,threshold=None,verbose=0):
#    if threshold is not None:
#        threshold = threshold
#    else:
#        threshold = 0.01
#        
#    with h5py.File(filename, "r") as fp:
#        column_names = fp.keys()
#        if label_tag is None or not (label_tag in column_names):
#            print('You did not select any label tag for mask production, please choose!',column_names)
#            assert 'Error, Tag is not found!'

#        lx,ly = fp["data"].shape

#        if nx==0 or nx==lx:
#            slx = slice(0, lx)                
#        else:
#            idx = np.random.randint(0, lx - nx)            
#            slx = slice(idx, (idx+nx))
#            
#        if ny==0 or ny==ly:
#            sly = slice(0, ly)                
#        else:
#            idy = np.random.randint(0, ly - ny)            
#            sly = slice(idy, (idy+ny))

#        data = fp["data"][slx, sly]
#        label = fp[label_tag][slx, sly]

#        if label_tag=='rfi_map':
#            label = 100*np.abs(label)>threshold
#            if verbose:
#                txt='percentage of rfi pixels with >{:.3f} % RFI fraction: {:.2f} %'
#                print( txt.format(threshold,
#                       np.divide(100.*np.sum(label), np.product(label.shape)) ))                
#    return data, label        
