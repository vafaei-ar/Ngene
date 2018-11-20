import glob 
import numpy as np
from astropy.io import fits
import h5py

class BaseDataProvider(object):
    
    def __init__(self,files,nx,ny,
                     verbose=0,channels=1,
                     thresholds=None,th_labels=None,
                     one_hot=False):
        
        self.a_min = a_min 
        self.a_max = a_max 
        self.verbose = verbose
        self.nx = nx
        self.ny = ny
        self.rgb = rgb       
        self.files = files
        self.channels = channels
        self.thresholds = thresholds
        self.th_labels = th_labels
        if th_labels is not None:
            assert len(thresholds)==len(th_labels)-1,'Incompatibale: len(thresholds)!=len(th_labels)! /n \
            Number of thresholds have to be (threshold labels-1)!'
        if one_hot:
            self.n_class = len(thresholds)+1
        else:
            self.n_class = 1
        self.one_hot = one_hot
        assert len(files) > 0, "No training files"
        if verbose: print("Number of files used: %s"%len(files))

    def read_chunck(self): 
        filename = self.files[self.file_idx]
        data,label = np.load(filename)
        return data,label
           
    def _next_data(self):
        self.file_idx = np.random.choice(len(self.files))
        data, label = self.read_chunck()
        
        assert self.nx<=data.shape[0],'Error, Something is wrong in the given nx. Seems better to decrease it!'
        
        n_try = -1
        if self.ny>0:
            n_try += 1
            ny = data.shape[1]
            while ny < self.ny:
                print('Warning! something is wrong with {} dimensions.'.format(self.files[self.file_idx]))
                self.file_idx = np.random.choice(len(self.files))
                data, label = self.read_chunck()
                ny = data.shape[1]
                assert n_try<1000,'Error, Something is wrong in the given ny. Seems better to decrease it!'
            
        return data, label           

    def pre_process(self, data, label, one_hot):
        
        data = np.clip(np.fabs(data), self.a_min, self.a_max) # clip values that fall outside of interval of min and max
        data -= np.amin(data) # remove the min offset
        data /= np.amax(data) # scale between 0 and 1
        data = np.expand_dims(data, axis=-1) # inserting a new axis at the end of the array shape
        if self.rgb:
            data = to_rgb(data).astype(np.uint8)

        if self.thresholds is None:
            return data,np.expand_dims(label, axis=-1)
            
        labels = threshold_mask(data=data,rfi=label,
                           thresholds=self.thresholds,
                           th_labels=self.th_labels,
                           one_hot=self.one_hot)
        return data,labels
    
    def __call__(self, n): 
        
        data, label = self._next_data()
        data, label = self.pre_process(data,label,self.one_hot)
        nx,ny,nc = data.shape   
        assert nc==self.channels,'Error, problem with given number of channel!'
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        X[0] = data
        Y[0] = label
        for i in range(1,n):
            data, label = self._next_data()
            data, label = self.pre_process(data, label,self.one_hot)
            X[i] = data
            Y[i] = label
    
        return X, Y

def get_slice(data,label,nx,ny):
    """Slice matrix in x and y direction"""
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
   
