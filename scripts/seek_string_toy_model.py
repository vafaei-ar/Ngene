import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pylab as plt
import ngene as ng
from ngene.architectures.simple import architecture 
import ccgpack as ccg
from tqdm import tqdm, trange
from scipy.stats import ttest_ind

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

cl = np.load('../data/cl_planck_lensed.npy')
sfs = ccg.StochasticFieldSimulator(cl)
size = 7.2

class DataProvider(object):
    
    def __init__(self,nside,size,alpha,num,n_buffer=200,reinit=1000):
        
        self.nside = nside
        self.alpha = alpha
        self.num = num
        self.size = size
        self.n_buffer = n_buffer
        self.reinit = reinit
        self.couter = 0
        
    def simulate(self):
        
        s = np.zeros((self.nside, self.nside), dtype=np.double)
        begins = ccg.random_inside(s,num=self.num)
        ends = ccg.random_inside(s,num=self.num)

        g = sfs.simulate(self.nside,self.size)
        g -= g.min()
        g /= g.max()    
        s = ccg.draw_line(s,begins=begins,ends=ends,value=1)

        return g,s
    
    def simulation_initiation(self):
        gs = []
        ss = []    
#         for i in tqdm(range(self.n_buffer), total=self.n_buffer, unit=" map", desc='Initiation', ncols=70):
        for i in range(self.n_buffer):
            g,s = self.simulate()
            gs.append(g)
            ss.append(s)
        return np.array(gs),np.array(ss)
        
    def __call__(self,n):
        
        if self.couter%self.reinit==0:
            self.gs, self.ss = self.simulation_initiation()
        self.couter += 1
        x_out = []
        y_out = []
        for i in range(n):
            i_g,i_s = np.random.randint(0,self.n_buffer,2)
            x_out.append(self.gs[i_g]+self.alpha*self.ss[i_s])
            y_out.append(self.ss[i_s])
            
        x_out = np.array(x_out)
        y_out = np.array(y_out)
        return np.expand_dims(x_out,-1),np.expand_dims(y_out,-1)

def check(name):
    l0 = []
    l1 = []
    for i in range(100):
        x,y = dp(1)
        x0,y = dp0(1)
        l0.append(model.conv(x0).std())
        l1.append(model.conv(x).std())
    b0,h0 = ccg.pdf(l0,20)
    b1,h1 = ccg.pdf(l1,20)
    plt.plot(b0,h0)
    plt.plot(b1,h1)
    plt.savefig('pdf_'+name+'.jpg')
    plt.close()
    
    fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,nrows=1,figsize=(15,7))

    x,y = dp(1)
    x_pred = model.conv(x)
    ax1.imshow(x[0,:,:,0])
    ax1.set_title('Input')
    ax2.imshow(y[0,:,:,0])
    ax2.set_title('Output')
    ax3.imshow(x_pred[:,:,0])
    ax3.set_title('Prediction')
    plt.savefig('sample_'+name+'.jpg')
    plt.close()
        
    print(name+'p-value:',ttest_ind(l0,l1)[1])
    return ttest_ind(l0,l1)[1]

nside=400
dp = DataProvider(nside=nside,size=7,alpha=0.7,num=50)
dp0 = DataProvider(nside=nside,size=7,alpha=0,num=50,n_buffer=100)
x,y = dp0(1)
x,y = dp(1)
#fig, (ax1,ax2)= plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
#ax1.imshow(x[0,:,:,0])
#ax1.axis('off')
#ax2.imshow(y[0,:,:,0])
#ax2.axis('off')


def arch(x_in):
    x_out = architecture(x_in=x_in,n_layers=5,res=2)
    return x_out

model = ng.Model(nx=nside,ny=nside,n_channel=1,n_class=1,
         restore=0,model_add='./model/'+str(0)+'_'+str(dp.alpha),arch=arch)

print('# of variables:',model.n_variables)

alphas = []

dalpha = 0.05
p_move = 0

for i in range(50):

    model.train(data_provider=dp,training_epochs = 10,iterations=50 ,n_s = 10,
                        learning_rate = 0.01, time_limit=None,
                        metric=None, verbose=1,death_preliminary_check = 30,
                        death_frequency_check = 1000)
    pv = check()
    print(pv)
    if pv<1e-7:
        if p_move == 1:
            dalpha = dalpha/2
        dp.alpha = dp.alpha-dalpha
        p_move = -1
    else:
        if p_move == -1:
            dalpha = dalpha/2
        dp.alpha = dp.alpha+dalpha
        p_move = 1
        
    alphas.append(dp.alpha)
    print(dp.alpha)
        
    model.model_add='./model/'+str(i+1)+'_'+str(dp.alpha)

