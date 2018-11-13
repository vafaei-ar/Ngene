from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from .utils import ch_mkdir,the_print,StopWatch

class Model(object):
    """
    CLASS Model: This class provides you to define, train, restore and operate a sequential convolutional neural network.
    
    --------
    METHODS:
    
    __init__:
    |    Arguments:
    |     nl : input layer size.
    |     learning_rate (default=0.001): learning rate.
    |     n_channel (default=1): number of channels in input layer.
    |     restore (default=False): restore flag.
    |     model_add (default='./model'): saved model address / the address trained model will be saved.
    |     arch_file_name (default=None): name of the architecture file. This file should be designed similar to the sample and located in same directory with the script.
    
    restore:
    This method restores a saved model with specific architecture.
    |    Arguments:
    |     No argument. 
    
    |    Returns:
    |        null
    
    train:
    This method trains CNN.
    |    Arguments:
    |     data_provider: data provider class to feed CNN.
    |        training_epochs (default=1): number of training epochs.
    |        n_s (default=1): number of used image(s) in each epoch.
    |        time_limit (default=None): time limit of training in minutes.
    
    |    Returns:
    |        null
    
    conv:
    This method convolve an image using trained network.
    |    Arguments:
    |        x_in: input image.
    
    | Returns:
    |        2D convolved image.
    
    conve large image:
    This method convolve a large image using trained network.
    |    Arguments:
    |        xsm: input image.
    |        pad (default=10): shared pad between windows.
    |        lw (default=400):    window size.
    
    | Returns:
    |        2D convolved image.
    
    """
    def __init__(self,nx,ny,n_channel=1,n_class=1,restore=False,model_add='./model',arch=None):

        tf.reset_default_graph()
        self.model_add = model_add
        self.x_in = tf.placeholder(tf.float32,[None,nx,ny,n_channel])
        self.y_true = tf.placeholder(tf.float32,[None,nx,ny,n_class])
        self.learning_rate = tf.placeholder(tf.float32)
        self.nx = nx
        self.ny = ny
        self.n_class = n_class
        self.sw = StopWatch()

        if arch is not None:
            if callable(arch):
                self.outputs = arch(self.x_in)
            elif isinstance(arch, str): 
            
                if arch[-3:]=='.py':
                    arch = arch[-3:]
                exec('from '+arch+' import architecture', globals())
                self.outputs = architecture(self.x_in)
                try:
                    os.remove(arch+'.pyc')
                except:
                    pass
                try:
                    shutil.rmtree('__pycache__')
                except:
                    pass
            else:
                assert 0,'Input architecture is not recognized!'

            if type(self.outputs) is list:
                self.x_out = self.outputs[-1]
            else:
                self.x_out = self.outputs
        else:
            self.outputs = self.architecture()
            
        self.r_zeros = tf.nn.zero_fraction(self.x_out)

        self.cost = tf.reduce_sum(tf.pow(self.y_true - self.x_out, 2))
#        self.cost = tf.losses.log_loss(self.y_true,self.x_out)

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
#        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.n_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()

        self.init = tf.global_variables_initializer()
        if restore:
            try:
                self.saver.restore(self.sess, model_add+'/model')
                [self.training_time,self.total_iterations,self.loss,self.metric] = np.load(model_add+'/properties.npy')
            except:
                print('Something is wrong, model can not be restored!')
                exit()
        else:
            self.sess.run(self.init)
            
            # Initializing training properties
            self.training_time = [np.array([0,0,0,0,0])]
            self.total_iterations = [0]
            self.loss = [0]
            self.metric = [0]
            self.properties = np.array([self.training_time,self.total_iterations,self.loss,self.metric],dtype=object)
            ch_mkdir(model_add)
            np.save(model_add+'/properties',self.properties)
                
    def architecture(self):
        xp = tf.layers.conv2d(self.x_in,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)
        x = tf.layers.conv2d(xp,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)
        x = tf.layers.conv2d(x,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)

        x = x+xp
        x = tf.layers.batch_normalization(x)

        x = tf.layers.conv2d(x,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)

        self.x_out = tf.layers.conv2d(x,filters=3,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)
        return self.x_out

    def restore(self):
        tf.reset_default_graph()
        self.saver.restore(self.sess, self.model_add+'/model')

    def train(self, data_provider,training_epochs = 1,iterations=10 ,n_s = 1,
                    learning_rate = 0.001, time_limit=None,
                    metric=None, verbose=0,death_preliminary_check = 30,
                    death_frequency_check = 1000, resuscitation_limit=100000):
                         
        counter = 0
        death = self.total_iterations[-1]==0
        not_dead = 0
        n_resuscitation = 0
        
        if time_limit is not None:
            import time
            t0 = time.time()

        for epoch in range(training_epochs):
                # Loop over all batches
                cc = 0
                ii = 0
                self.sw.reset()
                for i in range(iterations):
                    while True:
                        xb,yb = data_provider(n_s)
                        if xb is not None:
                            break
                    # Run optimization op (backprop) and cost op (to get loss value)
                    if death or (counter%death_frequency_check==0 and death_frequency_check):     
                        rr,_, c = self.sess.run([self.r_zeros,self.optimizer, self.cost],
                                                feed_dict={self.x_in: xb, self.y_true: yb,
                                                self.learning_rate: learning_rate})
                        if rr==1.:
                            self.sess.run(self.init)
                            not_dead = 0
                            death = True
                            if epoch%verbose==0:
                                print('Warning! Dead model! Reinitiating...')
                            n_resuscitation += 1
                        else:
                            not_dead += 1
                        if not_dead>=death_preliminary_check:
                            death = False
                        if n_resuscitation>resuscitation_limit:
                            assert 0,'Unsuccessful resuscitation, check the architecture.'
                            
                    else:
                        _, c = self.sess.run([self.optimizer, self.cost],
                                             feed_dict={self.x_in: xb, self.y_true: yb,
                                             self.learning_rate: learning_rate})
                    
                    cc += c
                    ii += 1                    
                    
                self.training_time.append(self.training_time[-1]+self.sw())
                self.total_iterations.append(iterations+self.total_iterations[-1])
                self.loss.append(cc/ii)
                if metric is None:
                    self.metric.append(0)
                else:
                    self.metric.append(metric())
                
                # Display loss per epoch step
                if verbose:
                    if epoch%verbose==0:
                        print('Epoch:{:d}, cost= {:f}'.format(epoch, cc/ii))
                if time_limit is not None:
                    t1 = time.time()
                    if (t1-t0)/60>time_limit:
                        the_print("Time's up, goodbye!",tc='red',bgc='green')
                        ch_mkdir(self.model_add)
                        self.saver.save(self.sess, self.model_add+'/model')
                        self.properties = np.array([self.training_time,self.total_iterations,self.loss,self.metric],dtype=object)
                        np.save(self.model_add+'/properties',self.properties)
                        return 0

        # Creates a saver.
        ch_mkdir(self.model_add)
        self.saver.save(self.sess, self.model_add+'/model')
        self.properties = np.array([self.training_time,self.total_iterations,self.loss,self.metric],dtype=object)
        np.save(self.model_add+'/properties',self.properties)

    def conv(self,x_in):
        x_out = self.sess.run(self.x_out, feed_dict={self.x_in: x_in})
        return x_out[0,:,:,:]

    def conv_large_image(self,xsm,pad=10,lx=276,ly=400):
    
        if self.n_class==1:
            prm = np.zeros((xsm.shape[1],xsm.shape[2]))
            li = xsm.shape[1]
            lj = xsm.shape[2]
            for i in np.arange(0,li,lx-2*pad):
                if i+lx<li:
                    iii = i
                    iei = i+lx
                else:
                    iii = li-lx
                    iei = li

                for j in np.arange(0,lj,ly-2*pad):
                    if j+ly<lj:
                      jii = j
                      jei = j+ly
                    else:
                      jii = lj-ly
                      jei = lj

                    conx = self.conv(xsm[:,iii:iei,jii:jei,:])
                    prm[iii+pad:iei-pad,jii+pad:jei-pad] = conx[pad:-pad,pad:-pad,0]
        else:      
            prm = np.zeros((xsm.shape[1],xsm.shape[2],self.n_class))
            li = xsm.shape[1]
            lj = xsm.shape[2]
            for i in np.arange(0,li,lx-2*pad):
                if i+lx<li:
                    iii = i
                    iei = i+lx
                else:
                    iii = li-lx
                    iei = li

                for j in np.arange(0,lj,ly-2*pad):
                    if j+ly<lj:
                      jii = j
                      jei = j+ly
                    else:
                      jii = lj-ly
                      jei = lj

                    conx = self.conv(xsm[:,iii:iei,jii:jei,:])
                    prm[iii+pad:iei-pad,jii+pad:jei-pad,:] = conx[pad:-pad,pad:-pad,:]
        return prm

    def get_filters(self):
        filts = [str(i.name).split('/')[0] for i in tf.trainable_variables() if 'kernel' in i.name]    
        weights = []
        for filt in filts:
            with tf.variable_scope(filt, reuse=True) as scope_conv:
                W_conv = tf.get_variable('kernel')
                weights.append(W_conv.eval())
                
        return weights
                
                
