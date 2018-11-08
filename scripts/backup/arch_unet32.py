import numpy as np
from collections import OrderedDict
import tensorflow as tf

def conv2d(x,num_outputs,kernel_size,stride,padding='VALID',activation_fn=tf.nn.relu,trainable=True,scope=None):
    return tf.contrib.layers.conv2d(
        x,
        num_outputs=num_outputs,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        data_format='NHWC',
        rate=1,
        activation_fn=activation_fn,
        normalizer_fn=None,
        normalizer_params=None,
        weights_regularizer=None,
        biases_initializer=tf.constant_initializer(0.1),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=trainable,
        scope=scope
    )

def deconv2d(x,num_outputs,kernel_size,stride,padding='VALID',activation_fn=tf.nn.relu,trainable=True,scope=None):
    return tf.contrib.layers.conv2d_transpose(
        x,
        num_outputs=num_outputs,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        data_format='NHWC',
        activation_fn=activation_fn,
        normalizer_fn=None,
        normalizer_params=None,
        weights_regularizer=None,
        biases_initializer=tf.constant_initializer(0.1),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=trainable,
        scope=scope
    )

def architecture(x_in,drop_out_rate):

#    x = tf.placeholder(dtype=tf.float32, shape=(20,l,l,5))
    x = x_in
    l = x.get_shape().as_list()[1]
    np2 = int(np.log(l)/np.log(2))
    main = 2**np2
    n_residue = l-main
    res = []
    nt = 1
    while n_residue>np.sum(res)-len(res):
        res.append(nt)
        nt += 8
    res = res[1:-1]
    res.append(n_residue-np.sum(res)+len(res)+1)
        
    print('===================================================')
    print('===================== ENCODER =====================')
    print('===================================================')
    print(x)
    layers = OrderedDict()
    nr = len(res)
    i_layer = 0
    n_layers = 0
    for i in range(nr):
        i_layer += 1
        ks = res[nr-i-1]
        x = conv2d(x,6,[ks,ks],[1,1],padding='VALID',scope='encode/deconv_'+str(i_layer))
        x = tf.layers.batch_normalization(x,name='encode/batch_'+str(i_layer))
        x = tf.layers.dropout(x,rate=drop_out_rate,training=True,name='encode/dropout_'+str(i_layer))
        x = tf.nn.relu(x,name='encode/relu_'+str(i_layer))
        layers['encode_'+str(i_layer)] = x
        print(x)
        
    for i in range(np2-5):
        i_layer += 1
        x = conv2d(x,6,[5,5],[2,2],padding='SAME',scope='encode/deconv_'+str(i_layer))
        x = tf.layers.batch_normalization(x,name='encode/batch_'+str(i_layer))
        x = tf.layers.dropout(x,rate=drop_out_rate,training=True,name='encode/dropout_'+str(i_layer))
        x = tf.nn.relu(x,name='encode/relu_'+str(i_layer))
        layers['encode_'+str(i_layer)] = x
        print(x)
    n_layers = i_layer
        
    print('')
    print('===================================================')
    print('===================== DECODER =====================')
    print('===================================================')

    i_layer = 0
    for i in range(np2-5):
        i_layer += 1
        x = deconv2d(x,6,[5,5],[2,2],padding='SAME',scope='decode/deconv_'+str(i_layer))
        x = tf.layers.batch_normalization(x,name='decode/batch_'+str(i_layer))
        x = tf.layers.dropout(x,rate=drop_out_rate,training=True,name='decode/dropout_'+str(i_layer))
        x = tf.nn.relu(x,name='decode/relu_'+str(i_layer))
        layers['decode_'+str(i_layer)] = x
        print(x)

    for i in range(nr):
        i_layer += 1
        ks = res[i]
        x = deconv2d(x,6,[ks,ks],[1,1],padding='VALID',scope='decode/deconv_'+str(i_layer))
        x = tf.layers.batch_normalization(x,name='decode/batch_'+str(i_layer))
        x = tf.layers.dropout(x,rate=drop_out_rate,training=True,name='decode/dropout_'+str(i_layer))
        x = tf.nn.relu(x,name='decode/relu_'+str(i_layer))
        layers['decode_'+str(i_layer)] = x
        print(x)
    
    return [x,layers]
