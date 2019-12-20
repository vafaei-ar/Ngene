try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

def conv_layer(x,scope,filters=12,kernel_size=5,norm=True,avtive=True):
    
    y = tf.layers.conv2d(x, filters=filters,
                            kernel_size=kernel_size,
                            padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
                            bias_initializer=tf.constant_initializer(0.1),
                            name=scope+'_conv')
    if norm:    y = tf.layers.batch_normalization(y,name=scope+'_norm')
    if avtive:    y = tf.nn.relu(y,name=scope+'_act')
    
    return y

def architecture(x_in,n_layers,n_channel=1,res=3):
    
    layers = [x_in]
    for i in range(n_layers-1):
        layers.append(conv_layer(layers[-1],scope='layer_'+str(i+1)))
        print(layers[-1])
        
        if res:
            if (((i-1)%res==0) & (i>1)):
                n_l = len(layers)
                layers.append(layers[n_l-1]+layers[n_l-res-1])
                print('Res layer',n_l-1,'+',n_l-res-1,':')
                print(layers[-1])
        
    layers.append(conv_layer(layers[-1],scope='layer_'+str(n_layers),filters=n_channel,avtive=0))
    print(layers[-1])

    return layers

