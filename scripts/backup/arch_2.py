import tensorflow as tf

def architecture(x_in,drop_out):
        
    xp = tf.layers.conv2d(x_in,filters=12,
    kernel_size=5,
    padding='SAME',
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
    bias_initializer=tf.constant_initializer(0.05))
    
    x = tf.layers.batch_normalization(xp)
    x = tf.nn.relu(x)
    x1 = tf.layers.conv2d(x,filters=12,
    kernel_size=5,
    padding='SAME',
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
    bias_initializer=tf.constant_initializer(0.05))
    
    x1 = tf.layers.batch_normalization(x1)
    x1 = tf.nn.relu(x1)
    x2 = tf.layers.conv2d(x1,filters=12,
    kernel_size=5,
    padding='SAME',
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
    bias_initializer=tf.constant_initializer(0.05))
    
    x2 = tf.layers.batch_normalization(x2)
    x2 = tf.nn.relu(x2)

    x3 = x2+xp

    x4 = tf.layers.conv2d(x3,filters=12,
    kernel_size=5,
    padding='SAME',
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
    bias_initializer=tf.constant_initializer(0.05))
    
    x4 = tf.layers.batch_normalization(x4)
    x4 = tf.nn.relu(x4)
    x5 = tf.layers.conv2d(x4,filters=12,
    kernel_size=5,
    padding='SAME',
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05),
    bias_initializer=tf.constant_initializer(0.05))
    
    x5 = tf.layers.batch_normalization(x5)
    x5 = tf.nn.relu(x5)

    x6 = tf.layers.dropout(x5, drop_out)
    x_out = tf.layers.conv2d(x6,filters=1,kernel_size=5,strides=(1, 1),padding='same',
            activation=tf.nn.relu)

    return x_out
