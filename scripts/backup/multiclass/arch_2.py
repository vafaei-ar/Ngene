import tensorflow as tf

def architecture(x_in,drop_out):

    xp = tf.layers.conv2d(x_in,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    xp = tf.layers.batch_normalization(xp)
    xp = tf.nn.relu(xp)
    x1 = tf.layers.conv2d(xp,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x1 = tf.layers.batch_normalization(x1)
    x1 = tf.nn.relu(x1)
    x2 = tf.layers.conv2d(x1,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x2 = tf.layers.batch_normalization(x2)
    x2 = tf.nn.relu(x2)

    x3 = x2+xp

    x4 = tf.layers.conv2d(x3,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x4 = tf.layers.batch_normalization(x4)
    x4 = tf.nn.relu(x4)
    x5 = tf.layers.conv2d(x4,filters=12,kernel_size=5,strides=(1, 1),padding='same')
    x5 = tf.layers.batch_normalization(x5)
    x5 = tf.nn.relu(x5)

    x6 = tf.layers.dropout(x5, drop_out)
    x_out = tf.layers.conv2d(x6,filters=1,kernel_size=5,strides=(1, 1),padding='same',
            activation=tf.nn.relu)

    return [x_out,x6,x5,x4,x3,x2,x1,xp]
