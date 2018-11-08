import tensorflow as tf

def architecture(x_in,drop_out):

	xp = tf.layers.conv2d(x_in,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x1 = tf.layers.conv2d(xp,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	x2 = tf.layers.dropout(x1, drop_out)
	x_out = tf.layers.conv2d(x2,filters=1,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	return x_out
