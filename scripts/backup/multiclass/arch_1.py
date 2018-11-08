import tensorflow as tf

def architecture(x_in,drop_out):

	xp = tf.layers.conv2d(x_in,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x1 = tf.layers.conv2d(xp,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x2 = tf.layers.conv2d(x1,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x3 = x2+xp
	x3 = tf.layers.batch_normalization(x3)
	x4 = tf.layers.conv2d(x3,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	x5 = tf.layers.dropout(x4, drop_out)
	x_out = tf.layers.conv2d(x5,filters=1,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	return [x_out,x5,x4,x3,x2,x1,xp]
