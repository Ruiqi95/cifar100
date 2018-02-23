import tensorflow as tf

def print_activations(t):
	print(t.op.name, ' ', t.get_shape().as_list())

def inference(images, dropout_rate):
	with tf.name_scope("conv_1") as scope:
		kernels = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[32]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		conv   = tf.nn.conv2d(images, kernels, strides=[1,1,1,1], padding='SAME')
		conv_1 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_1)

	with tf.name_scope("max_2") as scope:
		max_2 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		print_activations(max_2)

	with tf.name_scope("conv_3") as scope:
		kernels = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[64]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		conv   = tf.nn.conv2d(max_2, kernels, strides=[1,1,1,1], padding='SAME')
		conv_3 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_3)

	with tf.name_scope("max_4") as scope:
		max_4 = tf.nn.max_pool(conv_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		print_activations(max_4)

	with tf.name_scope("conv_5") as scope:
		kernels = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[128]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		conv   = tf.nn.conv2d(max_4, kernels, strides=[1,1,1,1], padding='SAME')
		conv_5 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_5)

	with tf.name_scope("conv_6") as scope:
		kernels = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[128]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		conv   = tf.nn.conv2d(conv_5, kernels, strides=[1,1,1,1], padding='SAME')
		conv_6 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_6)

	with tf.name_scope("conv_7") as scope:
		kernels = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[256]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		conv   = tf.nn.conv2d(conv_6, kernels, strides=[1,1,1,1], padding='SAME')
		conv_7 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_6)

	with tf.name_scope("fc_8") as scope:
		reshape = tf.reshape(conv_7, [-1, 16384])
		kernels = tf.Variable(tf.truncated_normal([16384, 1024], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[1024]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		fc_8 = tf.nn.relu(tf.matmul(reshape, kernels) + biases)
		fc_8_drop = tf.nn.dropout(fc_8, dropout_rate)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(fc_8)

	with tf.name_scope("fc_9") as scope:
		kernels = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[1024]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		fc_9 = tf.nn.relu(tf.matmul(fc_8, kernels) + biases)
		fc_9_drop = tf.nn.dropout(fc_9, dropout_rate)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(fc_9)

	with tf.name_scope("readout") as scope:
		kernels = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[1024]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		readout = tf.nn.relu(tf.matmul(fc_9, kernels) + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(readout)

	return readout

def cost(logits, labels):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=labels, logits=logits, name="cross_entropy_per_example")
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def predict(logits, labels):
	correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy

def predictTop5(logits, labels):
	top_prediction = tf.nn.in_top_k(logits, labels, k=5)
	accuracy = tf.reduce_mean(tf.cast(top_prediction, tf.float32))
	return accuracy
	
def train(loss, lr, global_step):
	lr_decay = tf.train.exponential_decay(lr, global_step, 10000, 0.9, staircase=True)
	optimizer = tf.train.MomentumOptimizer(lr_decay, momentum=0.9)
	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
	return train_op