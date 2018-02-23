import tensorflow as tf
import cifar100_input as _input
import cifar100_net as net

CHECKPOINT_DIR = "checkpoints"

with tf.Graph().as_default() as graph:

	evalSet  = _input.getEval()
	evalIter = evalSet.make_initializable_iterator()
	test_examples, test_labels = evalIter.get_next()

	global_step = tf.Variable(0, trainable=False, name="global_step")
	image_batch = tf.placeholder(tf.float32, shape=(None ,32, 32, 3))
	label_batch = tf.placeholder(tf.int64, shape=(None))
	dropout_rate  = tf.placeholder(tf.float32, shape=(None))
	learning_rate = tf.placeholder(tf.float32, shape=(None))

	logits = net.inference(image_batch, dropout_rate)
	loss   = net.cost(logits, label_batch)
	accuracy = net.predict(logits, label_batch)
	train_op = net.train(loss, learning_rate, global_step)

	saver  = tf.train.Saver()

	with tf.Session() as sess:
		
		saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
		sess.run(evalIter.initializer)

		x, y = sess.run([test_examples, test_labels])

		While(True):
			
			feed_dict={image_batch: x,
						label_batch: y,
						dropout_rate: 1}
			step, acc = sess.run([step, accuracy], feed_dict=feed_dict)
				
			print("STEP <{0}>, test accuracy:{2:3.2f}%".format(step, acc*100))
