import tensorflow as tf
import cifar100_input as _input
import cifar100_net as net

EPOCH = 600
BATCH_SIZE = 256
LEARNING_RATE = 0.05
DROPOUT_RATE = 0.5
SUMMARY_DIR = "summary"
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

	tf.summary.image("image", image_batch)
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("accuracy", accuracy)
	summary = tf.summary.merge_all()

	writer  = tf.summary.FileWriter(SUMMARY_DIR, graph)
	saver  = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
		sess.run(trainIter.initializer)
		
		for i in range(EPOCH):
			x, y = sess.run([next_examples, next_labels])
			feed_dict={ image_batch: x,
						label_batch: y,
						dropout_rate: DROPOUT_RATE,
						learning_rate: LEARNING_RATE}
			_, _loss, acc, step = sess.run([train_op, loss, accuracy, global_step], 
											feed_dict=feed_dict)
			print("STEP <{0}>, loss:{1:3.4f}, accuracy:{2:3.2f}%".format(step, _loss, acc*100))

			if(step % 10 == 0):
				writer.add_summary(sess.run(summary, feed_dict=feed_dict), step)
				
			if(step % 10000 == 0):
				saver.save(sess, CHECKPOINT_DIR)