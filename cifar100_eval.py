import tensorflow as tf
import cifar100_input as _input
import cifar100_net as net

BATCH_SIZE = 64

graph = tf.Graph().as_default()

trainSet  = _input.getTrain(BATCH_SIZE)
trainIter = trainSet.make_initializable_iterator()
next_examples, next_labels = trainIter.get_next()

evalSet  = _input.getEval()
evalIter = evalSet.make_initializable_iterator()
test_examples, test_labels = evalIter.get_next()

image_batch = tf.placeholder(tf.float32, shape=(None ,32, 32, 3))
label_batch = tf.placeholder(tf.int64, shape=(None))
dropout_rate  = tf.placeholder(tf.float32, shape=(None))

logits = net.inference(image_batch, dropout_rate)
accuracy = net.predict(logits, label_batch)

tf.summary.scalar("test_accuracy", accuracy)
summary = tf.summary.merge_all()
saver  = tf.train.Saver()

with tf.Session() as sess:
	#sess.run(tf.global_variables_initializer())
	writer  = tf.summary.FileWriter(".\\summary\\", sess.graph)

	sess.run(evalIter.initializer)
	x, y = sess.run([test_examples, test_labels])

	total_acc = 0
	for i in range(len(x)):
		saver.restore(sess, tf.train.latest_checkpoint(".\\checkpoints\\"))
		feed_dict={image_batch:x[i],
					label_batch:y[i],
					dropout_rate:1}
		acc = sess.run([accuracy], feed_dict=feed_dict)
		total_acc = total_acc + acc
	print(total_acc/len(x))

