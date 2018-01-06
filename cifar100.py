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

global_step = tf.Variable(0, trainable=False, name="global_step")
image_batch = tf.placeholder(tf.float32, shape=(None ,32, 32, 3))
label_batch = tf.placeholder(tf.int64, shape=(None))
dropout_rate  = tf.placeholder(tf.float32, shape=(None))
learning_rate = tf.placeholder(tf.float32, shape=())

logits = net.inference(image_batch, dropout_rate)
loss   = net.cost(logits, label_batch)
accuracy = net.predict(logits, label_batch)
train_op = net.train(loss, learning_rate, global_step)

tf.summary.image("image", image_batch)
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
summary = tf.summary.merge_all()
saver  = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(trainIter.initializer)
	writer  = tf.summary.FileWriter(".\\summary\\", sess.graph)

	epoch = 100000
	for i in range(epoch):
		x, y = sess.run([next_examples, next_labels])
		feed_dict={image_batch:x,
					label_batch:y,
					dropout_rate:0.5,
					learning_rate:0.05}
		_, _loss, acc, step = sess.run([train_op, loss, accuracy, global_step], feed_dict=feed_dict)
		print("STEP <{0}>, loss:{1:3.4f}, accuracy:{2:3.2f}%".format(step, _loss, acc*100))

		if(step % 10 == 0):
			writer.add_summary(sess.run(summary, feed_dict=feed_dict), step)
		if(step % 1000 == 0):
			saver.save(sess, ".\\checkpoints\\")