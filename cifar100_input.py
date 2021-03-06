import tensorflow as tf

RANDOM_SEED = 10
TRAIN_DIR = "./train"
TEST_DIR  = "./test"

def readTrain():
	import pickle
	with open(TRAIN_DIR, "rb") as f:
		dataDict = pickle.load(f, encoding="bytes")
	return dataDict[b'fine_labels'], dataDict[b'data']

def readEval():
	import pickle
	with open(TEST_DIR, "rb") as f:
		dataDict = pickle.load(f, encoding="bytes")
	return dataDict[b'fine_labels'], dataDict[b'data']

def _parse_function_train(image_string, label):
  image_decoded = tf.reshape(image_string, [32, 32, 3])
  image_float   = tf.cast(image_decoded, tf.float32)
  image_normal  = tf.image.per_image_standardization(image_float) 
  image_flipped = tf.image.random_flip_left_right(image_normal, seed=RANDOM_SEED) 
  return image_flipped, label

def _parse_function_eval(image_string, label):
  image_decoded = tf.reshape(image_string, [32, 32, 3])
  image_float   = tf.cast(image_decoded, tf.float32)
  image_normal  = tf.image.per_image_standardization(image_float) 
  return image_normal, label

def getTrain(batch_size):
	labels, images = readTrain()
	dataset = tf.data.Dataset.from_tensor_slices((images, labels))
	dataset = dataset.map(_parse_function_train)
	dataset = dataset.shuffle(buffer_size=1024)
	dataset = dataset.batch(batch_size)
	dataset = dataset.repeat()
	return dataset

def getEval():
	labels, images = readEval()
	dataset = tf.data.Dataset.from_tensor_slices((images, labels))
	dataset = dataset.map(_parse_function_eval)
	dataset = dataset.batch(len(labels))
	return dataset