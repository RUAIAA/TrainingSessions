import numpy as np
import tensorflow as tf
import random
import os


train_images = np.load("data/train_images.npy")
train_labels = np.load("data/train_labels.npy")
validation_images = np.load("data/validation_images.npy")
validation_labels = np.load("data/validation_labels.npy")
test_images = np.load("data/test_images.npy")
test_labels = np.load("data/test_labels.npy")
img_shape = np.shape(train_images[0])
label_shape = np.shape(train_labels[0])
batch_info = {
	"unused_indices": list(range(0, len(train_labels))),
	"epochs_completed": 0
}
def resetBatchInfo():
	batch_info["unused_indices"] = list(range(0, len(train_labels)))
	batch_info["epochs_completed"] = 0
def getNextBatch(batch_size):
	images_batch = np.zeros(shape=(batch_size, img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8)
	labels_batch = np.zeros(shape=(batch_size, label_shape[0]), dtype=np.uint8)
	unused_indices = batch_info["unused_indices"]
	if batch_size > len(unused_indices):
		batch_info["unused_indices"] = list(range(0, len(train_labels)))
		unused_indices = batch_info["unused_indices"]
		batch_info["epochs_completed"] += 1
	for i in range(batch_size):
		index = unused_indices.pop(random.randint(0, len(unused_indices)-1))
		images_batch[i, :, :, :] = train_images[index]
		labels_batch[i, :] = train_labels[index]
	return [images_batch, labels_batch]



num_runs_per_model = 1
conv_params = [
		{"output_channels": 12, "patch_size": 5, "stride": 1},
		{"output_channels": 24, "patch_size": 5, "stride": 2},
		{"output_channels": 36, "patch_size": 4, "stride": 2}]
fc_params = [
		{"output_channels": 100},
		{"output_channels": label_shape[0]}]
lr_initial_val_options = [0.003, 0.009, 0.027]
lr_decay_rate = 0.9995
lr_steps_between_decay = 1
num_epochs = 2
batch_size = 100



parent_tboard_log_directory = "tboardlogs/"
parent_checkpoint_directory = "checkpoints/"
if not os.path.exists(parent_tboard_log_directory):
	os.mkdir(parent_tboard_log_directory)
if not os.path.exists(parent_checkpoint_directory):
	os.mkdir(parent_checkpoint_directory)



for lr_initial_val in lr_initial_val_options:
	for run in range(num_runs_per_model):

		model_unique_str = "lr_initial_val=" + str(lr_initial_val) + ",run=" + str(run+1)
		tboard_train_log_directory = parent_tboard_log_directory + model_unique_str + ", track=train/"
		tboard_eval_log_directory = parent_tboard_log_directory + model_unique_str + ", track=eval/"
		checkpoint_directory = parent_checkpoint_directory + model_unique_str + "/"
		
		tf.set_random_seed(0)	
		tf.reset_default_graph()
		train_summaries = []
		eval_summaries = []



		x = tf.placeholder(dtype=tf.float32, shape=[None, img_shape[0], img_shape[1], img_shape[2]], name="x")
		y = tf.placeholder(dtype=tf.float32, shape=[None, label_shape[0]], name="y")

		def conv_layer(input, input_channels, output_channels, patch_size, stride, name):
			with tf.name_scope(name):
				w = tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, input_channels, output_channels], stddev=0.1), name="w")
				b = tf.Variable(tf.constant(0.1, shape=[output_channels]), name="b")
				act = tf.nn.relu(tf.nn.conv2d(input, filter=w, strides=[1,stride,stride,1], padding="SAME") + b, name="act")
				weight_summary = tf.summary.histogram("weights", w)
				bias_summary = tf.summary.histogram("biases", b)
				activation_summary = tf.summary.histogram("activations", act)
				train_summaries.append(weight_summary)
				train_summaries.append(bias_summary)
				train_summaries.append(activation_summary)
				return act
		def fc_layer(input, input_channels, output_channels, use_activation_function, name):
			with tf.name_scope(name):
				w = tf.Variable(tf.truncated_normal(shape=[input_channels, output_channels], stddev=0.1), name="w")
				b = tf.Variable(tf.constant(0.1, shape=[output_channels]), name="b")
				act = tf.matmul(input, w) + b
				if use_activation_function:
					act = tf.nn.relu(act)
				weight_summary = tf.summary.histogram("weights", w)
				bias_summary = tf.summary.histogram("biases", b)
				activation_summary = tf.summary.histogram("activations", act)
				train_summaries.append(weight_summary)
				train_summaries.append(bias_summary)
				train_summaries.append(activation_summary)
				return act
		previous_output_channels = img_shape[2]
		previous_layer=x
		for i in range(len(conv_params)):
			params = conv_params[i]
			input = previous_layer
			input_channels = previous_output_channels
			output_channels = params["output_channels"]
			patch_size = params["patch_size"]
			stride = params["stride"]
			name = "conv"+str(i+1)
			previous_output_channels = output_channels
			previous_layer = conv_layer(input, input_channels, output_channels, patch_size, stride, name)
		previous_output_channels = (previous_layer.shape[1] *previous_layer.shape[2] *previous_layer.shape[3]).value
		previous_layer = tf.reshape(previous_layer, shape=[-1, previous_output_channels])
		for i in range(len(fc_params)):
			params = fc_params[i] 
			input = previous_layer
			input_channels = previous_output_channels
			output_channels = params["output_channels"]
			use_activation_function = i != len(fc_params)-1
			name = "fc"+str(i+1)
			previous_output_channels = output_channels
			previous_layer = fc_layer(input, input_channels, output_channels, use_activation_function, name)
		final_layer = previous_layer
		prediction_probabilities = tf.nn.softmax(final_layer)
		predictions = tf.argmax(prediction_probabilities, axis=1, name="predictions")
			
		with tf.name_scope("loss"):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_layer, labels=y))
			loss_summary = tf.summary.scalar("loss", loss)
			train_summaries.append(loss_summary)
			eval_summaries.append(loss_summary)
		with tf.name_scope("accuracy"):
			is_correct = tf.equal(predictions, tf.argmax(y, axis=1))
			accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
			accuracy_summary = tf.summary.scalar("accuracy", accuracy)
			train_summaries.append(accuracy_summary)
			eval_summaries.append(accuracy_summary)
			
		global_step = tf.Variable(0, name="global_step", trainable=False)
		lr = tf.train.exponential_decay(lr_initial_val, global_step, lr_steps_between_decay, lr_decay_rate, staircase=True)
		lr_summary = tf.summary.scalar("learning_rate", lr)
		train_summaries.append(lr_summary)
		train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)



		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		train_writer = tf.summary.FileWriter(tboard_train_log_directory)
		eval_writer = tf.summary.FileWriter(tboard_eval_log_directory)
		train_writer.add_graph(sess.graph)
		all_train_summaries = tf.summary.merge(train_summaries)
		all_eval_summaries = tf.summary.merge(eval_summaries)



		print("Training following model: " + model_unique_str)
		eval_feed_dict = {x: validation_images, y:validation_labels}
		while batch_info["epochs_completed"] < num_epochs:
			gs = sess.run(global_step)
			batch_x, batch_y = getNextBatch(batch_size)
			train_feed_dict = {x: batch_x, y: batch_y}
			if gs % 100 == 0:
				print("\tepochs_completed: " + str(batch_info["epochs_completed"]) + ", global_step: " + str(gs))
				l, acc, summ = sess.run([loss, accuracy, all_eval_summaries], feed_dict=eval_feed_dict)
				print("\t\tCurrent Eval Loss: " + str(l))
				print("\t\tCurrent Eval Accuracy: " + str(acc))
				eval_writer.add_summary(summ, gs)
			if gs % 10 == 0:
				l, acc, summ = sess.run([loss, accuracy, all_train_summaries], feed_dict=train_feed_dict)
				#print("\t\tCurrent Train Loss: " + str(l))
				#print("\t\tCurrent Train Accuracy: " + str(acc))
				train_writer.add_summary(summ, gs)
			sess.run(train_step, feed_dict=train_feed_dict)
			
		l, acc, summ, gs = sess.run([loss, accuracy, all_eval_summaries, global_step], feed_dict=eval_feed_dict)
		print("\tFinal Eval Loss: " + str(l))
		print("\tFinal Eval Accuracy: " + str(acc))
		eval_writer.add_summary(summ, gs)
			
		resetBatchInfo()
		
		saver.save(sess, checkpoint_directory+"model.ckpt", global_step=gs)
		sess.close()