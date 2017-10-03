import numpy as np
import tensorflow as tf
import os
import argparse
from PIL import Image
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--img',required=True,type=str,help='path to pic, should be 30x30 and have 3 channels')
args = parser.parse_args()


IMG_FILEPATH = args.img
CHECKPOINT_DIRECTORY = "checkpoints/lr_initial_val=0.003,run=1/"
PREDICTION_LOOKUP_TABLE = ["black circle", "red circle", "yellow circle", "black square", "red square", "yellow square", "black triangle", "red triangle", "yellow triangle"]



if os.path.isfile(IMG_FILEPATH) == False:
	print("Error - " + IMG_FILEPATH + " doesn't exist.")
	sys.exit(1)
_img = Image.open(IMG_FILEPATH)
img = np.asarray(_img)
if img.shape != (30, 30, 3):
	print("Error - " + IMG_FILEPATH + " has shape of " + str(img.shape) + " but it must have a shape of (30, 30, 3)")
	sys.exit(1)
input = np.zeros(shape=(1,30,30,3), dtype=np.uint8)
input[0] = img


	

tf.reset_default_graph()

latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
restorer = tf.train.import_meta_graph(latest_checkpoint + ".meta")
sess = tf.Session()
restorer.restore(sess, latest_checkpoint)

x_tensor = sess.graph.get_tensor_by_name("x:0")
predictions_tensor = sess.graph.get_tensor_by_name("predictions:0")

prediction = sess.run(predictions_tensor, feed_dict={x_tensor:input})[0]

print("\n\nPrediction: " + PREDICTION_LOOKUP_TABLE[prediction])
_img.show()

_img.close()