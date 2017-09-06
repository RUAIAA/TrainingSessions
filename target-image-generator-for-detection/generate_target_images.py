'''
How to run this script: Run on command line with python. Optional command line arguments: --out, --num_pics. To see help message, use -h command.
Requirements: python3 (tested on python3, but python2 may also work), pillow
What this script does:
	- Generates a given number of images with targets, per class (that number is passed in as a command line arg).
	- The images generated are heavily randomized.
	- Alongside this script, there are 4 important directories: Shapes/, Colors/, Fonts/, Backgrounds/. This script iterates through the files in the Shapes and Colors directories and chooses random files from the Fonts and Backgrounds directories.
'''
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import json
import pdb
from multiprocessing import Process
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_pics_per_class',default=1,type=int,help='num images to generate per class')
parser.add_argument('--image_size',default=450,type=int,help='Length and height dimension of output images')
parser.add_argument('--continue_bool', default=False, type=bool, help='If specified as True, will attempt to continue previous job from where it left off')
args = parser.parse_args()

# Number of images to generate per class in a single execution of this script.
num_images_per_class_to_generate = args.num_pics_per_class
# Dimension of outputted image (both length and height)
image_size = args.image_size
# bool value that controls whether or not this program should attempt to continue the previous job from where it left off.
continue_from_saved_state = args.continue_bool




# Parent directory where this script will save generated images and json annotation files
parent_save_dir = "generated/"
# Directory where this script will save images
pic_save_dir = parent_save_dir + "images/"
# Directory where this script will save json labels for the images
label_save_dir = parent_save_dir + "labels/"
# Save annotations as described by Dylan in this file
annotations_filepath = parent_save_dir + "annotations.json"
# Script will save state information in this file and if the continue flag is specified as true, will look at this file to determine where the previous run left off
state_filepath = parent_save_dir + ".state.json"


# If the following directories don't exist, create them
if not os.path.exists(parent_save_dir):
	os.mkdir(parent_save_dir)
if not os.path.exists(pic_save_dir):
	os.mkdir(pic_save_dir)
if not os.path.exists(label_save_dir):
	os.mkdir(label_save_dir)



'''
HYPERPARAMETERS FOR ITERATED PARAMETERS DEFINED HERE
This script generates a constant number of images per class. In order to do that, it iterates sequentially iterates through some parameters like shape and alphanumeric.
The hyperparameters constraining the choices/ranges/directory-paths for these iterated parameters are defined here. Feel free to edit the hyperparameters.
Make sure you read the comments before editing any hyperparameters, otherwise you might break something
'''
# This script will iterate through the subdirectories in this directory (each subdirectory represents a shape class) and generate images for each. According to the AUVSI specification, these are the valid shapes: circle, semi_circle, quarter_circle, triangle, square, rectangle, trapezoid, pentagon, hexagon, heptagon, octagon, star, cross. This directory already has all the shapes necessary, so you should never need to add your own shapes here. If you want to add your own shapes, however, contact Ethan. The shape files here need to be in a very specific format that is too complicated to type here
shape_dir = "Shapes/"
# This script will iterate through the alphanumerics defined here and generate images for each one. 
#alphanumeric_choices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't' ,'T', 'u', 'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z']
alphanumeric_choices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't' ,'T', 'u', 'U', 'v', 'V', 'x', 'X', 'y', 'Y', 'z', 'Z']

'''
HYPERPARAMETERS FOR RANDOMIZED PARAMETERS DEFINED HERE
Every time this script generates an image, it randomizes several parameters, such as rotation angle, size, font, etc.
The hyperparameters constraining the choices/ranges/directory-paths for these parameters are defined here. Feel free to edit the hyperparameters.
Make sure you read the comments before editing any hyperparameters, otherwise you might break something
'''
# This script will choose two random colors from this directory every time it generates an image. One color is for the alphanumeric and the other is for the shape. According to the AUVSI specification, these are the valid colors: white, black, gray, red, blue, green, yellow, purple, brown, orange. If you want to add your own color file, make sure you follow the pre-existing naming convention and file content format.
color_dir = "Colors/"
# This script will choose a random font from this directory every time it generates an image. Feel free to add your own fonts.
font_dir = "Fonts/"
# This script will choose a random font size every time it generates an image. 75 seems to work as a minimum. The maximum size depends on the alphanumeric, font, and shape, so it cannot be specified here
font_size_minmax = [70, None]
# This script will choose 2 random offsets from center for the alphanumeric every time it generates an image.
alphanumeric_offset_from_center_range = range(-2, 3)
# This script will choose a random rotation angle between 0 and 360 degrees every time it generates an image. The rotation is executed to the target container after drawing the alphanumeric onto it
rotation_angle_range = range(0, 360)
# This script will choose a target container size every time it generates an image. It resizes the container after drawing the alphanumeric on it (if I resize the container before drawing the alphanumeric on it, I run the risk of making the container smaller than the alphanumeric). Note that the target container size does not equal the target size.
target_size_range = range(20, 40)
# This script will choose 3 random means and stddevs for the gaussian distributions that it samples noise from. It uses a different distribution for each channel (r g and b). Higher stddev values result in more noise. Be careful with the mean value - if it's too high or low, colors can accidentally be converted into other colors
noise_distribution_mean_range = range(-10, 11)
noise_distribution_stddev_range = range(20, 40)
# This script will choose a random background from this directory and crop it to image_size every time it generates an image. These images are mostly of grass, but it couldn't hurt to throw in some pavement too. Feel free to add your own.
background_dir = "Backgrounds/"
# This script will choose two random vertex positions (left and top) every time it generates an image. When it crops the background to image_size, it will use these values as 2/4 of the crop vertices. The other 2 vertices (right/bottom) are just image_size + the left/top vertices. Note that the range for the left/top values cannot be defined here, because they depends on the size of the full background image
background_crop_vertex_left_range = None
background_crop_vertex_top_range = None
# This script will choose a random rotation angle for the background every time it generates an image. It only rotates a multiple of 90 degrees so that the background size does not change
background_rotation_angle_choices = [0, 90, 180, 270]
# This script will choose a random brightness multipler for the background every time it generates an image. 
background_brightness_multiplier_range = np.arange(0.5, 1, 0.02)
# This script will choose two random offsets (x and y) every time it generates an image. It will draw the target onto the background at these offsets. Note that the max offset cannot be specified here, because it depends on the runtime size of the target, which is not known here.
target_x_offset_minmax = [0, None]
target_y_offset_minmax = [0, None]




# Randomly chooses all the randomized parameters (except x_offset and y_offset, which cannot be chosen here). Don't choose the iterated parameters.
def choose_random_parameters(shape_class_dir):
	# Choose a random shape file from the shape_class_dir directory
	shape_filename = random.choice(os.listdir(shape_dir + shape_class_dir + "/"))
	# Choose a random shape color class and shade file
	shape_color_class_dir = random.choice(os.listdir(color_dir))
	shape_color_filename = random.choice(os.listdir(color_dir + shape_color_class_dir + "/"))
	# Choose a random alphanumeric color class and shade file. Make sure the alphanumeric color class does not match the shape color class (otherwise the alphanumeric will not be visible)
	alphanumeric_color_class_dir = random.choice(os.listdir(color_dir))
	while alphanumeric_color_class_dir == shape_color_class_dir:
		alphanumeric_color_class_dir = random.choice(os.listdir(color_dir))
	alphanumeric_color_filename = random.choice(os.listdir(color_dir + alphanumeric_color_class_dir + "/"))
	# Choose a random font file
	font_filename = random.choice(os.listdir(font_dir))
	# Cannot choose font size here.
	font_size = None
	# Choose a random rotation angle. The rotation is executed to the target container after drawing the alphanumeric onto it
	rotation_angle = random.choice(rotation_angle_range)
	# Choose random x and y offsets from center for the alphanumeric
	alphanumeric_offset_from_center_x = random.choice(alphanumeric_offset_from_center_range)
	alphanumeric_offset_from_center_y = random.choice(alphanumeric_offset_from_center_range)
	# Choose a random size. Will later resize the target to this size after drawing the alphanumeric onto it. Note that the target container size does not equal the target size.
	target_size = random.choice(target_size_range)
	# Choose random mean and stddev values for the noise
	noise_distribution_mean_1 = random.choice(noise_distribution_mean_range)
	noise_distribution_stddev_1 = random.choice(noise_distribution_stddev_range)
	noise_distribution_mean_2 = random.choice(noise_distribution_mean_range)
	noise_distribution_stddev_2 = random.choice(noise_distribution_stddev_range)
	noise_distribution_mean_3 = random.choice(noise_distribution_mean_range)
	noise_distribution_stddev_3 = random.choice(noise_distribution_stddev_range)
	# Choose a random background
	background_filename = random.choice(os.listdir(background_dir))
	# Can't choose the left/top crop vertices here. Need to know the size of the full background image
	background_crop_vertex_left = None
	background_crop_vertex_top = None
	# Choose a random background rotation angle
	background_rotation_angle = random.choice(background_rotation_angle_choices)
	# Choose a random brightness multiplier for the background
	background_brightness_multiplier = random.choice(background_brightness_multiplier_range)
	# Can't choose the x_offset and y_offset here.
	target_x_offset = None
	target_y_offset = None
	# Return a dictionary object containing all the randomized params
	randomized_params = {"shape_filename": shape_filename,
						"shape_color_class_dir": shape_color_class_dir,
						"shape_color_filename": shape_color_filename, 
						"alphanumeric_color_class_dir": alphanumeric_color_class_dir,
						"alphanumeric_color_filename": alphanumeric_color_filename, 
						"font_filename": font_filename, 
						"font_size": font_size, 
						"alphanumeric_offset_from_center_x": alphanumeric_offset_from_center_x,
						"alphanumeric_offset_from_center_y": alphanumeric_offset_from_center_y,
						"rotation_angle": rotation_angle, 
						"target_size": target_size, 
						"noise_distribution_mean_1": noise_distribution_mean_1, 
						"noise_distribution_stddev_1": noise_distribution_stddev_1, 
						"noise_distribution_mean_2": noise_distribution_mean_2, 
						"noise_distribution_stddev_2": noise_distribution_stddev_2, 
						"noise_distribution_mean_3": noise_distribution_mean_3, 
						"noise_distribution_stddev_3": noise_distribution_stddev_3, 
						"background_filename": background_filename, 
						"background_crop_vertex_left": background_crop_vertex_left,
						"background_crop_vertex_top": background_crop_vertex_top,
						"background_rotation_angle": background_rotation_angle, 
						"background_brightness_multiplier": background_brightness_multiplier, 
						"target_x_offset": target_x_offset, 
						"target_y_offset": target_y_offset}
	return randomized_params
	

	
	
# Generates a random image and saves it in save_dir
def generate_image(img_savepath, label_savepath, shape_class_dir, alphanumeric, randomized_params):

	'''
	Step 1: Unpack the randomized params
	'''
	shape_filename = randomized_params["shape_filename"]
	shape_color_class_dir = randomized_params["shape_color_class_dir"] 
	shape_color_filename = randomized_params["shape_color_filename"]
	alphanumeric_color_class_dir = randomized_params["alphanumeric_color_class_dir"] 	
	alphanumeric_color_filename = randomized_params["alphanumeric_color_filename"] 
	font_filename = randomized_params["font_filename"] 
	font_size = randomized_params["font_size"]
	alphanumeric_offset_from_center_x = randomized_params["alphanumeric_offset_from_center_x"]
	alphanumeric_offset_from_center_y = randomized_params["alphanumeric_offset_from_center_y"]
	rotation_angle = randomized_params["rotation_angle"]
	target_size = randomized_params["target_size"]
	noise_distribution_mean_1 = randomized_params["noise_distribution_mean_1"]
	noise_distribution_stddev_1 = randomized_params["noise_distribution_stddev_1"]
	noise_distribution_mean_2 = randomized_params["noise_distribution_mean_2"]
	noise_distribution_stddev_2 = randomized_params["noise_distribution_stddev_2"]
	noise_distribution_mean_3 = randomized_params["noise_distribution_mean_3"]
	noise_distribution_stddev_3 = randomized_params["noise_distribution_stddev_3"]
	background_filename = randomized_params["background_filename"]
	background_crop_vertex_left = randomized_params["background_crop_vertex_left"]
	background_crop_vertex_top = randomized_params["background_crop_vertex_top"]
	background_rotation_angle = randomized_params["background_rotation_angle"]
	background_brightness_multiplier = randomized_params["background_brightness_multiplier"]
	target_x_offset = randomized_params["target_x_offset"]
	target_y_offset = randomized_params["target_y_offset"]
	

	'''
	Step 2: Execute all the pre-alphanumeric-draw operations on the target container
		1) Open the shape image
		2) Color it
	'''
	# Create the target container. To start off, set it to the shape image.
	target_container = Image.open(shape_dir+shape_class_dir+"/"+shape_filename)
	# Open the shape color file and extract the rgb values from it
	with open(color_dir+shape_color_class_dir+"/"+shape_color_filename, 'r') as shape_color_file:
		shape_color = json.load(shape_color_file)
	# Convert shape_color from rgb to rgba (a=255 which means fully opaque)
	shape_color.append(255)
	# Convert shape_color from a list to a tuple
	shape_color = tuple(shape_color)
	# Color the shape. Specify a mask so that only the actual shape, not the rest of the target container, gets filled with the color (note: this only works if the shape image's alpha channel is set correctly)
	target_container.paste(shape_color, (0,0,target_container.size[0],target_container.size[1]), mask=target_container)

	
	'''
	Step 3: Draw the alphanumeric on the target container and execute the post-alphanumeric-draw operations:
		1) Calculate the maximum allowable font size and choose a font size
			a) Predict the aspect ratio of the alphanumeric
			b) Find the largest rectangle (with an spect ratio equal to the aspect ratio of the alphanumeric) that fits inside the shape
			c) Find the largest font size that allows the alphanumeric to fit inside that rectangle
			d) Choose a fontsize
		2) Predict the pixel-size of the alphanumeric
		3) Use the predicted pixel-size to determine where the alphanumeric should be drawn so that it is centered on the target container. By centering the alphanumeric on the target container, it will also be centered on the shape.
		4) Draw the alphanumeric
		5) Rotate the target container
	'''
	'''
	# Now that we know which font and alphanumeric we are using, and drew the shape, we can calculate the maximum allowable font size and choose one. Only do this if font_size is None (will usually be the case, unless if user is debugging)
	if font_size == None and font_size_minmax[1] == None:
		# Predict the aspect ratio of the alphanumeric, also normalize it so that the lower value in the ratio is 2. Doesn't account for whitespace in the alphanumeric.
		alphanumeric_testsize = np.array(ImageFont.truetype(font_dir+font_filename, 100).getsize(alphanumeric))
		alphanumeric_aspect_ratio = alphanumeric_testsize / min(alphanumeric_testsize) * 2
		# Find the largest rectangle with the correct aspect ratio that fits inside the shape. Will save the dimensions in this rectangle in the variable alphanumeric_maxsize. Note that I subtract 5 from the rectangle's width and height for some extra padding. Also note that the algorithm used to find the rectangle is expensive and could be implemented better (but I don't feel like it)
		alphanumeric_maxsize = None
		target_container_center_x = int(target_container.width/2)
		target_container_center_y = int(target_container.height/2)
		target_container_alpha_channel_raw = np.asarray(target_container.split()[-1])
		i=1
		while True:
			alphanumeric_maxsize_candidate = alphanumeric_aspect_ratio * i
			left = target_container_center_x - int((alphanumeric_maxsize_candidate[0]+0.5) / 2)
			right = target_container_center_x + int((alphanumeric_maxsize_candidate[0]+1.5) / 2)
			top = target_container_center_y - int((alphanumeric_maxsize_candidate[1]+0.5) / 2)
			bottom = target_container_center_y + int((alphanumeric_maxsize_candidate[1]+1.5) / 2)
			if np.any(target_container_alpha_channel_raw[top:bottom+1,left:right+1] != 255):
				alphanumeric_maxsize = (right-left-7, bottom-top-7)
				break
			i += 1
		# Now that we know the dimensions of the largest rectangle that fits inside the shape, we test different fontsizes until we find the largest one that doesn't cause the alphanumeric to be bigger than the rectangle. The algorithm doesn't account for whitespace in the alphanumeric.
		font_size_max = None
		font_size_max_candidate = font_size_minmax[0] + 1
		while True:
			if ImageFont.truetype(font_dir+font_filename,font_size_max_candidate).getsize(alphanumeric)[0] >= alphanumeric_maxsize[0]:
				font_size_max = font_size_max_candidate
				break
			font_size_max_candidate += 1
		# Choose a font size (since we now know the minumum and maximum allowable font sizes)
		font_size = random.choice(range(font_size_minmax[0], font_size_max))
	elif font_size == None and font_size_minmax[1] != None:
		font_size = random.choice(range(font_size_minmax[0], font_size_minmax[1]))
	# Load the font
	font = ImageFont.truetype(font_dir+font_filename, font_size)
	# Predict the width and height of the alphanumeric in pixels. Note that there seems to be a bug in pillow: the font.getsize method doesn't account for whitespace in the return value
	alphanumeric_width_including_whitespace, alphanumeric_height_including_whitespace = font.getsize(alphanumeric)
	alphanumeric_whitespace_x, alphanumeric_whitespace_y = font.getoffset(alphanumeric)
	alphanumeric_width = alphanumeric_width_including_whitespace - alphanumeric_whitespace_x
	alphanumeric_height = alphanumeric_height_including_whitespace - alphanumeric_whitespace_y
	# Calculate the position to draw the alphanumeric so that it is centered
	draw_position_x = (target_container.width / 2) - alphanumeric_whitespace_x - (alphanumeric_width / 2) + alphanumeric_offset_from_center_x
	draw_position_y = (target_container.height / 2) - alphanumeric_whitespace_y - (alphanumeric_height / 2) + alphanumeric_offset_from_center_y
	# Create a ImageDraw.Draw instance which we will use to draw the alphanumeric onto the target container
	drawer = ImageDraw.Draw(target_container)
	# Open the alphanumeric color file and extract the rgb values from it
	with open(color_dir+alphanumeric_color_class_dir+"/"+alphanumeric_color_filename, 'r') as alphanumeric_color_file:
		alphanumeric_color_rgb = json.load(alphanumeric_color_file)
	# Draw the alphanumeric on the target container at the correct xy position so that it is centered. Specify the fill color and the font as well.
	drawer.text((draw_position_x, draw_position_y), alphanumeric, fill=tuple(alphanumeric_color_rgb), font=font)
	# Rotate the target container, allow it to expand so that nothing gets cut off.
	target_container = target_container.rotate(rotation_angle, expand=True)
	'''
	
	'''
	Step 4: Determine the target's dimensions and position relative to the target container (note: this is pretty stupid but I couldn't think of anything better) and use that information to choose the x and y offsets
		1) Search through the target container for the leftmost, topmost, rightmost, and bottommost pixels that are part of the target in order to determine the x-origin, y-origin, length, and height (this is pretty stupid that I have to do this)
		2) Crop out the target
		3) Resize the target
		4) Choose the x and y offsets (we can do this now because we know the final size of the target)
	'''
	target_container_alpha_channel = target_container.split()[-1]
	target_container_length = target_container.size[0]
	target_container_height = target_container.size[1]
	# Search through the target container to determine the x-origin of the target relative to the target container
	target_x_origin_relative_to_container = -1
	search_complete = False
	for i in range(target_container_length):
		if search_complete:
			break
		for j in range(target_container_height):
			if target_container_alpha_channel.getpixel((i, j)) != 0:
				target_x_origin_relative_to_container = i
				search_complete = True
				break
	# Search through the target container to determine the y-origin of the target relative to the target container
	target_y_origin_relative_to_container = -1
	search_complete = False
	for i in range(target_container_height):
		if search_complete:
			break
		for j in range(target_container_length):
			if target_container_alpha_channel.getpixel((j, i)) != 0:
				target_y_origin_relative_to_container = i
				search_complete = True
				break
	# Search through the target container to find rightmost target pixel and calculate the length of the target
	target_length_before_resize = -1
	search_complete = False
	for i in range(target_container_length):
		if search_complete:
			break
		for j in range(target_container_height):
			if target_container_alpha_channel.getpixel((target_container_length-1-i, j)) != 0:
				target_length_before_resize = target_container_length - target_x_origin_relative_to_container - i
				search_complete = True
				break
	# Search through the target container to find bottommost target pixel and calculate the height of the target
	target_height_before_resize = -1
	search_complete = False
	for i in range(target_container_height):
		if search_complete:
			break
		for j in range(target_container_length):
			if target_container_alpha_channel.getpixel((j, target_container_height-1-i)) != 0:
				target_height_before_resize = target_container_height - target_y_origin_relative_to_container - i
				search_complete = True
				break
	# Crop out the target
	target = target_container.crop((target_x_origin_relative_to_container, target_y_origin_relative_to_container, target_x_origin_relative_to_container+target_length_before_resize, target_y_origin_relative_to_container+target_height_before_resize))
	# Compute the post-resize target dimensions and resize
	target_length = int(target_length_before_resize * target_size / max(target_length_before_resize, target_height_before_resize))
	target_height = int(target_height_before_resize * target_size / max(target_length_before_resize, target_height_before_resize))
	target = target.resize((target_length, target_height))
	# Now that we know the target dimensions, we can specify the maximum allowed x and y offsets
	target_x_offset_minmax[1] = image_size - (target_length)
	target_y_offset_minmax[1] = image_size - (target_height)
	# Choose x and y offsets (only if the values are currently None. This may not be the case if the user is debugging)
	if target_x_offset == None:
		target_x_offset = random.choice(range(target_x_offset_minmax[0], target_x_offset_minmax[1]))
	if target_y_offset == None:
		target_y_offset = random.choice(range(target_y_offset_minmax[0], target_y_offset_minmax[1]))


	'''
	Step 5: Add gaussian noise to the target and blur
		1) Convert target to a raw numpy array
		2) Generate a noise array
		3) Add the noise array to the raw target array
		4) Convert back to pillow image
		5) Blur the image
	'''
	# Convert target to a np array. Note that the x and y axes are flipped when you do this
	target_raw = np.asarray(target)
	# Create a gaussian noise array and round all the elements to integers. Use different means and stddevs for each channel. The 4th channel should not have any noise
	noise = np.zeros([target_height, target_length, 4])
	noise[:,:,0] = np.random.normal(noise_distribution_mean_1, noise_distribution_stddev_1, target_height*target_length).reshape(target_height,target_length).round()
	noise[:,:,1] = np.random.normal(noise_distribution_mean_2, noise_distribution_stddev_2, target_height*target_length).reshape(target_height,target_length).round()
	noise[:,:,2] = np.random.normal(noise_distribution_mean_3, noise_distribution_stddev_3, target_height*target_length).reshape(target_height,target_length).round()
	# Add the noise to target_raw (needs to be converted to np.uint8 so that it can be converted back to a pillow image)
	target_raw = (target_raw + noise).clip(0,255).astype(np.uint8)
	# Convert back to pillow image
	target = Image.fromarray(target_raw)
	# Blur the target
	target = target.filter(ImageFilter.BLUR)


	'''
	Step 6: Open the background, execute some operations on it, and paste the target container onto the background
		1) Open the background image
		2) Crop the background to image_size
		3) Rotate the background
		4) Apply a brightness diminisher to the background
		5) Pate the shape onto the background at the correct offset
	'''
	# Open the background image
	background = Image.open(background_dir+background_filename)
	# Now that we know the size of the full background image, we can choose left/top crop vertices. Only do this if the vertex values are not None (will always be the case, unless if the user is debugging)
	if background_crop_vertex_left == None:
		background_crop_vertex_left_range = range(0, background.width-image_size)
		background_crop_vertex_left = random.choice(background_crop_vertex_left_range)
	if background_crop_vertex_top == None:
		background_crop_vertex_top_range = range(0, background.height-image_size)
		background_crop_vertex_top = random.choice(background_crop_vertex_top_range)
	# Crop the background to image_size
	background = background.crop((background_crop_vertex_left, background_crop_vertex_top, background_crop_vertex_left+image_size, background_crop_vertex_top+image_size))
	# Rotate the background
	background = background.rotate(background_rotation_angle)
	# Diminish the brightness of the background
	background = ImageEnhance.Brightness(background).enhance(background_brightness_multiplier)
	# Paste the target onto the background at the correct x,y offset. Also specify the mask so that the transparent areas don't get pasted
	background.paste(target, (target_x_offset, target_y_offset), mask=target)
	image = background
	


	'''Step 7: Save the image and close all the open handles'''
	# Convert the image to RGB just in case it's currently RGBA
	image = image.convert("RGB")
	# Save the image
	image.save(img_savepath)
	# Close the open Image handles
	target_container.close()
	target.close()
	background.close()
	image.close()
	
	
	
	'''Step 8: Write to the labels file'''
	labels = {
		"shape": shape_class_dir,
		"alphanumeric": alphanumeric,
		"shape_color": shape_color_class_dir,
		"alphanumeric_color": alphanumeric_color_class_dir,
		"target_x_origin": target_x_offset,
		"target_y_origin": target_y_offset,
		"target_length": target_length,
		"target_height": target_height,
		"rotation_angle": rotation_angle,
		"shape_filename": shape_filename, 
		"shape_color_filename": shape_color_filename,
		"alphanumeric_color_filename": alphanumeric_color_filename, 
		#"font_filename": font_filename, 
		#"font_size": font_size, 
		"alphanumeric_offset_from_center_x": alphanumeric_offset_from_center_x,
		"alphanumeric_offset_from_center_y": alphanumeric_offset_from_center_y,
		"target_size": target_size, 
		"noise_distribution_mean_1": noise_distribution_mean_1, 
		"noise_distribution_stddev_1": noise_distribution_stddev_1, 
		"noise_distribution_mean_2": noise_distribution_mean_2, 
		"noise_distribution_stddev_2": noise_distribution_stddev_2, 
		"noise_distribution_mean_3": noise_distribution_mean_3, 
		"noise_distribution_stddev_3": noise_distribution_stddev_3, 
		"background_filename": background_filename, 
		"background_crop_vertex_left": background_crop_vertex_left,
		"background_crop_vertex_top": background_crop_vertex_top,
		"background_rotation_angle": background_rotation_angle, 
		"background_brightness_multiplier": background_brightness_multiplier, 
	}
	json.dump(labels, open(label_savepath, 'w'), indent=0)


if __name__ == '__main__':

	# Get current time in seconds since epoch, will use this to generate filenames
	timestamp = str(int(time.time()))

	# Get array containing shape filenames
	shape_class_dirs = os.listdir(shape_dir)
	# Calculate the number of classes
	num_classes = len(alphanumeric_choices) * len(shape_class_dirs)
	# Calculate the total number of images to generate
	num_images_to_generate = num_images_per_class_to_generate * num_classes
	
	# Write to the annotations file
	with open(annotations_filepath, 'w') as annotations_file:
		color_choices = os.listdir(color_dir)
		annotations	= "{"
		annotations += "\nShape: " + str({shape_class_dirs[i]: i for i in range(len(shape_class_dirs))}) + ","
		annotations += "\nAlphanumeric: " + str({alphanumeric_choices[i]: i for i in range(len(alphanumeric_choices))}) + ","
		annotations +=	"\nBackground Color: " + str({color_choices[i]: i for i in range(len(color_choices))}) + ","
		annotations +=	"\nAlphanumeric Color: " + str({color_choices[i]: i for i in range(len(color_choices))})
		annotations += "\n}"
		annotations_file.write(annotations)
	
	# Variables used for storing and reloading from the state file
	num_images_skipped = 0
	num_images_to_skip = 0
	if continue_from_saved_state == True:
		saved_state = json.load(open(state_filepath, 'r'))
		num_images_to_skip = saved_state["generated"]
		num_images_to_generate = saved_state["total"]
		num_images_per_class_to_generate = int(num_images_to_generate / num_classes)
		timestamp = saved_state["timestamp"]
	
	# Print some useful stuff for the user
	print("\nGenerating " + str(num_images_per_class_to_generate) + " images per class. Number of classes: " + str(num_classes) + ". Total images being generated: " + str(num_images_to_generate))
	if continue_from_saved_state == True:
		print("\nContinuing from where previous run left off, skipping " + str(num_images_to_skip) + " images\n")
	
	
	# Keep track of the number of images generated
	count = 0
	# Iterate throught the classes
	for shape_class_dir in shape_class_dirs:
		for alphanumeric in alphanumeric_choices:
			for i in range(num_images_per_class_to_generate):
				# Skip images until we are up to where we left off in the previous run (images are only skipped if the user specifies the continue flag as true)
				if num_images_skipped != num_images_to_skip:
					num_images_skipped += 1
					count += 1
					continue
				print("Generating image #" + str(count+1) + "/" + str(num_images_to_generate))
				# Choose all the random parameters
				randomized_params = choose_random_parameters(shape_class_dir)
				# Generate the save filepath for the image and json file using the system time and count
				img_savepath = pic_save_dir + timestamp + "-" + str(count+1) + ".jpg"
				label_savepath = label_save_dir + timestamp + "-" + str(count+1) + ".json"
				# Generate the image. Pass in all the savepaths, iterated parameters, and randomized parameters
				generate_image(img_savepath, label_savepath, shape_class_dir, alphanumeric, randomized_params)
				# Dump the progress information to the state_filename
				state = {"total": num_images_to_generate, "generated": count+1, "timestamp": timestamp}
				json.dump(state, open(state_filepath, 'w'))
				count += 1