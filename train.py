import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from os import walk
from tensorflow import keras
from numpy import expand_dims
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from keras import Input
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import InceptionV3
from keras.losses import CategoricalCrossentropy
from keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, ReLU, Softmax, Activation
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])
LABELS = [0,1,2,3,4,5,6,7,8]
CLASSES = ["Coast", "Forest", "Highway", "Kitchen", "Mountain", "Office", "Store", "Street", "Suburb"]
SIZE = [259, 447, 607, 717, 991, 1106, 1321, 1513, 1655]

def get_folder(folder, end):
	_, _, filenames = next(walk(folder))
	file = []
	for f in filenames:
		file.append(folder + "/" + f)
	return file

def readImageData():
	coast_test = get_folder("images/Coast/test", ".jpg")
	coast_train = get_folder("images/Coast/train", ".jpg")
	forest_test = get_folder("images/Forest/test", ".jpg")
	forest_train = get_folder("images/Forest/train", ".jpg")
	highway_test = get_folder("images/Highway/test", ".jpg")
	highway_train = get_folder("images/Highway/train", ".jpg")
	kitchen_test = get_folder("images/Kitchen/test", ".jpg")
	kitchen_train = get_folder("images/Kitchen/train", ".jpg")
	mountain_test = get_folder("images/Mountain/test", ".jpg")
	mountain_train = get_folder("images/Mountain/train", ".jpg")
	office_test = get_folder("images/Office/test", ".jpg")
	office_train = get_folder("images/Office/train", ".jpg")
	store_test = get_folder("images/Store/test", ".jpg")
	store_train = get_folder("images/Store/train", ".jpg")
	street_test = get_folder("images/Street/test", ".jpg")
	street_train = get_folder("images/Street/train", ".jpg")
	suburb_test = get_folder("images/Suburb/test", ".jpg")
	suburb_train = get_folder("images/Suburb/train", ".jpg")
	test_dict = {
			'Coast': coast_test,
			'Forest': forest_test,
			'Highway': highway_test,
			'Kitchen': kitchen_test,
			'Mountain': mountain_test,
			'Office': office_test,
			'Store': store_test,
			'Street': street_test,
			'Suburb': suburb_test,
	}
	train_dict = {
			'Coast': coast_train,
			'Forest': forest_train,
			'Highway': highway_train,
			'Kitchen': kitchen_train,
			'Mountain': mountain_train,
			'Office': office_train,
			'Store': store_train,
			'Street': street_train,
			'Suburb': suburb_train
	}
	return test_dict, train_dict

def getGrayImages(dict):
	l = []
	for key in dict.keys():
		for pic in dict[key]:
			dsize = (224, 224)
			image = img_to_array(load_img(pic, target_size=dsize, color_mode="grayscale"))
			l.append(image)
	X = np.asarray(l).reshape(len(l), 224, 224, 1)
	return np.array(X)

def getExtraGrayImages(dict):
	l = []
	y = []
	answers = {
		'Coast': 0,
		'Forest': 1,
		'Highway': 2,
		'Kitchen': 3,
		'Mountain': 4,
		'Office': 5,
		'Store': 6,
		'Street': 7,
		'Suburb': 8,
	}
	for key in dict.keys():
		for pic in dict[key]:
			dsize = (224, 224)
			image = img_to_array(load_img(pic, target_size=dsize, color_mode="grayscale"))
			image = tf.expand_dims(image, 0)
			y.append(answers[key])
			l.append(image)
			for i in range(9):
				augmented_image = data_augmentation(image)
				l.append(augmented_image)
				y.append(answers[key])
	X = np.asarray(l).reshape(len(l), 224, 224, 1)
	Y = tf.keras.utils.to_categorical(y)
	Y = np.asarray(Y).reshape(len(Y), 9)
	return np.array(X), Y

def getColorImages(dict):
	l = []
	for key in dict.keys():
		for pic in dict[key]:
			dsize = (224, 224)
			image = img_to_array(load_img(pic, target_size=dsize, color_mode="rgb"))
			l.append(image)
	X = np.asarray(l).reshape(len(l), 224, 224, 3)
	return np.array(X)

def getExtraColorImages(dict):
	l = []
	y = []
	answers = {
		'Coast': 0,
		'Forest': 1,
		'Highway': 2,
		'Kitchen': 3,
		'Mountain': 4,
		'Office': 5,
		'Store': 6,
		'Street': 7,
		'Suburb': 8,
	}
	for key in dict.keys():
		for pic in dict[key]:
			dsize = (224, 224)
			image = img_to_array(load_img(pic, target_size=dsize, color_mode="rgb"))
			image = tf.expand_dims(image, 0)
			y.append(answers[key])
			l.append(image)
			for i in range(9):
				augmented_image = data_augmentation(image)
				l.append(augmented_image)
				y.append(answers[key])
	X = np.asarray(l).reshape(len(l), 224, 224, 3)
	Y = tf.keras.utils.to_categorical(y)
	Y = np.asarray(Y).reshape(len(Y), 9)
	return np.array(X), Y

def getLable(dict):
	answers = {
		'Coast': 0,
		'Forest': 1,
		'Highway': 2,
		'Kitchen': 3,
		'Mountain': 4,
		'Office': 5,
		'Store': 6,
		'Street': 7,
		'Suburb': 8,
	}
	l = []
	for key in dict.keys():
		for pic in dict[key]:
			l.append(answers[key])
	Y = tf.keras.utils.to_categorical(l)
	Y = np.asarray(Y).reshape(len(Y), 9)
	return Y

def decode(data):
	collect = []
	for i in range(len(data)):
		best = 0
		index = 0
		for j in range(len(data[0])):
			if data[i][j] > best:
				best = data[i][j]
				index = j
		collect.append(index)
	collect = np.asarray(collect).reshape(len(data))
	return collect.transpose()

def problem2b():
	test_image_dict, train_image_dict = readImageData()
	test_images = getGrayImages(test_image_dict)
	train_images = getGrayImages(train_image_dict)
	test_label = getLable(test_image_dict)
	train_label = getLable(train_image_dict)

	model = keras.models.load_model("models/problem2b")
	prediction = model.predict(test_images)
	col_pred = decode(prediction)
	col_true = decode(test_label)
	for i in range(len(col_pred)):
		if (col_pred[i] != col_true[i]):
			print(str(i + 1) + " -> " + CLASSES[col_true[i]] + " : " + CLASSES[col_pred[i]] )
	# cm = confusion_matrix(col_true, col_pred, labels=LABELS, normalize='true')
	# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
	# disp.plot(cmap='plasma')
	# plt.show()


	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=3, input_shape=(224,224,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Conv2D(filters=32, kernel_size=3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Conv2D(filters=32, kernel_size=3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Flatten())
	model.add(Dense(units=9))
	model.add(Softmax())
	opt = SGD(learning_rate=0.001, momentum=0.9)
	t1 = TopKCategoricalAccuracy(k=1, name="top 1", dtype=None)
	t3 = TopKCategoricalAccuracy(k=3, name="top 3", dtype=None)
	model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=[t1, t3])
	model.summary()
	model.fit(train_images, train_label, batch_size=64, epochs=20, validation_data=(test_images, test_label), shuffle=True)
	prediction = model.predict(test_images)
	# col_pred = decode(prediction)
	# col_true = decode(test_label)
	# cm = confusion_matrix(col_true, col_pred, labels=LABELS, normalize='true')
	# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
	# disp.plot(cmap='plasma')
	# plt.show()
	model.save("models/problem2b")

def problem2c():
	test_image_dict, train_image_dict = readImageData()
	train_images, train_label = getExtraGrayImages(train_image_dict)
	test_label = getLable(test_image_dict)
	test_images = getGrayImages(test_image_dict)

	model = keras.models.load_model("models/problem2c")
	prediction = model.predict(test_images)
	col_pred = decode(prediction)
	col_true = decode(test_label)
	for i in range(len(col_pred)):
		if (col_pred[i] != col_true[i]):
			print(str(i + 1) + " -> " + CLASSES[col_true[i]] + " : " + CLASSES[col_pred[i]] )

	# # (900, 224, 224, 1)
	# print(train_images.shape)
	# # (1695, 224, 224, 1)
	# print(test_images.shape)
	#
	# # (900, 9)
	# print(train_label.shape)
	# # (1695, 9)
	# print(test_label.shape)

	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=3, input_shape=(224,224,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Conv2D(filters=32, kernel_size=3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Dropout(0.5, seed=69))
	model.add(Conv2D(filters=32, kernel_size=3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Flatten())
	model.add(Dense(units=9))
	model.add(Softmax())
	opt = SGD(learning_rate=0.001, momentum=0.9)
	t1 = TopKCategoricalAccuracy(k=1, name="top 1", dtype=None)
	t3 = TopKCategoricalAccuracy(k=3, name="top 3", dtype=None)
	model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=[t1, t3])
	model.summary()
	model.fit(train_images, train_label, batch_size=64, epochs=20, validation_data=(test_images, test_label), shuffle=True)
	prediction = model.predict(test_images)
	# col_pred = decode(prediction)
	# col_true = decode(test_label)
	# cm = confusion_matrix(col_true, col_pred, labels=LABELS, normalize='true')
	# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
	# disp.plot(cmap='plasma')
	# plt.show()
	model.save("models/problem2c")

def basic_transfer_learning_model():
	"""
	The `basic_transfer_learning_model()` function is responsible for training a 
	Transfer Learning model using InceptionV3 architecture for image classification. 
	It follows these steps:	
    
	1. Load image data for testing and training using the `readImageData` function.
	2. Preprocess color images and labels for testing and training using the `getColorImages` 
	   and `getLable` functions, respectively.
	3. Define a new input layer with shape (224, 224, 3) for color images.
	4. Load the InceptionV3 model with pre-trained weights from 'imagenet' and exclude the 
	   top classification layer.
	5. Set the InceptionV3 layers as non-trainable to use pre-trained features.
	6. Add a global max-pooling layer, flatten the output, and connect it to a dense layer 
	   with 9 output units (representing image classes) and softmax activation.
	7. Create a new model using the specified input and the modified InceptionV3 architecture.
	8. Compile the model using Stochastic Gradient Descent (SGD) optimizer, Categorical 
	   Crossentropy loss, and top-1 and top-3 categorical accuracy metrics.
	9. Display a summary of the model architecture.
	10. Train the model on the preprocessed training data for 20 epochs with a batch size of 64.
	11. Save the trained model in the 'models' directory with the name 'problem3a'.
    
	Note: This function leverages transfer learning to benefit from InceptionV3's pre-trained 
	features while fine-tuning the classification layer for the specific image classes.
	"""
 
	# Load image data and preprocess
	test_image_dict, train_image_dict = readImageData()
	test_images = getColorImages(test_image_dict)
	train_images = getColorImages(train_image_dict)
	test_label = getLable(test_image_dict)
	train_label = getLable(train_image_dict)

	# Define new input layer
	new_input = Input(shape=(224,224,3))
 
	# Load InceptionV3 with pre-trained weights
	model = InceptionV3(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling=max)
	model.trainable = False
 
	# Apply InceptionV3 to the new input
	param = model(new_input, training=False)
	param = MaxPool2D(pool_size=3)(param)
	param = Flatten()(param)
 
	# Connect to a dense layer with softmax activation
	output = Dense(9, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(param)
	
	# Create a new model with input and output
	model = Model(inputs=new_input, outputs=output)
	
	# Compile the model
	t1 = TopKCategoricalAccuracy(k=1, name="top 1", dtype=None)
	t3 = TopKCategoricalAccuracy(k=3, name="top 3", dtype=None)
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=[t1, t3])
	
	# Display model summary
	model.summary()
	
	# Train the model
	model.fit(train_images, train_label, batch_size=64, epochs=20, validation_data=(test_images, test_label), shuffle=True)
	
	# Save the trained model
	model.save("models/basic_transfer_learning_model")

def problem3b():
	test_image_dict, train_image_dict = readImageData()
	train_images, train_label = getExtraColorImages(train_image_dict)
	# test_label = getLable(test_image_dict)
	# train_label = getLable(train_image_dict)

	new_input = Input(shape=(224,224,3))
	model = InceptionV3(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling=max)
	model.trainable = False
	param = model(new_input, training=False)
	param = MaxPool2D(pool_size=3)(param)
	param = Flatten()(param)
	output = Dense(9, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(param)
	model = Model(inputs=new_input, outputs=output)
	t1 = TopKCategoricalAccuracy(k=1, name="top 1", dtype=None)
	t3 = TopKCategoricalAccuracy(k=3, name="top 3", dtype=None)
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=[t1, t3])
	model.summary()
	model.fit(train_images, train_label, batch_size=64, epochs=20, validation_data=(test_images, test_label), shuffle=True)
	model.save("models/problem3b")




def basic_CNN():
	"""
	The `basic_CNN()` function is responsible for training a Convolutional Neural Network (CNN)
	model for image classification. It follows these steps:

	1. Load image data for testing and training using the `readImageData` function.
	2. Preprocess grayscale images and labels for testing and training using the `getGrayImages` 
	   and `getLabel` functions, respectively.
	3. Build a CNN model with three convolutional layers, batch normalization, ReLU activation, 
	   and max-pooling. The model ends with a fully connected layer with 9 output units 
	   representing the image classes and a softmax activation.
	4. Compile the model using Stochastic Gradient Descent (SGD) optimizer, Categorical Crossentropy 
	   loss, and top-1 and top-3 categorical accuracy metrics.
	5. Display a summary of the model architecture.
	6. Train the model on the preprocessed training data for 20 epochs with a batch size of 64.
	7. Save the trained model in the 'models' directory with the name 'basic_CNN'.

	Note: This function serves as a baseline model, and you can modify it or experiment with 
	different architectures based on specific project requirements.
	"""

    # Load image data for testing and training
	test_image_dict, train_image_dict = readImageData()

	# Preprocess images and labels for testing
	test_images = getGrayImages(test_image_dict)
	train_images = getGrayImages(train_image_dict)
 
 	# Preprocess images and labels for training
	test_label = getLable(test_image_dict)
	train_label = getLable(train_image_dict)

	# Build the CNN model
	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=3, input_shape=(224,224,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Conv2D(filters=32, kernel_size=3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Conv2D(filters=32, kernel_size=3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=3))
	model.add(Flatten())
	model.add(Dense(units=9))
	model.add(Softmax())
 
	# Compile the model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	t1 = TopKCategoricalAccuracy(k=1, name="top 1", dtype=None)
	t3 = TopKCategoricalAccuracy(k=3, name="top 3", dtype=None)
	model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=[t1, t3])
	
	# Display model summary
	model.summary()
	
	# Train the model
	model.fit(train_images, train_label, batch_size=64, epochs=20, validation_data=(test_images, test_label), shuffle=True)
	
 	# Save the trained model
	model.save("models/basic_CNN")


if __name__ == "__main__":
	# basic_CNN()
	# problem2b()
	# problem2c()
	basic_transfer_learning_model()
	# problem3b()
