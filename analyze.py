# example of calculating the frechet inception distance in Keras

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

import cv2
import argparse
import matplotlib.pyplot as plt
from keras.utils import generic_utils
import time
try:
	import cPickle as pickle 
except:
	import pickle


parser = argparse.ArgumentParser(description='Analyze Results')
parser.add_argument('--real_path', type=str, default=None, help="Path to folder of real images")
parser.add_argument('--gen_path', type=str, default=None, help="Path to folder of generated images")
parser.add_argument('--num_img', default=-1, type=int, help='Cap number of images')
parser.add_argument('--path', type=str, default=None, help="Path to folders of epochs for analysis")
parser.add_argument('--save_path', type=str, default=None, help="Path to saved results")
parser.add_argument('--save_plot', type=str, default=None, help="Path to FID vs Epoch plot")
parser.add_argument('--interval', type=int, default=300, help="Scanning Interval In Seconds")
parser.add_argument('--scan', action="store_true", default=False, help="Scan for new folders")
args = parser.parse_args()

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
def process_images(images1, images2):
	# convert integer to floating point values
	images1 = images1.astype('float32') / 256
	images2 = images2.astype('float32') / 256

	# resize images
	images1 = scale_images(images1, (299,299,3)) * 256
	images2 = scale_images(images2, (299,299,3)) * 256
	#print('Scaled', images1.shape, images2.shape)

	# pre-process images
	images1 = preprocess_input(images1)
	images2 = preprocess_input(images2)
	return images1, images2		

def get_random_images():
	images1 = randint(0, 255, 10*32*32*3)
	images2 = randint(0, 255, 10*32*32*3)
	images1 = images1.reshape((10,32,32,3))
	images2 = images2.reshape((10,32,32,3))
	print('Prepared', images1.shape, images2.shape)
	return images1, images2

def find_random_fid(model):
	# define two fake collections of images
	images1, images2 = get_random_images()
	images1, images2 = process_images(images1, images2)
	print('FID (same): %.3f' % calculate_fid(model, images1, images1))
	print('FID (different): %.3f' % calculate_fid(model, images1, images2))
	return images1, images2

def get_images(real_path, gen_path, cap=-1):
	images1 = []
	images2 = []
	for img in os.listdir(real_path):
		images1.append(cv2.imread(os.path.join(real_path,img)))
	for img in os.listdir(gen_path):
		images2.append(cv2.imread(os.path.join(gen_path,img)))
	if cap > 0:
		images1 = images1[:cap]
		images2 = images2[:cap]
	return asarray(images1), asarray(images2)
	
def find_fid(model, real_path, gen_path, cap=-1):
	# define two fake collections of images
	images1, images2 = get_images(real_path, gen_path, -1)
	#print('Real Images:', images1.shape, 'Generated Images:',  images2.shape)
	images1, images2 = process_images(images1, images2)
	fid = calculate_fid(model, images1, images2)
	#print('FID (different): %.3f' % fid)
	return fid

def load_fid_data(save_path):
	try:
		infile = open(save_path, "rb") 
		data = pickle.load(infile)
		infile.close()
		return data
	except: 
		print("No Saved FDI Data Found")
		return {}

def save_fid_data(data, save_path):
	outfile = open(save_path, "wb")
	pickle.dump(data, outfile)
	outfile.close()

def find_training_fid(model, path, save_path):

	fid_data = load_fid_data(save_path)
	epochs = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i)) and i.isnumeric() and int(i) not in fid_data]	
	if len(epochs) == 0: return fid_data
	
	progbar = generic_utils.Progbar(len(epochs))#, stateful_metrics=["epoch", "val", "train"])

	for i, epoch in enumerate(epochs):
		fid_train = find_fid(model, os.path.join(path, epoch, "training", "full"), os.path.join(path, epoch, "training", "gen"))
		fid_val = find_fid(model, os.path.join(path, epoch, "validation", "full"), os.path.join(path, epoch, "validation", "gen"))
		
		fid_data[int(epoch)] = [fid_train, fid_val]
		progbar.update(i+1, values=[("epoch", int(epoch)), ("train", fid_train), ("val", fid_val)])
		save_fid_data(fid_data, save_path)

	return fid_data

def scan_training_fid(model, path, save_path, save_plot, interval=300):
	while True:
		try:
			find_training_fid(model, path, save_path)
			plot_fid(save_path, save_plot)
		except Exception as e:
			print("Could Not Save Figure:", type(e).__name__)
		time.sleep(interval)
		
def plot_fid(save_path, save_plot):
	
	fid_data = load_fid_data(save_path)
	epochs = []
	train = []
	val = []

	for epoch in fid_data:
		epochs.append(epoch)
		train.append(fid_data[epoch][0])			
		val.append(fid_data[epoch][1])	
	
	fig = plt.figure()
	ax = plt.subplot(111)

	plt.title("FID Over %d Epochs" % (max(epochs)))
	plt.xlabel("Number of Epochs")
	plt.ylabel("Frechet Inception Distance")

	ax.plot(epochs, train, "bo-", label="Training FID")
	ax.plot(epochs, val, "ro-", label="Validation FID")
	ax.legend()
	plt.savefig(save_plot)
			
# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
#images1, images2 = find_random_fid(model)
if __name__ == "__main__":
	
	if args.real_path and args.gen_path: 
		images1, images2 = find_fid(model, args.real_path, args.gen_path, args.num_img)
	elif args.scan and args.path and args.save_path and args.save_plot:
		scan_training_fid(model, args.path, args.save_path, args.save_plot, args.interval)
	elif args.path and args.save_path and args.save_plot:
		find_training_fid(model, args.path, args.save_path)		
		plot_fid(args.save_path, args.save_plot)	
