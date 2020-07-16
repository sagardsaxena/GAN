import argparse
import os
import random
import cv2
import sys

parser = argparse.ArgumentParser(description="Copy N Random Samples From Source To Target Directory")
parser.add_argument('-s', '--source', type=str, required=True)
parser.add_argument('-d', '--destination', type=str, required=True)
parser.add_argument('-n', '--sample-size', type=int, default=5)
parser.add_argument('--split', action='store_true', default=False)
parser.add_argument('--nested', action='store_true', default=False)
args = parser.parse_args()

if not args.nested: 
	files = os.listdir(args.source)
else:
	folds = os.listdir(args.source)
	files = []
	for fold in folds:
		if not os.path.isdir(os.path.join(args.source, fold)): continue
		files += [os.path.join(fold, f) for f in os.listdir(os.path.join(args.source, fold))]

if len(files) < args.sample_size: 
	print("Insufficient Number of Files in Source Directory")	
	sys.exit(0)
files = random.sample(files, args.sample_size)

if not os.path.isdir(args.destination):
	os.system("mkdir -p '%s'" % args.destination)


for f in files:
	if not args.split:
		os.system("cp '%s' '%s'" % (os.path.join(args.source, f), os.path.join(args.destination, f.split("/")[-1])))
	else:
		im = cv2.imread(os.path.join(args.source, f))
		f = f.split("/")[-1]
	
		g = f.split(".")[0] + "_1." + f.split(".")[1]
		tim = im[:,:int(im.shape[1])//2]
		cv2.imwrite(os.path.join(args.destination, g), tim)
		
		h = f.split(".")[0] + "_2." + f.split(".")[1]
		tim = im[:,int(im.shape[1])//2:]
		cv2.imwrite(os.path.join(args.destination, h), tim)
