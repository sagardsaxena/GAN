import argparse
import os
import cv2
import random

parser = argparse.ArgumentParser(description="Copy N Random Samples From Source To Target Directory")
parser.add_argument('-s', '--source', type=str, required=True)
parser.add_argument('-d', '--destination', type=str, required=True)
parser.add_argument('-n', '--sample-size', type=int, default=5)
parser.add_argument('-l', '--height', type=int, default=20)
parser.add_argument('-w', '--width', type=int, default=50)
parser.add_argument('-r', '--resolution', type=int, default=256)
args = parser.parse_args()

grid = cv2.imread(args.source)
vis = set() 

if not os.path.isdir(args.destination):
	os.system("mkdir -p '%s'" % args.destination)

for i in range(args.sample_size):
	h = int(random.random() * args.height) * args.resolution
	w = int(random.random() * args.width) * args.resolution
	if (h, w) in vis: 
		i-=1
		continue
	cv2.imwrite(os.path.join(args.destination, "%d.jpg" % i), grid[h:h+args.resolution, w:w+args.resolution])
	vis.add((h, w))
	
