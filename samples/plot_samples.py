import argparse
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import sys
import os

parser = argparse.ArgumentParser(description="Plot Sample Images")
parser.add_argument('-s', '--source', type=str, required=True)
parser.add_argument('-n', '--sample-images', type=int, required=True)
parser.add_argument('-p', '--save-plot', type=str, required=True)
parser.add_argument('-r', '--resolution', type=int, default=256)
parser.add_argument('-o', '--order', nargs='+', type=str, default=None)
parser.add_argument('-t', '--title', default="")
args = parser.parse_args()

folders = sorted(os.listdir(args.source))
if args.order != None: folders = args.order
if len(folders) < 1: 
	print("Insufficient Data")
	sys.exit(1)

fig = plt.figure(figsize=(int(2*args.sample_images), 2*len(folders)))
grid = ImageGrid(fig, 111, nrows_ncols=(len(folders), args.sample_images), axes_pad=0.1)

i = 0
for folder in folders:
	imgs = sorted(os.listdir(os.path.join(args.source, folder)))
	if len(imgs) < args.sample_images: 
		print("Insufficient Number of Images")
		sys.exit(1)
	
	imgs = imgs[:args.sample_images]
	for img in imgs:
		im = cv2.imread(os.path.join(args.source, folder, img))
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		im = cv2.resize(im, (args.resolution,args.resolution))

		grid[i].imshow(im)
		i+=1

	grid[i-args.sample_images].set_ylabel(folder.replace("_", "\n"), size="large", multialignment='center')

fig.suptitle(args.title)
plt.savefig(args.save_plot)
