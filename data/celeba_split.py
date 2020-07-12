import random
from tqdm import tqdm
import os
import cv2

tt_split = .9

def stylegan2():
	dapth = "img_align_celeba"
	svpth = '/vulcan/scratch/nayeem/GAN/data/CelebA_%s'
	imgs = os.listdir(dapth)
	print(len(imgs))

	for imp  in tqdm(imgs):
		if imp[-3:] != "jpg": continue
		impth = os.path.join(dapth, imp)
		im = cv2.imread(impth)
		im = cv2.resize(im, (256,256))
		cv2.imwrite(os.path.join(svpth % ("Train_90", "Test_10")[random.random() > tt_split], imp), im)

#stylegan2()
