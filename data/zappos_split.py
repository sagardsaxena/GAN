import random
import tqdm
import os
import cv2

tt_split = .9
dapth = "edges2shoes"

svpath = "/vulcan/scratch/nayeem/GAN/data/Zappos_%s"

splts = os.listdir(dapth)
for splt in splts:
    ipth = os.path.join(dapth, splt)
    imgs = os.listdir(ipth)
    for img in tqdm.tqdm(imgs):
        impth = os.path.join(ipth, img)
        im = cv2.imread(impth)
        ch = random.random()
        cv2.imwrite(os.path.join(svpath % ("Pairs_Train_90", "Pairs_Test_10")[ch > tt_split], img), im)
        tim = im[:,:int(im.shape[1])//2]
        tim = cv2.resize(tim, (256,256))
        cv2.imwrite(os.path.join(svpath % ("Edges_Train_90", "Edges_Test_10")[ch > tt_split], img), tim)
        tim = im[:,int(im.shape[1])//2:]
        tim = cv2.resize(tim, (256,256))
        cv2.imwrite(os.path.join(svpath % ("Train_90", "Test_10")[ch > tt_split], img), tim)
