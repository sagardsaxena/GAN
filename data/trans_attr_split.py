import random
import tqdm
import os
import cv2

tt_split = .9
dapth = "night2day"

svpath = "/vulcan/scratch/nayeem/GAN/data/Trans_Attr_%s"

train = set()
test = set()

splts = os.listdir(dapth)
for splt in splts:
    ipth = os.path.join(dapth, splt)
    imgs = os.listdir(ipth)
    for img in tqdm.tqdm(imgs):
        impth = os.path.join(ipth, img)
        im = cv2.imread(impth)

        lid = img.split("_")[0]
        aid = img.split("_")[1] + ".jpg"
        bid = img.split("_")[3]
        ch = 1 if lid in test else 0 if lid in train else random.random()
        (train, test)[ch > tt_split].add(lid)

        cv2.imwrite(os.path.join(svpath % ("Pairs_Train_90", "Pairs_Test_10")[ch > tt_split], img), im)
        tim = im[:,:int(im.shape[1])//2]
        tim = cv2.resize(tim, (256,256))
        cv2.imwrite(os.path.join(svpath % ("Night_Train_90", "Night_Test_10")[ch > tt_split], aid), tim)
        tim = im[:,int(im.shape[1])//2:]
        tim = cv2.resize(tim, (256,256))
        cv2.imwrite(os.path.join(svpath % ("Day_Train_90", "Day_Test_10")[ch > tt_split], bid), tim)
