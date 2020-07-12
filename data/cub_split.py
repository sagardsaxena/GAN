import random
import tqdm
import os
import cv2

tt_split = .9
dapth = "CUB_200/CUB_200_2011/images"
birds = os.listdir(dapth)
for bird  in tqdm.tqdm(birds):
    bpth = os.path.join(dapth, bird)
    imgs = os.listdir(bpth)
    for img in imgs:
        impth = os.path.join(bpth, img)
        im = cv2.imread(impth)
        im = cv2.resize(im, (256,256))
        cv2.imwrite(os.path.join("../StyleGAN2/data/CUB_200_%s" % ("Train_90", "Test_10")[random.random() > tt_split], img), im)
