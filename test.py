from preprocess import *
from PIL import Image
import os
import sys

sys.path.append('D:\实习\data\LIDC-IDRI-0003\1.3.6.1.4.1.14519.5.2.1.6279.6001.101370605276577556143013894866\1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264')

dir = './'

if __name__=='__main__':

    imgs, mask_2, spc = mask_extractor(dir)
    sliceim, _ = mask_processing(imgs,mask_2,spc)
    print(sliceim.shape)

    for i in range(len(sliceim)):
        img = Image.fromarray(sliceim[i])
        img.save('%d.jpg'%i)


