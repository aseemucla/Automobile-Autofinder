import numpy as np
from PIL import Image, ImageTk
import glob
import json
import csv

def placeimagearr():
	image_list = []
	i = 0

	for filename in glob.glob('testfolder/*'): 
		im = Image.open(filename).resize((1024, 786), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list.append(arr)
		print(image_list[i].shape)
		i = i + 1

	with open("imagetest.txt", "w") as outfile:
		json.dump(np.array(image_list).tolist(), outfile)

def getimagearr():
	with open("imagetest.txt", "r") as infile:
		image_list = json.load(infile)
		print(np.array(image_list).shape)

