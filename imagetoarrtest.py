import numpy as np
from PIL import Image, ImageTk
import glob

import csv

def getimagearr():
	image_list = []
	i = 0

	for filename in glob.glob('testfolder/*'): 
		im = Image.open(filename).resize((1024, 786), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list.append(arr)
		matrixwriter.writerow(row)
		print(image_list[i].shape)
		i = i + 1

	with open('imagetest.txt', 'wb') as csvfile:
			matrixwriter = csv.writer(csvfile, delimiter=' ')
			for row in image_list:
				matrixwriter.writerow(row)


image_list = []


with open('imagetest.txt', 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=' ')
	for row in matrixreader:
		image_list.append(row)

print(np.array(image_list[0][0][0]))