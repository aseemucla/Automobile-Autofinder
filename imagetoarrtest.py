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
		print(image_list[i].shape)
		i = i + 1

	with open('imagetest.txt', 'w') as csvfile:
			matrixwriter = csv.writer(csvfile, delimiter=' ')
			for row in image_list:
				matrixwriter.writerow(row)


# image_list = []
# i = 0

# for filename in glob.glob('testfolder/*'): 
# 	im = Image.open(filename).resize((1024, 786), Image.ANTIALIAS).convert('L')
# 	arr = np.array(im)
# 	image_list.append(arr)
# 	print(np.array(image_list).shape)
# 	i = i + 1


# with open('imagetest.txt', 'w') as csvfile:
# 	i = 0
# 	matrixwriter = csv.writer(csvfile, delimiter='\t')
# 	for row in image_list:
# 		matrixwriter.writerow(row)
# 		i = i + 1
# 		print(i)


image_list = np.array([])
real_list = np.array([])


with open('imagetest.txt', 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter='\t')
	matrixreader = list(matrixreader)
	i= 0
	for row in matrixreader:
		image_list = np.concatenate((image_list, np.array(row)))
		i = i + 1
		print(i)

#print((image_list[0].replace("[", "").split())[0])

print(image_list.shape)
