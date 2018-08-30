import numpy as np
from PIL import Image, ImageTk
import glob
import json
import csv

def createimagearr():
	image_list = []
	value_list = []
	image_list_test = []
	value_list_test = []

	for filename in glob.glob('Lamborghini/*'): 
		im = Image.open(filename).resize((512, 393), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list.append(arr)
		value_list.append(0)

	for filename in glob.glob('Ferrari/*'): 
		im = Image.open(filename).resize((512, 393), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list.append(arr)
		value_list.append(1)

	# for filename in glob.glob('Bugatti/*'): 
	# 	im = Image.open(filename).resize((1024, 786), Image.ANTIALIAS).convert('L')
	# 	arr = np.array(im)
	# 	image_list.append(arr)
	# 	value_list.append(2)

	for filename in glob.glob('McLaren/*'): 
		im = Image.open(filename).resize((512, 393), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list.append(arr)
		value_list.append(2)

	for filename in glob.glob('LamboTest/*'): 
		im = Image.open(filename).resize((512, 393), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list_test.append(arr)
		value_list_test.append(0)

	for filename in glob.glob('FerrariTest/*'): 
		im = Image.open(filename).resize((512, 393), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list_test.append(arr)
		value_list_test.append(1)

	# for filename in glob.glob('BugattiTest/*'): 
	# 	im = Image.open(filename).resize((1024, 786), Image.ANTIALIAS).convert('L')
	# 	arr = np.array(im)
	# 	image_list_test.append(arr)
	# 	value_list_test.append(2)

	for filename in glob.glob('McLarenTest/*'): 
		im = Image.open(filename).resize((512, 393), Image.ANTIALIAS).convert('L')
		arr = np.array(im)
		image_list_test.append(arr)
		value_list_test.append(2)

	print("created arrays")

	with open("images.txt", "w") as outfile:
		json.dump(np.array(image_list).tolist(), outfile)

	with open("types.txt", "w") as outfile:
		json.dump(np.array(value_list).tolist(), outfile)

	with open("imagestest.txt", "w") as outfile:
		json.dump(np.array(image_list_test).tolist(), outfile)

	with open("typestest.txt", "w") as outfile:
		json.dump(np.array(value_list_test).tolist(), outfile)
	

def getimagearr():
	overalllist = []
	with open("drive6/automobileautofinder (7fa602ef)/images.txt", "r") as infile:
		image_list = json.load(infile)
		print("shape of image_list:")
		print(np.array(image_list).shape)
		overalllist.append(np.array(image_list))
	with open("drive6/automobileautofinder (7fa602ef)/types.txt", "r") as infile:
		value_list = json.load(infile)
		print("shape of value_list:")
		print(np.array(value_list).shape)
		overalllist.append(np.array(value_list))
	with open("drive6/automobileautofinder (7fa602ef)/imagestest.txt", "r") as infile:
		image_list_test = json.load(infile)
		print("shape of image_list_test:")
		print(np.array(image_list_test).shape)
		overalllist.append(np.array(image_list_test))
	with open("drive6/automobileautofinder (7fa602ef)/typestest.txt", "r") as infile:
		value_list_test = json.load(infile)
		print("shape of value_list_test:")
		print(np.array(value_list_test).shape)
		overalllist.append(np.array(value_list_test))
	return overalllist
