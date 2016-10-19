import numpy as np
import cv2 
from time import sleep
from skimage import filters
from skimage import measure
from scipy.ndimage.measurements import find_objects
import os

np.set_printoptions(threshold=10000000)
myImage = cv2.imread("./1.jpg",0) # 0 converts the image to greyscale

print "Loaded image with properties..."
print "ROWS:",len(myImage) # 206
print "COLUMNS:",len(myImage[0]) # 253

			
####################################################################
### Count Number of symbols in a file
# Otsu method to seperate foreground from background
def count_symbols():
		print "\n count_symbols()"
		image = myImage
		val = filters.threshold_otsu(image)
		mask = myImage < val # boolean matrix 

		all_labels = measure.label(mask)  
		symbols_index = find_objects(all_labels)

		print "NUMBER OF SYMBOLS:",len(symbols_index)

		for c,symbol_i in enumerate(symbols_index):
				symbol = myImage[symbol_i]
				path = os.path.join("./output/symbols/"+str(c)+".jpg")
				cv2.imwrite(path,symbol)


####################################################################
### Count Number of Black Pixels in a symbol
def count_blk_pixels():
		print "\n count_blk_pixels()"
		symbol = cv2.imread("./output/symbols/0.jpg",0) 
		cv2.imwrite("./output/symbol.jpg",symbol)
		blk_pixel_count = 0 
		blk_pixel_positions = []

		for row,x in enumerate(symbol):
				for column,y in enumerate(symbol[row]):
						if symbol[row][column] == 0: 
								blk_pixel_positions.append((row,column))
								blk_pixel_count +=1
		print "Row/Column tuple of black pixels:",blk_pixel_positions
		print "Total pixels:",len(symbol[0])*len(symbol[0])
		print "# of black pixels:",blk_pixel_count
		print "% of black pixels:",float(blk_pixel_count)/(len(symbol[0])*len(symbol[0])) 

####################################################################
### Image Convolution(1) Row
def img_conv1():
		print "\n img_conv1()"
		image = myImage
		new_image = np.empty(np.shape(myImage)) 
		convolution =[[0.2,0.2,0.2,0.2,0.2]]

		padded_image=np.pad(image,((0,0),(2,2)),"constant",constant_values=0)
		for row,x in enumerate(image):
				for column,y in enumerate(image[row]):
						column+=2
						new_image[row][column-2]=np.dot(convolution[0],padded_image[row][column-2:column+3])
		cv2.imwrite("./output/convolution1.jpg",new_image)
		print "Convolution applied. Output of img_conv1() to path './output/convolution1.jpg'"

### Image Convolution(2) Column
def img_conv2():
		print "\n img_conv2()"
		image = myImage
		new_image = np.empty(np.shape(myImage)) 
		convolution =[[0.2],[0.2],[0.2],[0.2],[0.2]]

		padded_image=np.pad(image,((2,2),(0,0)),"constant",constant_values=0)
		for row,x in enumerate(image):
				row+=2
				for column,y in enumerate(image[row-2]):
						x_ = [[padded_image[row-2:row+3][i][column]] for i in range(5)]
						total = 0
						# Dot product
						for a,b in zip(x_,convolution):
								total+=a[0]*b[0]
						new_image[row-2][column]= total

		cv2.imwrite("./output/convolution2.jpg",new_image)
		print "Convolution applied. Output of img_conv2() to path './output/convolution2.jpg'"

####################################################################
### Scale Image
def scale_image():
		print "\n scale_image()"
# Helper Function
		def get_i_list(a,b):
				""" Given a range of floats return a list of  integers in range"""
				array = [a]
				r = int(np.floor(b)-np.ceil(a)) +1
				array.extend([int(ii+np.ceil(a))    for ii in range(r)])
				if (array[-1] !=b):
						array.append(b)
				return array

		# Parameters
		original_img = myImage
		scale_size = (16,16)
		scaled_img = np.ones(scale_size)
		ratio= ((np.shape(original_img)[0]-1)/float(scale_size[0]),(np.shape(original_img)[1]-1)/float(scale_size[1]))
		threshold = 0.2

		for i in range(scale_size[0]): # For 48 iterations
				x_0 = i* ratio[0]
				x_1 = (i+1) *ratio[0]-1
				for j in range(scale_size[1]): # For 32 iterations
						y_0 = j*ratio[1]
						y_1 =(j+1)*ratio[1]-1

						area = 0
						for x_i,y_i in zip([x_0,x_1],[y_0,y_1]):
								if y_i != 252.0:
										x_list = get_i_list(x_0,x_1)
										y_list = get_i_list(y_0,y_1)
										counter = 0
										for l in x_list:
												for m in y_list:
														if type(l) == int and type(m) == int: 
																counter +=1
																area += original_img[l][m]

														if type(l) == float or type(m) == float:
																fraction,fraction2 = 1,1
																if x_list[-1] == l:
																		fraction = l - np.floor(l)
																elif x_list[0] == l:
																		fraction = np.ceil(l) - l
																if y_list[-1] == m:
																		fraction2 = m - np.floor(m)
																elif y_list[0] == m:
																		fraction2 = np.ceil(m) - m
																counter+=fraction*fraction2
																area += original_img[int(l)][int(m)]*fraction*fraction2
								if y_i == 252.0:
										continue
						# Normalize total area, returns value between 0-255
						scaled_img[i][j] = area/counter

						if scaled_img[i][j] < (255*(1-threshold)):
								# Lower 80% of pixels will be converted to black
								scaled_img[i][j] = 0
						else: 
								scaled_img[i][j] = 255
		cv2.imwrite("./output/scaled.jpg",scaled_img)
		print "Image scaled to",scale_size,"output sent to path './output/scaled.jpg'"

if __name__ =="__main__":
		count_symbols()
		count_blk_pixels()
		img_conv1()
		img_conv2()
		scale_image()
		

