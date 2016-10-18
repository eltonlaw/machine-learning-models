import numpy as np
import cv2 
from time import sleep
from skimage import filters
from skimage import measure
from scipy.ndimage.measurements import find_objects

np.set_printoptions(threshold=10000000)
myImage = cv2.imread("./1.jpg",0) # 0 converts the image to greyscale

print "ROWS:",len(myImage) # 206
print "COLUMNS:",len(myImage[0]) # 253

			
####################################################################
# Count Number of symbols in a file

# Otsu method to seperate foreground from background
image = myImage
val = filters.threshold_otsu(image)
mask = myImage < val # boolean matrix 

all_labels = measure.label(mask)  
symbols = find_objects(all_labels)
print len(symbols)
mySymbol = symbols[0]

####################################################################
# Count Number of Black Pixels in a symbol
symbol = myImage[mySymbol]
blk_pixel_count = 0 
blk_pixel_positions = []
for row,x in enumerate(symbol):
		for column,y in enumerate(symbol[row]):
				if symbol[row][column] == 0: 
						blk_pixel_positions.append((row,column))
						blk_pixel_count +=1
print "Row/Column tuple of black pixels:",blk_pixel_positions, "\n"
print "Total # of pixels:",len(symbol[0])*len(symbol[0])
print "# of black pixels:",blk_pixel_count
print "% of black pixels[0,1]:",float(blk_pixel_count)/(len(symbol[0])*len(symbol[0])) 

####################################################################
# Image Convolution(1) Row
image = myImage
new_image = np.empty(np.shape(myImage)) 
convolution =[[0.2,0.2,0.2,0.2,0.2]]

padded_image=np.pad(image,((0,0),(2,2)),"constant",constant_values=0)
for row,x in enumerate(image):
		for column,y in enumerate(image[row]):
				column+=2
				new_image[row][column-2]=np.dot(convolution[0],padded_image[row][column-2:column+3])
cv2.imwrite("output_1_convolution1.jpg",new_image)

# Image Convolution(2) Column
image = myImage
new_image = np.empty(np.shape(myImage)) 
convolution =[[0.2],[0.2],[0.2],[0.2],[0.2]]

padded_image=np.pad(image,((2,2),(0,0)),"constant",constant_values=0)
for row,x in enumerate(image):
		row+=2
		for column,y in enumerate(image[row:-2]):
				print "(row,column):",row,column
				x_ = [[padded_image[row-2:row+3][i][column]] for i in range(5)]
				total = 0
				for a,b in zip(x_,convolution):
						total+=a[0]*b[0]
				new_image[row-2][column]= total

cv2.imwrite("output_1_convolution2.jpg",new_image)

####################################################################
# Scale Image
original_img = myImage
scale_size = (48,32)
ratio= ((np.shape(image)[0]-1)/float(scale_size[0]),(np.shape(image)[1]-1)/float(scale_size[1]))
threshold = 0.2

# If the current steps to take is > 1, take a whole step and add the area you stepped to total area
for i in range(scale_size[0]): # For 48 iterations
		x = 0
		y = 0
		for j in range(scale_size[1]): # For 32 iterations
				area = 0
				c = [ratio[0],ratio[1]]
				
				print "[",i,"][",j,"](x,y):",x,y
				while c != [0,0]: 	
						# For x column
						if c[0] > 1:
								area+=original_img[x][y]
								x+=1
								c[0] -= 1
						else:
								area+=original_img[x][y]*c[0]
								x+=c[0]
								c[0] = 0
						# For y column
						if c[1] > 1:
								area+=original_img[x][y]
								y+=1
								c[1] -= 1
						else: 
								area+=original_img[x][y]*c[1]
								y+=c[1]
								c[1] = 0
				# Normalize total area, returns value between 0-255
				scaled_img[i][j] = area/(ratio[0]*ratio[1])	

				if scaled_img[i][j] < (255*(1-threshold)):
						scaled_img[i][j] = 0
				else: 
						scaled_img[i][j] = 255

for i in range(scale_size[0])
		# need some way to iterate over floating point numbers
		x = 0
		y = i*ratio[1]
scaled_image = cv2.resize(original_img,scale_size)	
cv2.imwrite("output_1_scaled.jpg",scaled_img)

