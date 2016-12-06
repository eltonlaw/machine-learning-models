import tensorflow as tf

# Reduce dimensionality of data
x = cv2.imread(input_path,0) # Grayscale 
x = np.resize(x,(110,84)) # Resize to 110x84
x= x[12:98]# Crop edges to 84x84,

n_valid_actions =  # Varies between 4 and 18 for the games considered
layers = [84*84*4,168*8,324*4,n_valid_actions]

for l1,l2 in zip(layers[1:],layers[:1]):
    
