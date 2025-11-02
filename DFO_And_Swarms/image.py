'''
TASKS:
	Use this code as a guide for simple image analysis.
	
	1- Update the DFO code to find the brightest area in the input image.
	2- Try the code for other input images
	3- Change input image half-way through and see if your algorithm can perform

ADVANCED TASKS:
	4- Is it possible to trace a moving object? 
	5- How about tracing gradually fading and emerging objects?
	6- What if the input image has two or more bright spots?
	7- If the task was to find symmetry in an image, what would be your fitness function?
'''  

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
#import numpy as np
img = mpimg.imread('input1.png')

imgW = len(img[0])
imgH = len(img)

print('Image size:', imgW, 'x', imgH)

# PRINT PIXEL VALUE AT (ROW No,COLUMN No)
# print(img[100][360]) # PRINT SAMPLE VALUE AT ROW 100, COLUMN 360

# WHITE PIXEL VALUE = 1.0 (PNG) or 255
# BLACK PIXEL VALUE = 0.0 (PNG) or 0  

plt.ion()
for i in range(50):
	imgplot = plt.imshow(img, cmap='gray') # SHOW IMAGE
	rowNo = random.randint(0,imgW) # PICK RANDOM ROW
	colNo = random.randint(0,imgH) # PICK RANDOM COLUMN
	swarmBestCircle = plt.Circle( (rowNo, colNo), 5, color='r') # DRAW A RED CIRCLE OF SIZE 5

	plt.gca().add_patch(swarmBestCircle)	# ADD THE CIRCLE
	plt.draw()	# DRAW THE IMAGE AND THE CIRCLE
	plt.pause(0.0001) # PAUSE BEFORE THE NEXT ITERATION IN BETWEEN
	plt.clf()	# CLEAR THE CANVAS 

