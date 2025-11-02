import numpy as np 
import math as m
from random import random

# FITNESS FUNCTION 
def fitness_function( func_no, x ):
    global FEs
    
    # SPHERE FUNCTION
    if func_no == 1: 
        sum = 0.0
        for i in range( len(x) ):
            sum = sum + m.pow(x[i],2)

    # WRITE A NEW FUNCTION HERE
	# elif func_no == 2:

    FEs = FEs + 1
    return sum


FEs = 0		# COUNTING THE NUMBER OF TIMES FITNESS FUNCTION IS CALLED
D   = 30	# ASSUME THE DIMENSIONALITY OF THE PROBLEM TO BE 30


###############
# CONSTRAINTS #
###############
lower_bound = -5.12
upper_bound =  5.12


############################
# GENERATE RANDOM SOLUTION #
############################
x = [None] * D 		# A SOLUTION

# INITIALISE THE VALUES FOR x
# ---> 

# DISPLAY THE SAMPLE SOLUTION
print ( 'x: \n', x, '\n') 


###########################################
# EVALUATE AND DISPLAY THE SAMPLE SOLUTON #
###########################################
# ---> 


###########################################
# GENERATE 50 SOLUTIONS AND FIND THE BEST #
###########################################
N = 50				# NUMBER OF SOLUTIONS
# --->  X = ?		# 2D LIST HOLDING ALL N SOLUTIONS
# X SHOULD BE (N x D)


###############################
# EVALUATE THE SAMPLE SOLUTON #
###############################
fitness = [None]*N	# FITNESS VALUES

# ---> 
# ---> 


###################################
# FIND THE INDEX OF BEST SOLUTION #
###################################
# ---> best_index = ?


#######################################################################
# DISPLAY ALL FITNESS VALUES, THE BEST SOLUTION, AND THE BEST FITNESS #
#######################################################################
# ---> print('Fitness values for all solutions: \n', ?, '\n')
# ---> print('Best solution: \n', ?, '\n')
# ---> print('Best fitness value: ', ?)
