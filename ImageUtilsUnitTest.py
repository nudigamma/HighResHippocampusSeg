# Nadim Farhat
# V.001
# just a unit test to test the functionality implemented 


import SimpleITK as sitk 
import numpy as np
import sys, time ,os 
import matplotlib.pyplot as plt
import ImageUtils
#main goes here 
def main():
	    	
    if len(sys.argv) < 3:
        print("Usage: python" + __file__ + " outdirectory")
        sys.exit( 1 )

    nii_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    ImageUtils.Im3DToIm2D(nii_path,"Test",output_dir_path,True)

    


main() # main goes here 




