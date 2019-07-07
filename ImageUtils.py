import SimpleITK as sitk 
import numpy as np 
import os 
#TODO: handle error and exceptions without quittting 
#TODO: Should images be 16 bit to have a better dynamic range ? 
#TODO : Extract slices in a different plan of acquistion

'''
    0      "Clear Label"
    1      "CA1"
    2       CA2"
    3      "DG"
    4      "CA3"
    5      "Tail"
    6      "Label 6"
    7      "Label 7"
    8      "Sub"
    9      "ErC"
   10      "A35"
   11       "A36"
   12       "PhC"
   13       "Cysts"
   17       "Label 17"

'''
MIN_TH = 1 
MAX_TH = 8
def  merge_all_labels(label_image,lower_thershold,upper_thershold):
    # type itk image
    thershold_filter = sitk.BinaryThresholdImageFilter()
    thershold_filter.SetInsideValue(1)
    thershold_filter.SetOutsideValue(0)
    thershold_filter.SetUpperThreshold(upper_thershold) # more than five are not hippoampus value
    thershold_filter.SetLowerThreshold(lower_thershold) #
    label_image = thershold_filter.Execute(label_image)
    return label_image 




def Im3DToIm2D(im_3D_Filename,output_dir_path,prefix,label=False):
    ''' Take a 3D file like NIFTI or DCM and then extracts the 3D images as a 
    set of 3D images'''
    print("Started Extracting {} to dir {}".format(im_3D_Filename,output_dir_path))
    #Sanity Checks for input filename 
    if not os.path.isfile(im_3D_Filename):
        print('''Couldn't find input filname, Please make sure the file exist or 
                the path is correct ''')
        # i am not sure how to handle exceptions at this stage
    #Sanity check for output directory 
    if not os.path.isdir(output_dir_path):
        print('''Output dir doenst exist, Please make sure the file exist or 
                the path is correct ''')

        quit() # i am not sure how to handle exceptions at this stage


    # reads the file into simpleITK
    image = sitk.ReadImage(im_3D_Filename)
    #prepare to write to disk by creating the writer
    if label:
        image = merge_all_labels(image,MIN_TH,MAX_TH)
        
    
    # rescale the intensity , I am not sure if we need to do this in 16 bit later on
    #sitk.WriteImage(image,os.path.join(output_dir_path,'one_label.nii')) # just for validation

    sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8), 
               [os.path.join(output_dir_path, prefix +'{0:03d}.jpeg'.format(i)) for i in range(image.GetSize()[2])])

    print("Finished extracting!")

