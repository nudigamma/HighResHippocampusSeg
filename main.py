''' Main goes here'''

import ImageUtils
import FileUtils
import random
import os 
import subprocess

configuration = dict()

# change here depending on your workstation 
configuration = {'running_dir':'/home/nadim/build/test','images':"images","masks":"labels",
                'raw_data':'/media/nadim/308ad81c-784b-4b17-be80-e9c758341b81/RFLab/Atlas_Free_HiResHippocmpusAtlasFree/Data/ashs_atlas_magdeburg_7t_20180416/train'}


def prepareData(raw_dir):
    ''' after unziping the downloaded data this beast of a function convert 3D MRI to 2D MRI images and the training dir structures'''
    # create directories
    # the directory images contains the MRI images
    # the directory masks conatins the true labels for MRI Images 
    FileUtils.createDirectories(configuration['running_dir'],configuration['images'])
    FileUtils.createDirectories(configuration['running_dir'],configuration['masks'])
    train_dir = raw_dir# raw_data is where the zip file is extracted that contains the training directory
    subjects = (FileUtils.getSubjectDirs(train_dir))
    for subject in subjects:
        print(os.path.join(train_dir,subject))
        full_path_l= (FileUtils.getImagesFileName(os.path.join(train_dir,subject),'left',''))
        full_path_r= (FileUtils.getImagesFileName(os.path.join(train_dir,subject),'right',''))
        ImageUtils.Im3DToIm2D(full_path_l,'/home/nadim/build/test/images',FileUtils.extractPrefix(full_path_l),False)
        ImageUtils.Im3DToIm2D(full_path_r,'/home/nadim/build/test/images',FileUtils.extractPrefix(full_path_r),False)
        full_path_ls= (FileUtils.getImagesFileName(os.path.join(train_dir,subject),'left','_seg'))
        full_path_rs= (FileUtils.getImagesFileName(os.path.join(train_dir,subject),'right','_seg'))
        ImageUtils.Im3DToIm2D(full_path_ls,'/home/nadim/build/test/labels',FileUtils.extractPrefix(full_path_ls),True)
        ImageUtils.Im3DToIm2D(full_path_rs,'/home/nadim/build/test/labels',FileUtils.extractPrefix(full_path_rs),True)
   
    return
  





def main():
    #Step 1 prepare data
    prepareData(configuration['raw_data'])
    #Step 2 run Training
    subprocess.run(['python','Train.py'])


main()
