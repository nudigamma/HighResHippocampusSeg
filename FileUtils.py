''' file utils '''
''' takes care of creating directories and randomizing validation and training set'''
import shutil as sh 
import os
import ImageUtils
import random
import zipfile
import numpy as np
#TODO: The training data is fetched frm previous segmentatoin online
#      need to implement the unzip


def extract_zip_file(file_name,extract_destination):
    with zipfile.ZipFile(file_name) as zip_ref:
        zip_ref.extractall(extract_destination)


def createDirectories(root_directory,name):
    
    try:
        os.makedirs(os.path.join(root_directory,name))
    except OSError as e:
        print("Coudn't create directory {}".format(e))


def getSubjectDirs(train_dir):
    '''get list of training subjects'''
    if not os.path.isdir(train_dir):
        print("directory does not exist or the path is not correct")
    
    return os.listdir(train_dir)


def getImagesFileName(subject_path, side, seg_suffix):
    base_name = 'tse_native_chunk_'
    seg = seg_suffix
    full_name = base_name+ side + seg +  '.nii.gz'
    return os.path.join(subject_path,full_name)

def extractPrefix(full_name):

    parts = full_name.split(os.sep)
    subparts = parts[-1].split('_')
    if '_seg' in full_name:
        prefix = parts[-2] + '_' + subparts[-2]
    else:
        prefix = parts[-2] + '_' + subparts[-1][:-7]
    return prefix

def shuffleimgs(imgs_filenames):

    return random.sample(imgs_filenames,len(imgs_filenames))

def split_train_valid(imgs_filenames,ratio):
    valid_size = int(np.ceil(len(imgs_filenames)*ratio))

    return imgs_filenames[:valid_size],imgs_filenames[valid_size:]

def copyfiles_train_valid(main_dir,imgs_filenames,destination):
    try:
        for imgs_filename in imgs_filenames:
            sh.copy(os.path.join(main_dir,imgs_filename),os.path.join(main_dir,destination))
    except Exception as e :
        print("Something wrong went with copy {}".format(e))






