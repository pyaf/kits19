'''
Hmmm, lets move those folders and structure them
I will you have a folder in which you will have 210 scans like so case_0000, case_0001 ....
Lets start

This is not tested since i had wrriten mine long back,test this on one file and then you can do this for the whole directory
'''
import os
import shutil
scan_folder = '/media/bubbles/fecf5b15-5a64-477b-8192-f8508a986ffe/ai/ryan/data/'
#make folder where you will move this
#This is exactly my structure too.This is also to avoid keeping data in the main folder
move_path_images  = '../data/kits/Images'
move_path_masks   = '../data/kits/Masks'
#os.mkdir(move_path_masks)
#os.mkdir(move_path_images)
for scans in os.listdir(scan_folder):
    #lets skip irrelevent files
    if scans == 'kits.json':
        continue
    if scans == 'LICENSE':
        continue
    #break
    for item in os.listdir(os.path.join(scan_folder, scans)):
        #lets take the segmenation and move it
        if item.endswith('on.nii.gz'):
            os.rename(os.path.join(scan_folder, scans, item), os.path.join(scan_folder,scans, scans+'.nii.gz'))
            shutil.copy(os.path.join(os.path.join(scan_folder,scans,scans+'.nii.gz')), os.path.join(move_path_masks, scans+'.nii.gz'))#not sure if this should be scans+'.nii.gz' or just 'item'
        else:
            os.rename(os.path.join(scan_folder, scans, item), os.path.join(scan_folder,scans, scans+'.nii.gz'))
            shutil.copy(os.path.join(os.path.join(scan_folder,scans,scans+'.nii.gz')), os.path.join(move_path_images, scans+'.nii.gz'))#not sure if this should be scans+'.nii.gz' or just 'item'
