import os
import SimpleITK as sitk
import numpy as np
from numpy.lib import stride_tricks
import tqdm
import shutil
import torch
import logging
import sys
from torchnet.meter import ConfusionMeter
import json
import cv2

#Step 1 read in the images
def read_image_and_seg(image_file_path, seg_file_path):
    """
    takes in filepath and returns the numpy array of image and seg
    """
    image = sitk.ReadImage(image_file_path)
    seg   = sitk.ReadImage(seg_file_path)
    image = sitk.GetArrayFromImage(image)
    seg   = sitk.GetArrayFromImage(seg)
    return image , seg

def read_sitk_image(image_file_path):
    """
    takes in filepath and returns the numpy array of image and seg
    """
    image = sitk.ReadImage(image_file_path)
    image = sitk.GetArrayFromImage(image)
    return image

def read_sitk_image(mask_file_path):
    """
    takes in filepath and returns the numpy array of image and seg
    """
    mask = sitk.ReadImage(mask_file_path)
    mask = sitk.GetArrayFromImage(mask)
    return image

#Logging
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def get_patches(image, out_path, phase):
    with open('../data/kits/shape_info.json', 'r') as fp:
        data = json.load(fp)
    image_name = image
    read_image = sitk.ReadImage(image)
    image = sitk.GetArrayFromImage(read_image)
    patch_size = 128

    depth, height , width  = image.shape

    if height < patch_size:
        delta_h = patch_size - height
        delta_h+=patch_size#Can you check this again
        image =np.pad(image, ((0,0), (0,delta_h), (0, 0)), 'constant')

    if width < patch_size:
        delta_w = patch_size - width
        delta_w+=patch_size
        image =np.pad(image, ((0,0), (0,0), (0, delta_w)), 'constant')

    if depth < patch_size:
        delta_z = patch_size - depth
        delta_z+=patch_size
        image =np.pad(image, ((0,delta_z), (0,0), (0, 0)), 'constant')

    depth, height , width  = image.shape
    if not height%patch_size==0:
        mod = height%patch_size
        delta_h = patch_size - mod
        image =np.pad(image, ((0,0), (0,delta_h), (0, 0)), 'constant')

    if not width%patch_size==0:
        mod = width%patch_size
        delta_w = patch_size - mod
        image =np.pad(image, ((0,0), (0,0), (0, delta_w)), 'constant')

    if not depth%patch_size==0:
        mod = depth%patch_size
        delta_z = patch_size - mod
        image =np.pad(image, ((0,delta_z), (0,0), (0, 0)), 'constant')

    count=0
    depth_step = image.shape[0] - patch_size 
    height_step  = image.shape[1] - patch_size 
    width_step  = image.shape[2] - patch_size   
    data[image_name] = image.shape

    with open('../data/kits/shape_info.json', 'w') as fp:
        json.dump(data, fp)
    for z in range(0, depth_step+1, patch_size):
        for y in range(0, height_step+1, patch_size):
            for x in range(0,width_step+1, patch_size):
                patch = image[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                if phase == 'train':
                    np.save(os.path.join('{}','{}').format(out_path,image_name.split('/')[-1].split('.')[0]+str(count)), patch)
                    count+=1
                else:
                    np.save(os.path.join('{}','{}').format(out_path,image_name.split('/')[-1].split('_seg')[0]+str(count)), patch)
                    count+=1

def recon_image(npy_folder,original_image, out_path):
    #first convert these to a single numpy array
    with open('../data/kits/shape_info.json', 'r') as fp:
        data = json.load(fp)
    image_name = original_image
    original_image = sitk.ReadImage(original_image)
    origin = original_image.GetOrigin()
    direction = original_image.GetDirection()
    image_to_fill = np.zeros((data[image_name]))

    filenames = os.listdir(npy_folder)
    filenames = sorted(filenames, key = lambda files: files.split('/')[-1].split('.')[0][-3:] ) 
    
    patch_size = 128

    count=0
    depth_step = image_to_fill.shape[0] - patch_size 
    height_step  = image_to_fill.shape[1] - patch_size 
    width_step  = image_to_fill.shape[2] - patch_size   

    print
    for z in range(0,depth_step+1 , patch_size):
        for y in range(0,height_step+1, patch_size):
            for x in range(0,width_step+1,patch_size):
                image_to_fill[x:x+patch_size, y:y+patch_size,z:z+patch_size] = np.load(os.path.join(npy_folder, filenames[count]))
                count+=1
    convert_to_image = sitk.GetImageFromArray(image_to_fill)
    convert_to_image.SetOrigin(origin)
    convert_to_image.SetDirection(direction)
    sitk.WriteImage(convert_to_image,os.path.join(out_path,image_name.split('/')[-1].split('.')[0]+'recon.nii.gz'))
    return 

#Borrowed from https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
def get_metrics(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return  TPR , TNR , PPV , FPR , FNR , ACC

class Meter(ConfusionMeter):
    def __init__(self, k, phase, epoch, save_folder, normalized=False):
        ConfusionMeter.__init__(self, k, normalized=normalized)
        self.predictions = []
        self.targets = []
        self.threshold = 0.5  # used for confusion matrix
        self.phase = phase
        self.epoch = epoch
        self.save_folder = save_folder

    def update(self, targets, outputs):
        self.targets.extend(targets)
        self.predictions.extend(outputs)
        outputs = (outputs > self.threshold).type(torch.Tensor).squeeze()
        self.add(outputs, targets)

    def get_metrics(self):
        conf = self.value().flatten()
        total_images = np.sum(conf)
        TN, FP, FN, TP = conf
        acc = (TP + TN) / total_images
        tpr = TP / (FN + TP)
        fpr = FP / (TN + FP)
        tnr = TN / (TN + FP)
        fnr = FN / (TP + FN)
        precision = TP / (TP + FP)
        roc = roc_auc_score(self.targets, self.predictions)
        plot_ROC(
            roc,
            self.targets,
            self.predictions,
            self.phase,
            self.epoch,
            self.save_folder,
        )
        return acc, precision, tpr, fpr, tnr, fnr, roc

def plot_ROC(roc, targets, predictions, phase, epoch, folder):
    roc_plot_folder = os.path.join(folder, "ROC_plots")
    mkdir(os.path.join(roc_plot_folder))
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_plot_name = "ROC_%s_%s_%0.4f" % (phase, epoch, roc)
    roc_plot_path = os.path.join(roc_plot_folder, roc_plot_name + ".jpg")
    fig = plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.legend(["diagonal-line", roc_plot_name])
    fig.savefig(roc_plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # see footnote [1]

    plot = cv2.imread(roc_plot_path)
    log_images(roc_plot_name, [plot], epoch)

def remove_non_label_patches(mask_dir, image_dir):
    for label in os.listdir(mask_dir):
        read_label = np.load(os.path.join(mask_dir, label))
        if len(np.unique(read_label)) <2:
            os.remove(os.path.join(mask_dir, label))
            try:
                os.remove(os.path.join(image_dir, label))
            except Exception as FileNotFoundError:
                pass

def get_image_from_npy(npy_file,out_path):
    file = np.load(npy_file)
    file = sitk.GetImageFromArray(file)
    return sitk.WriteImage(file,os.path.join(out_path,npy_file.split('/')[-1].split('.')[0]+'.nii.gz') )

MIN_BOUND = -100.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

if __name__ == "__main__":
    image_dir = '../data/kits/patches/images/'
    mask_dir  = '../data/kits/patches/masks/'
    remove_non_label_patches(mask_dir, image_dir)
    # image_folder = '../data/kits/images/'
    # mask_folder  = '../data/kits/masks/'
    # for image in os.listdir(image_folder):
    #     label = image.split('/')[-1].split('.')[0]+'_seg.nii.gz'
    #     print(os.path.join(image_folder, image))
    #     print(os.path.join(mask_folder,label))
    #     get_patches(os.path.join(image_folder, image), '../data/kits/patches/images/','train')
    #     get_patches(os.path.join(mask_folder, label), '../data/kits/patches/masks/','mask')