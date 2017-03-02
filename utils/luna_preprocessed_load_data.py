import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import dicom
import os
import math
import csv
# import scipy.ndimage  as nd
# import matplotlib.pyplot as plt
# from skimage import measure, morphology
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gzip
import pickle
import SimpleITK as sitk
# from multiprocessing import Pool, TimeoutError
# from scipy.misc import imresize
# import tensorflow as tf 
# import json

class DataLoad():
    def __init__(self, config):
        self.data_path = config.data_path
        self.mask_path = config.mask_path
        self.batch_size = config.batch_size
        self.train_ratio = config.train_ratio
        self.original_data_path = config.original_data_path
        # self.circular = True
        self._build_data(config)

    def _build_data(self, config):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            os.mkdir(self.mask_path)
            self.preprocess(config)
            
        self.patient_images = os.listdir(self.data_path)
        self.train_num = math.floor(len(self.patient_images) * self.train_ratio)
        self.val_num = len(self.patient_images) - self.train_num
        self.patient_images = np.random.permutation(self.patient_images)
        self.train_images = self.patient_images[:self.train_num]
        self.val_images = self.patient_images[self.train_num:]
        self.current_idx = 0

    def train(self):
        self._reset()
        self.train_mode = True
        self.p_imgs = self.train_images

    def validation(self):
        self._reset()
        self.validation_mode = True
        self.p_imgs = self.val_images

    def _reset(self):
        self.train_mode = False
        self.validation_mode = False
        self.current_idx = 0

    # def self.has_next_batch()
    #   return self.current_idx < len(p_imgs)

    def next_batch(self, batch_size):
        assert self.train_mode or self.validation_mode, "Please set mode, train, validation or test. e.g. DataLoad.train()"
        idx_next_batch = [(self.current_idx + i)%len(self.p_imgs) for i in range(self.batch_size)]
        patient_img_next_batch = [ self.p_imgs[idx] for idx in idx_next_batch]
        batch_image = []
        batch_mask = []
        for image in patient_img_next_batch:
            fi = gzip.open(self.data_path + image, 'rb')
            img = pickle.load(fi)
            img = np.expand_dims(img, axis=2)
            batch_image.append(img)
            fi.close()
            fm = gzip.open(self.mask_path + image, 'rb')
            mask = pickle.load(fm)
            fm.close()
            mask_binary_class = np.zeros([mask.shape[0],mask.shape[1],2])
            mask_binary_class[:,:,0][mask == 0] = 1
            mask_binary_class[:,:,1][mask == 1] = 1
            batch_mask.append(mask_binary_class)
        self.current_idx = (self.current_idx + batch_size) % len(self.p_imgs)
        batched_image = np.stack(batch_image)
        batched_mask = np.stack(batch_mask)
        return batched_image, batched_mask

    def __call__(self, batch_size):
        return self.next_batch(batch_size)





    def preprocess(self, config):
        self.annotation = self.read_annotation_CSV(config.annotation_file_path)
        self.patient_scan_fns = os.listdir(self.original_data_path)
        self.patient_scan_fns = [f for f in self.patient_scan_fns if f.endswith(".mhd") if f.rstrip(".mhd") in self.annotation.keys()]
        print("There are " + str(len(self.patient_scan_fns)) + " scans to process")
        
        for idx, patient_scan_fn in enumerate(self.patient_scan_fns):
            print("Processing {} image".format(idx) )
            image, origin, spacing = self.load_itk(self.original_data_path + patient_scan_fn)
            image = self.normalize(image)
            image = self.zero_center(image)
            patient_id = patient_scan_fn.rstrip(".mhd")
            annos = self.annotation[patient_id]

            mask = np.zeros(image.shape)
            layer_nums = set()

            for anno in annos:
                anno = [float(num) for num in anno]
                voxelCoord = self.round(self.worldToVoxelCoord(np.array(anno[:3]), origin, spacing))
                nodule_radius = (self.round(anno[3]/spacing[0]/2) , self.round(anno[3]/spacing[1]/2), self.round(anno[3]/spacing[2]/2))
                layer_nums = layer_nums | set(range(voxelCoord[2]-nodule_radius[2], voxelCoord[2] + nodule_radius[2]))

                for x in range(voxelCoord[0]-nodule_radius[0], voxelCoord[0]+nodule_radius[0]):
                    for y in range(voxelCoord[1]-nodule_radius[1], voxelCoord[1]+nodule_radius[1]):
                        for z in range(voxelCoord[2]-nodule_radius[2], voxelCoord[2]+nodule_radius[2]):
                            current_world_idx = self.voxel_2_world(np.array([x,y,z]), origin, spacing)
                            if np.linalg.norm(current_world_idx - np.array(anno[:3])) < anno[3]/2:
                                mask[x,y,z] = 1


            for layer_num in layer_nums:
                f = gzip.open(config.data_path + patient_id + '_slice{}.pkl.gz'.format(layer_num) ,'wb')
                pickle.dump(image[:,:,layer_num], f)
                f.close()

                f = gzip.open(config.mask_path + patient_id + '_slice{}.pkl.gz'.format(layer_num) ,'wb')
                pickle.dump(mask[:,:,layer_num], f)
                f.close()
                #Store the masks and original image


    def worldToVoxelCoord(self, worldCoord, origin, spacing):
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        return voxelCoord

    def voxel_2_world(self, voxel_coord, origin, spacing):
        stretched_voxel_coord = voxel_coord * spacing
        world_coord = stretched_voxel_coord + origin
        return world_coord

    def load_itk(self, filename):
        itkimage = sitk.ReadImage(filename)
        image = np.transpose(sitk.GetArrayFromImage(itkimage))
        origin = np.array(itkimage.GetOrigin())
        spacing = np.array(itkimage.GetSpacing())
        return image, origin, spacing

    def normalize(self, image):
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image

    def zero_center(self,image):
        PIXEL_MEAN = 0.25
        image = image - PIXEL_MEAN
        return image

    def read_annotation_CSV(self,filename):
        lines = []
        with open(filename,'r') as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                lines.append(line)
        csvdict = {}
        for i in range(1, len(lines)):
            key, coordX, coordY, coordZ, diameter_mm = lines[i]
            info_list = csvdict.get(key, -1)
            if info_list != -1:
                info_list.append((coordX, coordY, coordZ, diameter_mm))
            else:
                csvdict[key] = [(coordX, coordY, coordZ, diameter_mm)]
        return csvdict

    def round(self, num):
        return np.round(num).astype(int)




