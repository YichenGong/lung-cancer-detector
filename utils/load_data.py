import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage  as nd
# import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
from multiprocessing import Pool, TimeoutError
from scipy.misc import imresize
import tensorflow as tf 



class DataLoad():
    def __init__(self,config):
        # INPUT_FOLDER = config.input_folder 
        self.config = config
        self.height = config.height
        self.width = config.width
        self.layers = config.layers 
        self.batch_size = config.batch_size
        self.data_type = config.data_type
        self.padding_number = 0
        self.is_train = config.is_train

        self._build_data_info(self.data_type)



    def next_batch(self):
        idx_next_batch = [(self.current_idx + i) % self.num_patients for i in range(self.batch_size)]
        patient_id_next_batch = [ self.patient_ids[idx] for idx in idx_next_batch]
        batch_sample = []
        for patient_id in patient_id_next_batch:
            scan = pickle.load(open("data/preprocessed/" + self.data_type + "/" + patient_id + ".p","rb"))
            resized_scan = self.resize(scan)
            batch_sample.append(resized_scan)
        self.current_idx = (self.current_idx + self.batch_size) % self.num_patients 
        return np.stack(batch_sample), np.array([self.labels[pid] for pid in patient_id_next_batch])


    def resize(self, scan):
        print(scan.shape)
        # scan = imresize(scan, (self.height, self.width))
        # scan_layer = scan.shape[0]
        # if scan_layer < self.layers:
        #     #pad
        #     top_layer = (self.layers - scan_layer) / 2 
        #     bottom_layer = self.layers - scan_layer - top_layer
        #     top_pad = np.zeros((top_layer, self.height, self.width)) + self.padding_number
        #     bottom_pad = np.zeros((bottom_layer, self.height, self.width)) + self.padding_number
        #     resized_scan = np.vstack([top_pad, scan, bottom_pad])

        # else:
        #     # peel
        #     top_layer = (scan_layer - self.layers) / 2
        #     bottom_layer = scan_layer - self.layers - top_layer
        #     resized_scan = scan[top_layer: top_layer + self.layers, :,:]

        zoomFactors = [bi/float(ai) for ai, bi in zip(scan.shape, (self.layers, self.height, self.width))]
        resized_scan = nd.interpolation.zoom(scan, zoom=zoomFactors) 

        return resized_scan


    def _build_labels(self):
            label_file = pd.read_csv("data/stage1_labels.csv", index_col=0).T.to_dict()
            labels = {key: value['cancer'] for key, value in label_file.items()}
            return labels


    def _build_data_info(self,typeData="sample"):
        assert(typeData == "sample" or typeData == "stage1")
        files = os.listdir("data/preprocessed/" + typeData + "/")
        files = [f for f in files if f.endswith(".p")]
        self.labels = self._build_labels()
        self.patient_ids = [fn.rstrip(".p") for fn in files]
        if self.is_train:
            self.patient_ids = [i for i in self.patient_ids if self.labels.get(i, -1) != -1]
        self.num_patients = len(self.patient_ids)
        self.read_idx = list(np.random.permutation(self.num_patients))
        self.current_idx = 0



        

    def preprocess_data(self,typeData="sample"):
        assert(typeData == "sample" or typeData == "stage1")
        if typeData == "sample":
            self.input_folder = 'data/sample/images/'
        else:
            self.input_folder = 'data/stage1/images/'

        self.patients = os.listdir(self.input_folder)
        self.patients.sort() # patient ids 
        self.patients = [item for item in self.patients if not item.startswith("._")]

        
        for idx, patientId in enumerate(self.patients):
            print("Processing " + str(idx) + " image")
            patient_scan = self.load_scan(self.input_folder + patientId)
            patient_pixel = self.get_pixels_hu(patient_scan)
            pix_resampled, spacing = self.resample(patient_pixel, patient_scan, [1,1,1])
            segmented_lungs_fill = self.segment_lung_mask(pix_resampled, True)
            # scans[patientId] = segmented_lungs_fill
            print("finish " + str(idx))
            print(segmented_lungs_fill.shape)
            fh = open("data/preprocessed/" + typeData + "/" + patientId +".p", 'wb')
            pickle.dump(segmented_lungs_fill, fh)
            fh.close()

    def multiprocess_preprocess_data(self, typeData="stage1", patient_ids):
        for idx, patientId in enumerate(self.patients):
            print("Processing " + str(idx) + " image")
            patient_scan = self.load_scan(self.input_folder + patientId)
            patient_pixel = self.get_pixels_hu(patient_scan)
            pix_resampled, spacing = self.resample(patient_pixel, patient_scan, [1,1,1])
            segmented_lungs_fill = self.segment_lung_mask(pix_resampled, True)
            scans[patientId] = segmented_lungs_fill
            print("finish " + str(idx))
            print(segmented_lungs_fill.shape)
            fh = open("data/preprocessed/" + typeData + "/" + patientId +".p", 'wb')
            pickle.dump(scans, fh)
            fh.close()


    def load_scan(self,path):
        dire = [item for item in os.listdir(path) if not item.startswith("._")]
        slices = [dicom.read_file(path + '/' + s) for s in dire]

        # print(slices[1])
        slices.sort(key = lambda x: int(x.InstanceNumber))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            
        for s in slices:
            s.SliceThickness = slice_thickness
        
        return slices

    def get_pixels_hu(self,scans):
        image = np.stack([s.pixel_array for s in scans])
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        
        # Convert to Hounsfield units (HU)
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
        
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
            
        image += np.int16(intercept)
        
        return np.array(image, dtype=np.int16)

    def resample(self,image, scan, new_spacing=[1,1,1]):
        # Determine current pixel spacing
        spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
        spacing = np.array(list(spacing))

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        
        image = nd.interpolation.zoom(image, real_resize_factor)
        
        return image, new_spacing

    def largest_label_volume(self,im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    def segment_lung_mask(self,image, fill_lung_structures=True):
    
        # not actually binary, but 1 and 2. 
        # 0 is treated as background, which we do not want
        binary_image = np.array(image > -320, dtype=np.int8)+1
        labels = measure.label(binary_image)
        
        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air 
        #   around the person in half
        background_label = labels[0,0,0]
        
        #Fill the air around the person
        binary_image[background_label == labels] = 2
        
        
        # Method of filling the lung structures (that is superior to something like 
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self.largest_label_volume(labeling, bg=0)
                
                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        
        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1
        
        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0
     
        return binary_image


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer("width", 300, "width")
    flags.DEFINE_integer("height", 300, "height")
    flags.DEFINE_integer("layers", 300, "layers")
    flags.DEFINE_integer("batch_size", 3, "batch size")
    flags.DEFINE_bool("is_train", True, "is train")
    flags.DEFINE_string("data_type", "sample", "sample or stage1")
    config = flags.FLAGS

    data_loader = DataLoad(config=config)
    # data_loader.preprocess_data("sample")
    # print("Done proprocess sample")

    # data_loader.preprocess_data("stage1")
    # print("Done preprocess stage 1")
    for i in range(20):
        batch_data, label = data_loader.next_batch()
        print(batch_data.shape)