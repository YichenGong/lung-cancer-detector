import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
# import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle



class DataLoad:
    def __init__(self,config):
        # INPUT_FOLDER = config.input_folder 
        self.config = config
        self.INPUT_FOLDER = 'sample_images/'
        self.patients = os.listdir(INPUT_FOLDER)
        self.patients.sort() # patient ids 
        self.patients = [item for item in patients if not item.startswith("._")]

    def loadData(self,typeData="HU"):
        pass

    def preprocess_data(self,typeData="HU"):
        scans = {}
        for idx, patientId in enumerate(self.patients):
            patient_scan = self.load_scan(INPUT_FOLDER + patientId)
            patient_pixel = self.get_pixels_hu(patient_scan)
            pix_resampled, spacing = self.resample(patient_pixel, patient_scan, [1,1,1])
            segmented_lungs_fill = self.segment_lung_mask(pix_resampled, True)
            scans[patientId] = segmented_lungs_fill
        fh = open("data/preprocessed/sample_images.p", 'wb')
        pickle.dump(scans, fh)
        fh.close()

    def load_scan(self,path):
        dire = [item for item in os.listdir(path) if not item.startswith("._")]
        slices = [dicom.read_file(path + '/' + s) for s in dire]

        print(slices[1])
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
        
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        
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
                l_max = largest_label_volume(labeling, bg=0)
                
                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        
        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1
        
        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0
     
        return binary_image

    def largest_label_volume(self, im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None
