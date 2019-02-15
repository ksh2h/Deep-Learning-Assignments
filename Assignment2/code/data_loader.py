from zipfile import ZipFile
import numpy as np
from os.path import join,dirname,abspath

'''load your data here'''

class DataLoader(object):
    def __init__(self):
        self.DIR = join(dirname(dirname(abspath("data_loader.py"))),"data")
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = join(self.DIR , label_filename + '.zip')
        image_zip = join(self.DIR , image_filename + '.zip')
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8).copy()
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784).copy().astype(np.float32)/255

        return images, labels

    def create_batches(self):
        pass