#Imports
import numpy   as np #Numpy
import os

from PIL import Image #Image
from io import BytesIO

from image_processor import image_processor

class soin_image:

    #Attributes
    img_array     = None
    img_handle    = None
    img_path      = ''
    img_width     = None
    img_height    = None
    is_rgb        = 0
    img_orig      = None
    img_hsv       = [None, None, None]
    img_proc_hand = None

    def __init__(self, path, img_proc_hand):
        self.img_path = path
        self.img_proc_hand = img_proc_hand

    def open_image(self):

        print("> Loading image ...")
        try:    
            self.img_handle = Image.open(self.img_path)#.convert('L')#Converting to grayscale
            self.img_array  = np.array(self.img_handle)#Open the image as Image object and covert to a numpy array
            self.img_orig   = self.img_array
            self.img_width, self.img_height = self.img_handle.size

            if (len(self.img_array.shape) == 3):
                #self.hsv_update()
                self.is_rgb = 1
            
        except IOError:#No file
            print('> An error occurred trying to read the image.')
            return -1
        except:
            print('> An error occurred trying to convert the image to array.')
            return -1

    def save_image(self, dest_path):
        try:
            self.img_handle = Image.fromarray(self.img_array)#Convert array to image object
            self.img_handle.save(dest_path)
        except:
            print('> An error occurred trying to save the image.')
            return -1

    def show_image(self):
        try:
            self.img_handle = Image.fromarray(self.img_array)#Convert array to image object
            self.img_handle.show()
        except:
            print('> An error occurred trying to show the image.')
            return -1

    def update_image(self):
        self.img_handle = Image.fromarray(self.img_array)#Convert array to image object
        self.img_width, self.img_height = self.img_handle.size
        
        if (len(self.img_array.shape) == 3):
                #self.hsv_update()
                self.is_rgb = 1
        else:
            self.is_rgb = 0

    def hsv_update(self):
        self.img_hsv = np.zeros([self.img_width, self.img_height, 3])
        for i in range(0, self.img_width):
            for j in range(0, self.img_height):
                self.img_hsv[i,j,:] = self.img_proc_hand.rgb2hsv(self.img_array[i,j,:])

    def rgb_update(self):
        for i in range(0, self.img_width):
            for j in range(0, self.img_height):
                self.img_array[i,j,:] = self.img_proc_hand.hsv2rgb(self.img_hsv[i,j,:])