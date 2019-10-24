#TODO:
#   ERRORS TO FIX:
#		- Equalization RGB images -> HSV
#       - Geometric Mean is not working
#       - Gaussian Bandreject has a bad behavior
#       - Some image does not open
#       - High exponent values has strange bahviour in power function
#       - Close plot window block a new window generation


#Imports
import numpy   as np #Numpy
import sys
import os

sys.path.append('modules/')

from soin_image import soin_image
from soin_gui import soin_gui
from image_processor import image_processor, filter_processor, fourier

#################
### UNIT TEST ###
#################

##Filter Processor
#filt_proc = filter_processor()
#
##Fourier Init
#fourier_hand = fourier()
#
##Image Processor Init
#img_proc      = image_processor(filt_proc, fourier_hand)
#
##Init Image
#soin_img_hand = soin_image(r"image_examples/chap3/Fig0304(a)(breast_digital_Xray).tif", img_proc)
#soin_img_hand.open_image()
#
##Showing Original
#soin_img_hand.show_image()
#
##Processing Image
##soin_img_hand.hsv_update()
#
#p = img_proc.rotate(soin_img_hand.img_array, 90)
#
##p =img_proc.chroma_key(soin_img_hand.img_array, None, [0,255,0], 10)
#
#
##Updating and showing processed image
#soin_img_hand.img_array = p;
#soin_img_hand.update_image()
#soin_img_hand.show_image()

#################
###APPLICATION###
#################

def run():
	#Filter Processor
	filt_proc = filter_processor()
	
	#Fourier Init
	fourier_hand = fourier()
	
	#Image Processor Init
	img_proc      = image_processor(filt_proc, fourier_hand)
	
	##Image Init
	soin_img_hand = soin_image("image_examples/general/lena_gray_256.tif", img_proc)
	soin_img_hand.open_image()
	
	#Gui init
	current_path = os.path.dirname(os.path.abspath(__file__))
	soin_gui_hand = soin_gui(None, img_proc, current_path)
	soin_gui_hand.build_wm()
	soin_gui_hand.run_wm()

run()