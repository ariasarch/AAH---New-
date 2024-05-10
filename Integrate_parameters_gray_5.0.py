import sys
import time
import os
os.environ['PATH']+= 'C:\\Users\\heather\\.conda\\envs\\minian_ari\\Library\\bin'
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect

import numpy as np
import video_processing_12 as vp

sys.setrecursionlimit(10**6)

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import sys
import pandas as pd
from xarray import DataArray
import os.path
import rechunker
from uuid import uuid4
import VSC_Minian_demo_videos_v5 as VSC

# Handles communication to and from subprocesses such as Play_video and Save_changes_2_xarray
# and communication to and from user inputs from MainWindow
class Processing_HUB(QObject):
    updateFrame = pyqtSignal(QImage)
    updateProgressBar = pyqtSignal(int)
    
    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.video_path = None
        self.data_array = None
        self.seed_array=None
        self.play_video_instance = None
        self.current_function_index = 0
        self.deglow = False
        self.current_function_index=0
        self.current_function='None'
        self.decimal_tenths=np.array([9,13])
        self.decimal_hundreth=np.array([5,6,7,8,10,11,12,14,15])
        self.decimal_tenths_2=np.array([5,7,8,12,15])
        self.init_slider_val=0
        self.slider_value = 0
        self.slider_value_2=None
        self.slider_value_3=None
        self.slider_value_4=None
        self.slider_value_5=None
        self.seeds=None
        self.temp_seeds=None
        self.convert_2_contours=False
        self.A=None
        self.C=None
        self.pnr=None
        self.ks=None
        self.init_slider_val_2=None
        self.Truth_array=[]
        self.function_indices_alter_image=np.array([2,3]) # indices of functions that may alter individual frames 
        self.parameters=None
        self.visualize_seeds=False
        self.pause=False
        self.method=None
        self.both_xarrays=[6,7] # the function index where both arrays are needed
        self.White=(255,255,255)
        self.interval_sliders=np.array([10,11,14])
        self.col_names = ['Val1_name', 'Value_1', 'Val2_name', 'Value_2', 'Val3_name', 'Value_3', 'Val4_name', 'Value_4','Val5_name', 'Value_5']
        self.function_names = ['remove_glow', 'denoise', 'remove_background', 'seeds_init', 'pnr_refine', 'ks_refine', 'seeds_merge', 'initialize',
                                'init_merge', 'get_noise', 'first_spatial','first_temporal', 'first_merge', 'second_spatial', 'second_temporal']
        self.parameter_list = pd.DataFrame(columns=self.col_names, data=[['deglow', False, '', np.nan, '', np.nan, '', np.nan, '', np.nan],
                                            ['method', np.nan,'ksize', np.nan, '', np.nan, '', np.nan, '', np.nan],
                                            [ 'method', np.nan, 'wnd', np.nan,'', np.nan, '', np.nan, '', np.nan], 
                                                       ['method', np.nan,'wnd_size', np.nan, 'stp_size', np.nan, 'max_wnd', np.nan, 'dif_thres', np.nan],
                                                        ['noise_freq', np.nan,'thres', np.nan, '', np.nan, '', np.nan, '', np.nan],
                                                        ['sig', np.nan, '', np.nan, '', np.nan, '', np.nan, '', np.nan],
                                                        ['noise_freq', np.nan ,'thres_corr', np.nan, 'thres_dist', np.nan,'', np.nan, '', np.nan],
                                                        ['noise_freq', np.nan, 'thres_corr', np.nan,  'wnd', np.nan,'', np.nan, '', np.nan],
                                                        ['thres_corr', np.nan, '', np.nan, '', np.nan, '', np.nan, '', np.nan],
                                                        ['noise_range', np.nan, '', np.nan, '', np.nan, '', np.nan, '', np.nan],
                                                        [ 'sparse_penal', np.nan, 'dl_wnd', np.nan,'size_thres', np.nan, '', np.nan, '', np.nan],
                                                        ['noise_freq', np.nan, 'jac_thres', np.nan,'sparse_penal', np.nan, 'p', np.nan, 'add_lag', np.nan],
                                                        ['thres_corr', np.nan, '', np.nan, '', np.nan, '', np.nan, '', np.nan],
                                                        [ 'sparse_penal', np.nan,  'dl_wnd', np.nan,'size_thres', np.nan, '', np.nan, '', np.nan],
                                                        ['noise_freq', np.nan, 'jac_thres', np.nan,'sparse_penal', np.nan, 'p',np.nan, 'add_lag', np.nan],
                                                        ], index=self.function_names)

    def set_file_path(self, file_path):
        self.video_path = file_path
        vp.set_file_path(os.path.abspath(os.path.join(os.path.dirname(file_path), '..', 'Minian_saved_files'))) 
        # Need to have file saved as 'Minian_saved_files' for this to work
        # Possible alternative: vp.set_file_path(os.path.abspath(os.path.abspath(file_path+'{}'.format(input('Minian_saved_files')))

    def set_csv_file_path(self, csv_file_path):
        self.csv_file_path=csv_file_path

    def  set_folder_file_path(self,file_path_2):
        self.folder_file_path=file_path_2

    def call_VSC_minian(self):
        # VSC.run_minian(self.csv_file_path, self.folder_file_path) ### Function to be created that passes (parameter_path, video_folder_path)
        print('Running minian')

    def get_initial_video(self):
        if self.video_path is not None and len(self.video_path) > 0:
            self.data_array_orig, self.seed_array = self.load_video_data()
            self.data_array=self.data_array_orig
            if self.data_array is not None and len(self.data_array) > 0:
                self.play_video_instance = Play_video(self.data_array)
                self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                self.play_video_instance.start()
            print('Video started')
        else:
            print('Failed to get video path')

    def handle_update_frame(self, qimage):
        new_qimage = qimage  # Default to the original image
        if self.current_function_index == 2 or self.current_function_index == 3:
            new_qimage = self.frame_altering_function(frame=qimage)
        elif self.convert_2_contours and not (self.current_function_index == 0 or self.current_function_index == 1):
            frame = self.qimg_2_array(qimage)
            new_frame = self.convert_contours(frame)
            new_qimage = self.array_2_qimg(new_frame)
        elif self.current_function_index in [4,5,6]:
            # self.visualize_seeds=True
            self.convert_2_contours=False
            new_qimage = self.seed_altering_function(frame=qimage)
            # Check if new_qimage is None, and if so, assign a default QImage
        elif self.current_function_index ==7:
            self.visualize_seeds=False
        if new_qimage is None:
            width, height = qimage.width(), qimage.height()
            new_qimage = QImage(width, height, QImage.Format_Grayscale8)
            new_qimage.fill(Qt.black)  # Fill with black or any other appropriate default color
            print('Failed to get frame')

        self.updateFrame.emit(new_qimage)

    def handle_xarray_saved(self, data_array):
        self.data_array = data_array
        if self.data_array is not None and len(self.data_array) > 0:
            print('Xarray saved to Processing HUB')

    def handle_seed_array_saved(self, seed_array):
        self.seed_array = seed_array
        if self.seed_array is not None and len(self.seed_array) > 0:
            print('Xarray for seeds saved to Processing HUB')

    def load_video_data(self):
        initial_file_opening = Initial_file_opening(self.video_path)
        initial_file_opening.xarray_saved.connect(self.handle_xarray_saved)
        initial_file_opening.seeds_init_array.connect(self.handle_seed_array_saved)
        initial_file_opening.start()
        initial_file_opening.wait()
        return initial_file_opening.data_array, initial_file_opening.seeds_init_array

    def save_changes_2_xarray(self):
        if self.play_video_instance is not None:
            self.play_video_instance.stop_thread = True
        if self.data_array is not None and self.current_function_index<15:
            if len(self.data_array) > 0:
                self.apply_changes_2()
            else:
                print('Data array is empty')
        self.play()

    def play(self):
        if self.play_video_instance is not None and self.play_video_instance.isFinished:
            if self.data_array is not None and len(self.data_array) > 0:
                self.play_video_instance = Play_video(self.data_array)
                self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                self.play_video_instance.start()
        elif self.play_video_instance is not None and self.play_video_instance.isRunning():
            self.play_video_instance.requestInterruption()
            if self.data_array is not None and len(self.data_array) > 0:
                self.play_video_instance = Play_video(self.data_array)
                self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                self.play_video_instance.start()

    def apply_changes_2(self):  # 
        self.generate_parameter_array()
        self.add_parameters_2_csv()
        if self.current_function_index==2:
            self.deglow = False
        if self.current_function_index>=5:
            if self.seeds is not None:
                Seed_related_files={'Seeds':self.seeds}
                if self.pnr is not None:
                    Seed_related_files['pnr']=self.pnr
                if self.ks is not None:
                    Seed_related_files['ks']=self.ks
                if self.A is not None:
                    Seed_related_files['A']=self.A 
                if self.C is not None:
                    Seed_related_files['C']=self.C 
                if self.current_function_index in self.both_xarrays: # work arround for when both arrays are needed
                    Seed_related_files['data_xarray']=self.data_array 
                save_xarray_instance = Save_changes_2_xarray(self.seed_array, self.current_function_index, self.parameters, Seed_related_files)
            else:
                save_xarray_instance =Save_changes_2_xarray(self.seed_array, self.current_function_index, self.parameters)
        else:
            save_xarray_instance = Save_changes_2_xarray(self.data_array, self.current_function_index, self.parameters)
        save_xarray_instance.start()
        save_xarray_instance.wait()
        if save_xarray_instance.seeds is not None:
            self.seeds=save_xarray_instance.seeds
        if save_xarray_instance.seeds_pnr is not None:
            self.pnr=save_xarray_instance.seeds_pnr
        if save_xarray_instance.seeds_ks is not None:
            self.ks=save_xarray_instance.seeds_ks
        if save_xarray_instance.A is not None:
            self.A=save_xarray_instance.A
            print("A obtained by HUB")
        if save_xarray_instance.C is not None:
            self.C=save_xarray_instance.C
        if save_xarray_instance.data_array is not None:
            self.data_array = save_xarray_instance.data_array
        else:
            print('Failed to obtain xarray from: Save_changes_2_xarray')
        self.parameters=None
        self.slider_value=0
        self.slider_value_2=None
        self.slider_value_3=None
        self.slider_value_4=None
        self.slider_value_5=None
        self.methods=None
        return self.data_array
    
    def add_parameters_2_csv(self):
        if self.parameters is not None:
            funct_name = self.function_names[self.current_function_index - 1]
            if self.current_function_index == 1:
                self.parameter_list.loc[funct_name, self.col_names[1]] = self.deglow  # selects column name
            elif self.parameters != []:
                num_params = len(self.parameters)
                for inx in range(num_params):
                    col_index = 2 * inx + 1
                    if col_index < len(self.col_names):  # Ensure col_index is within bounds
                        if isinstance(self.parameters[inx], tuple):
                            # If the parameter is a tuple, extract its elements and assign them to consecutive columns
                            for i, val in enumerate(self.parameters[inx]):
                                self.parameter_list.loc[funct_name, self.col_names[col_index + i]] = val
                        else:
                            self.parameter_list.loc[funct_name, self.col_names[col_index]] = self.parameters[inx]
                    else:
                        print(f"Column index {col_index} out of bounds for function {funct_name}")
            print(self.parameter_list.loc[funct_name, :])
            print('Current function is ' + str(self.function_names[self.current_function_index]))

    def update_button_indx(self, button_index):
        self.current_function_index = button_index
        print('The button_index from progress_HUB is ' + str(self.current_function_index))

    def get_method(self, value):
        self.method=value
        print(value)

    def get_init_val(self,value_init):
        self.slider_value=value_init
        
    def get_init_val_2(self,value_init_2):
        self.slider_value_2=value_init_2

    def get_init_val_3(self,value_init_3):
        self.slider_value_3=value_init_3

    def get_init_val_4(self,value_init_4):
        self.slider_value_4=value_init_4

    def get_init_val_5(self,value_init_5):
        self.slider_value_5=value_init_5

    def get_init_method(self,method):
        self.method=method

    def remove_glow(self): ### Causes seeds init to crash for unknown reason
        if self.data_array is not None and len(self.data_array) > 0:
            print('deglow function activated')
            if self.deglow and self.play_video_instance is not None:
                self.play_video_instance.requestInterruption()
                # note 0.7 can be made into a slider at a later date if required
                self.data_array = self.data_array_orig.sel(frame=slice(None)) - (0.7)*self.data_array.min(dim='frame')  # applies deglow 
                self.play_video_instance.data_array = self.data_array
                self.play()     
        else:
            print('DataArray is empty')

    def glow_check(self):
        self.deglow = True
        print('self.deglow=True')
        self.remove_glow()

    def glow_unchecked(self):
        self.deglow = False
        print('self.deglow=False')
        self.play_video_instance.requestInterruption()
        self.data_array = self.data_array_orig
        self.play_video_instance.data_array = self.data_array
        self.play() 
    
    @pyqtSlot()
    def resume(self): # allows the frame to be paused and resume from last frame
        if self.play_video_instance:
            self.play_video_instance.resume_thread()
            self.pause=False

    @pyqtSlot()
    def stop_thread_play(self):
        if self.play_video_instance:
            self.play_video_instance.pause_thread()
            self.pause=True       
    
#### Get current function:
    def frame_altering_function(self,frame):
        if self.current_function_index == 2:
                return self.denoise(frame)
     
        elif self.current_function_index == 3:
                return self.remove_background(frame)

    ### not currently being used but will probably select a subset of the xarray based on current frame
    def seed_altering_function(self,frame):
        if self.visualize_seeds==True:
            if self.current_function_index == 4:
                self.seeds_init_wrapper(frame)
            if self.current_function_index == 5:
                self.pnr_refine_wrapper(frame)
            elif self.current_function_index == 6:
                self.ks_refine_wrapper(frame)
            if self.temp_seeds is not None:
                return self.show_seeds(frame)
        else:
                return frame

### adjusts parameters to appropriate scale
    def parameters_for_current_functions(self):
        if self.current_function_index==5:
            self.method=None
        if self.data_array is not None and len(self.data_array) > 0:
            print('data_array obtained by current function')
            if self.current_function_index in self.decimal_tenths:
                self.slider_value= (self.slider_value/10)
            if self.current_function_index in self.decimal_hundreth:
                self.slider_value = (self.slider_value / 100)
            if self.current_function_index in self.decimal_tenths_2:
                self.slider_value_2 = (self.slider_value_2 / 10)

    def generate_parameter_array(self):
        self.parameters_for_current_functions()
        self.parameters = []  # Initialize the parameters list as an empty list
        if self.method is not None and not (self.current_function_index in self.interval_sliders):
            self.parameters.append(str(self.method))  # Convert method to string and append
        if self.slider_value != 0:
            self.parameters.append(str(self.slider_value))  # Convert slider_value to string and append
        if self.slider_value_2 is not None:
            self.parameters.append(str(self.slider_value_2))  # Convert slider_value_2 to string and append
        if self.slider_value_3 is not None:
            self.parameters.append(str(self.slider_value_3))  # Convert slider_value_3 to string and append
        if self.slider_value_4 is not None:
            self.parameters.append(str(self.slider_value_4))  # Convert slider_value_4 to string and append
        if self.slider_value_5 is not None:
            self.parameters.append(str(self.slider_value_5))  # Convert slider_value_5 to string and append

        self.parameters = ','.join(self.parameters)  # Join the list elements with a comma
        if self.parameters:
            self.parameters = self.parameters.split(',')  # Split the joined string by commas
            for i, parameter in enumerate(self.parameters):
                try:
                    self.parameters[i] = float(parameter)  # Convert the number to a float
                except ValueError:
                    pass  # Ignore if conversion to float fails (e.g., for non-numeric values)
            if self.current_function_index in self.interval_sliders and self.method is not None:
                new_interval = (float(self.parameters[-1]), self.method)
                self.parameters[-1] = new_interval  # Convert interval_sliders to string and append
        print(self.parameters)
        
    def denoise(self, frame):
        frame=self.qimg_2_array(frame)
        if self.slider_value==0:
            kernel_size = 5
        else:
            kernel_size = int(self.slider_value)
        if kernel_size % 2 == 0:
                kernel_size += 1
        new_frame=vp.denoise_by_frame(frame, method='gaussian', ksize=kernel_size)
        if self.convert_2_contours==True:
                self.new_frame=new_frame
                new_frame=self.convert_contours(self.new_frame)
        return self.array_2_qimg(new_frame)
    
    def remove_background(self,frame):
        frame=self.qimg_2_array(frame)
        self.kernel_size=self.slider_value
        new_frame=vp.remove_background_by_frame(frame, method=self.method, kernel_size=self.kernel_size)
        return self.array_2_qimg(new_frame)

# following functions must be updated to use 
    
    def show_seeds(self, frame):
        if self.seeds is not None and len(self.seeds) > 0:
            img = self.qimg_2_array(frame)
            for seed in self.seeds:
                center = (int(seed[1]), int(seed[0]))  # Assuming seed is a tuple (y, x)
                radius = 30  # Adjust the radius as needed
                cv2.circle(img, center, radius, self.White, -1)  # Draw a filled circle
            return self.array_2_qimg(img)
        else:
            return frame  # Return the original frame if no seeds are available
    
    ### need to update functions from here down to align with VSC_Minian
    def seeds_init_wrapper(self, frame):
            print('first seeds function called')
            if self.method is None:
                self.method='rolling'
            if   self.slider_value is None:
                self.slider_value_2=500
            if   self.slider_value_3 is None:
                self.slider_value_3=200
            if self.slider_value_4 is None:
                self.slider_value_4= 15
            if self.slider_value_5 is None:
                self.slider_value_5= 3
            new_frame=self.qimg_2_array(frame)
            self.temp_seeds = vp.seeds_init(new_frame ,wnd_size=int(self.slider_value_2), method=self.method,stp_size=int(self.slider_value_3),
                                                        max_wnd=int(self.slider_value_4), diff_thres=int(self.slider_value_5))
            return frame

    
    ### Contour function for visualization
    def convert_contours(self, frame):
        if frame is not None and len(self.data_array) > 0:
            self.stop_thread_play()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sigma_1=0.01
            med_gray = np.median(frame)
            lower = int(max(0, (1.0 - sigma_1) * med_gray))
            upper = int(min(255, (1.0 + sigma_1) * med_gray))
            gX = cv2.Sobel(frame, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            gY = cv2.Sobel(frame, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
            # the gradient magnitude images are now of the floating point data
            # type, so we need to take care to convert them back a to unsigned
            # 8-bit integer representation so other OpenCV functions can operate
            # on them and visualize them
            gX = cv2.convertScaleAbs(gX)
            gY = cv2.convertScaleAbs(gY)
            # combine the gradient representations into a single image
            combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
            ret, thresh_frame = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY)
            contours, her = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            self.whiteFrame = 255 * np.ones((frame.shape[0],frame.shape[1],3), np.uint8)
            img = cv2.drawContours(self.whiteFrame, contours, -1,(127,127, 127), 1)   
        else:
            img=frame
            print('Failed to convert frame to contour')
        return img
    
    def qimg_2_array(self, frame):
        if isinstance(frame, np.ndarray): # Check if frame is already a numpy array
            return frame 
        img = frame.convertToFormat(QImage.Format_Grayscale8)  # Convert image to RGB format
        width = img.width()
        height = img.height()
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))  # Reshape 
        # images_array.astype(np.uint8)
        return arr

    def array_2_qimg(self,frame):
        if len(frame.shape)==3:
            height, width, channel = frame.shape
        if len(frame.shape)==2:
            height, width = frame.shape
        if frame is not None:
            q_img = QImage(frame.data, width, height, frame.strides[0], QImage.Format_Grayscale8)
        return q_img
    
    def contour_check(self):
        self.convert_2_contours=True

    def contour_unchecked(self):
        self.convert_2_contours=False

    ### self defined functions for passing values
    def update_button_indx(self, button_index):
        self.current_function_index= button_index
        self.allow_funct=False
        # print('The button_index from Processing_HUB is '+str(self.current_function_index))

    def update_data_array(self, data_array):
        self.data_array=data_array

    def temp_mod_frame(self, value): # takes value from on__change and adjusts value for functions
        # call function based on passed value
        self.slider_value = value
        if self.convert_2_contours==True and self.current_function_index==2:
           self.denoise(self.new_frame)
        
    def temp_mod_frame_2(self, value_2):
        self.slider_value_2=value_2

    def temp_mod_frame_3(self, value_3):
        self.slider_value_3=value_3

    def temp_mod_frame_4(self, value_4):
        self.slider_value_4=value_4

    def temp_mod_frame_5(self, value_5):
        self.slider_value_5=value_5

    def allow_xarray_functions(self):
            self.allow_funct=True

    def get_xarray(self):  # Returns the current xarray for saving
        return self.data_array
    
    def get_parameters(self):
        return self.parameter_list

class Initial_file_opening(QThread):
    xarray_saved = pyqtSignal(DataArray)
    seeds_init_array = pyqtSignal(DataArray)

    def __init__(self, file_path):
        super(Initial_file_opening, self).__init__()
        self.first_play = 0
        self.limit = 10 ** 6
        self.video_path = file_path

    def run(self):
        sys.setrecursionlimit(self.limit)
        if self.video_path:
            if self.first_play == 0:
                self.load_avi_perframe(self.video_path)
                self.frame_array_2_xarray()
                print('xarray_generated_from_file')
                self.first_play += 1
        else:
            print('No video file selected. Click: Upload Video: to select .avi video file')

    def load_avi_perframe(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        channel = 3  # 1 for black and white, 3 for red-blue-green
        frame_array = np.empty((frame_number, height, width), dtype=np.uint8)
        for i in range(frame_number):
            ret, frame = cap.read()
            if ret:
                frame_conv = np.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), axis=0)
                frame_array[i] = frame_conv
            else:
                break
        self.frame_array = np.float64(frame_array)
        self.stats = [frame_number, height, width]

    def frame_array_2_xarray(self):
        self.data_array = DataArray(
            self.frame_array,
            dims=["frame", "height", "width"],
            coords={
                "frame": np.arange(self.stats[0]),
                "height": np.arange(self.stats[1]),
                "width": np.arange(self.stats[2]),
                
            },
        )
        self.data_array.name='Insert_name_here'
        self.get_chunk()
        self.data_array= self.data_array.chunk({"frame": self.chunk_comp["frame"], "height": -1, "width": -1})
        seeds_init_array=self.data_array.chunk({"frame": -1, "height": self.chunk_comp["height"], "width": self.chunk_comp["width"]})
        self.xarray_saved.emit(self.data_array)
        self.seeds_init_array.emit(seeds_init_array)
    
    def get_chunk(self):
        if self.data_array is not None and len(self.data_array) > 0:
            self.chunk_comp, self.chunk_store = vp.get_optimal_chk(self.data_array, dtype=float)

# Plays video of Passed self.data_array and updates frame based on current function
class Play_video(QThread):
    updateFrame = pyqtSignal(QImage)

    def __init__(self, data_array):
        super(Play_video, self).__init__()
        self.data_array = data_array
        self.frame_rate = 5
        self.stop_thread = False
        self.convert_2_contours = False
        self.play_index=0 # For get_video progress

    def run(self):
        self.ThreadActive=True
        self.get_video()

    def get_video(self):
        self.stop_thread = False
        time_frame = 1 / self.frame_rate
        if self.data_array is not None:
            while self.ThreadActive:
                for i in range(len(self.data_array)):
                    if not self.stop_thread:
                        img = self.data_array[i].values
                        if img is not None:
                            height, width = img.shape
                            img = np.uint8(img) 
                            height, width = img.shape
                            q_img = QImage(img, width, height, width, QImage.Format_Grayscale8)
                            self.updateFrame.emit(q_img)
                            time.sleep(time_frame)
                        else:
                            print('Failed to get frame')
                            continue   
            #else:
            while self.stop_thread:
                        img = self.data_array[i].values
                        time.sleep(1*time_frame) # update slowed for compuatation
            else:
                            print('Failed to get frame')   
                                    

    def pause_thread(self):
        self.ThreadActive=False
        self.stop_thread=True

    def resume_thread(self):
        self.stop_thread=False
        self.ThreadActive=True

# Applies Get_current_function to self.data_array and saves to Processing_HUB using
class Save_changes_2_xarray(QThread):
        updateSeeds=pyqtSignal(pd.DataFrame)
        def __init__(self,data_array, function_index=0, parameters=None, Seed_related_files=None):
            super(Save_changes_2_xarray, self).__init__()
            self.data_array=data_array
            self.function_index=function_index
            self.parameters=parameters
            self.frame_index = 0
            self.seeds=None
            self.A=None
            self.C=None
            self.seeds_pnr=None
            self.seeds_ks=None
            self.value=None
            self.value_2=None
            self.value_3=None
            self.value_4=None
            self.value_5=None
            if Seed_related_files is not None:
                self.seeds=Seed_related_files['Seeds']
                self.seed_array=self.data_array
                if hasattr(Seed_related_files, 'pnr'):
                    self.seeds_pnr=Seed_related_files['pnr']
                if hasattr(Seed_related_files, 'ks'):
                    self.seeds_ks=Seed_related_files['ks']
                if hasattr(Seed_related_files, 'A'):
                    self.A=Seed_related_files['A']
                print('seeds from A have been passed to save_changes')
                if hasattr(Seed_related_files, 'C'):
                    self.C=Seed_related_files['C']
                if hasattr(Seed_related_files, 'data_xarray'):
                    self.data_array=Seed_related_files['data_xarray']

        def run(self):
            if self.parameters is not None and self.parameters !=[]:
                self.parameters_unpacking()
            if self.data_array is not None and len(self.data_array) > 0:
               self.apply_changes()
            else:
                print('Failed to pass data_array to Save_changes_2_xarray')

        def apply_changes(self):
            self.current_function(self.data_array[0].values)
            print('Changes saved for ' + str(self.function_index))
            return self.data_array
        
        def perframe_processing(self):
                array_len = len(self.data_array)
                for indx in range(array_len):
                    self.frame_index = indx
                    frame = self.data_array[indx].values
                    processed_frame = self.function(frame)  # Obtain the processed frame
                    if processed_frame is not None:
                        self.data_array[indx].values = processed_frame
                    elif self.function_index>10:
                        print('Failed to convert frame ' + str(indx))
                self.frame_index = 0
                return self.data_array
        
        def parameters_unpacking(self):
            if self.parameters is not None and len(self.parameters)!=0:
                print(self.parameters)
                self.value=self.parameters[0]
                print('1st passed value is '+ str(self.value))
            else:
                print('Parameters not obtained by Save_changes_2_xarray')
            if len(self.parameters)==2:
                self.value_2=self.parameters[1] 
            if len(self.parameters)==3:
                self.value_3=self.parameters[2]
            if len(self.parameters)==4:
                self.value_4=self.parameters[3]
            if len(self.parameters)==5:
                self.value_5=self.parameters[4]
                
        def current_function(self, frame):
            # self.current_function_array[int(self.function_index)](frame)
            if self.function_index==0:
                return frame 
            if self.function_index==1:
                return frame
            elif self.function_index == 2:
                self.denoise()
                return frame
            elif self.function_index == 3:
                self.remove_background()
                self.estimate_motion()
                return frame
            elif self.function_index == 4:
                self.seeds_init_wrapper()  # Assuming this updates internal state and doesn't modify the frame directly
                return frame
            elif self.function_index == 5:
                return self.pnr_refine_wrapper(frame)
            elif self.function_index == 6:
                return self.ks_refine_wrapper(frame)
            elif self.function_index == 7:
                self.seeds_merge_wrapper(frame)  # Again, assuming updates internal state
                return frame
            elif self.function_index == 8:
                return self.initA_wrapper(frame)
            elif self.function_index == 9:
                self.unit_merge_wrapper(frame)  # Assuming updates internal state
                return frame
            elif self.function_index == 10:
                return self.get_noise_fft_wrapper(frame)
            elif self.function_index == 11:
                return self.update_spatial_wrapper()
            elif self.function_index == 12:
                return self.update_temporal_wrapper()
            elif self.function_index == 13:
                self.unit_merge_wrapper(frame)  # Assuming updates internal state
                return frame
            elif self.function_index == 14:
                return self.update_spatial_wrapper()
            elif self.function_index == 15:
                return self.update_temporal_wrapper()
            else:
                print('All functions have been called. Please save parameters and/or video in a seperate folder for reference')
                return frame
            
        # def function(self,frame):
           # if  self.function_index ==3:
               # return self.estimate_motion(frame)

        def return_self(self,frame):
            return frame
        
        def denoise(self):
            if self.value_2 == 0:
                self.value_2 = 5
            kernel_size = int(self.value_2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if self.value is None:
                self.value = 'gaussian'
                print('Method not passed to remove background')
            denoise_parameters = {'method': self.value, 'ksize': kernel_size}
            self.data_array = vp.denoise(self.data_array, **denoise_parameters)
    
        def remove_background(self):
            self.kernel_size=int(self.value_2)
            if self.value is None:
                self.value='uniform'
                print('method not passed to remove background')
            self.removed_bck=vp.remove_background(self.data_array, method=self.value, wnd=self.kernel_size) #applies remove background

        def estimate_motion(self):
                self.previous_frame = self.removed_bck[self.frame_index-1].values
                param_estimate_motion = {"dim": "frame"}
                self.motion_vector = vp.estimate_motion(self.removed_bck, **param_estimate_motion)
                chk,_=vp.get_optimal_chk(self.data_array,dtype=float)
                self.motion_vector=self.motion_vector.chunk({"frame": chk["frame"]})
                self.data_array= vp.apply_transform(self.removed_bck, self.motion_vector) # applies motion transform

### need to update functions from here down to align with VSC_Minian
        def seeds_init_wrapper(self):
            if self.value is None:
                self.value='rolling'
            if   self.value_2 is None or not self.value_2.isDigit:
                self.value_2=500
            if   self.value_3 is None:
                self.value_3=200
            if self.value_4 is None:
                self.value_4= 15
            if self.value_5 is None:
                self.value_5= 3
            #  self.data_array = self.data_array.sel(frame=slice(None)) 
            self.seeds = vp.seeds_init(self.data_array,wnd_size=int(self.value_2), method=self.value,stp_size=int(self.value_3),
                                                        max_wnd=int(self.value_4), diff_thres=int(self.value_5))
            if self.seeds is not None:
                ('After seeds initiation there are this many seeds: ' +str(len(self.seeds)))
            else:
                print('No seeds where generated')
    
        def pnr_refine_wrapper(self, frame): 
            if hasattr(self, 'seeds'):
                seeds, pnr, gmm = vp.pnr_refine(self.seed_array, self.seeds, self.value, self.value_2)
                if pnr is not None:
                    self.seeds_pnr = pnr
                    self.seeds=seeds
                    print('After PNR refinement there are this many seeds: ' + str(len(self.seeds)),self.seeds )
                else:
                    print('No refined seeds found after PNR refinement.')
            else:
                print("No frame or seeds available for PNR refinement.")
            return frame

        def ks_refine_wrapper(self, frame):
            if hasattr(self, 'seeds'):
                self.seeds_ks = vp.ks_refine(self.seed_array, self.seeds, self.value)
                print('After KS refinement there are this many seeds: ' +str(len(self.seeds)))
                print(self.seeds)
            else:
                print("No frame or seeds available for KS refinement.")
            return frame
    
        def seeds_merge_wrapper(self, frame): # Need to pass both self.seeds_ks and self.seeds_pnr
            if hasattr(self, 'seeds'):
                        merge_parameters = {'thres_dist':self.value_3, 'thres_corr': self.value_2,'noise_freq':self.value}
                        seeds_final=self.seeds[self.seeds["mask_pnr"] & self.seeds["mask_ks"]].reset_index(drop=True)
                        self.max_proj=self.data_array.max("frame")
                        self.seeds = vp.seeds_merge(self.seed_array,self.max_proj, seeds_final, **merge_parameters)
                        print(self.seeds)
            else:
                print('No seeds to merge')
            return frame

        def initA_wrapper(self, frame):
            print(type(self.seeds))
            if  hasattr(self, 'seeds'):
                self.A = vp.initA(self.seed_array, self.seeds[self.seeds["mask_mrg"]], self.value_2, int(self.value_3), self.value) 
                self.initC_wrapper(frame)
            return frame

        def initC_wrapper(self, frame):
            if  hasattr(self,'A'):
                self.C = vp.initC(self.data_array, self.A)
            return frame

        def unit_merge_wrapper(self, frame): # May have to run for both initC and initA
            print(type(self.A), self.A, self.seeds)
            if hasattr(self, 'A'):
                self.A, self.C = vp.unit_merge(self.A,self.C,{'thres_corr': self.value})
            return frame

        def get_noise_fft_wrapper(self, frame):
                self.noise_fft = vp.get_noise_fft(self.data_array, self.value)
                return frame
            
### Dont need to pass the following functions frames therefore may deal with them seperately without calling loop
        def update_spatial_wrapper(self):
            if hasattr(self, 'A') and self.A:
                parameters={'dl_wnd':self.value_2,'sparse_penal': self.value, 'size_thres': self.value_3}
                self.A = vp.update_spatial(self.data_array, self.A, self.noise_fft, parameters)
            else:
                print('No spatial component to update')     

        def update_temporal_wrapper(self):
            parameters={ "noise_freq": self.value,
                        "sparse_penal": self.value_3,
                        "p": self.value_4,
                        "add_lag": self.value_5,
                        "jac_thres": self.value_2,}
            if self.C is not None:
                self.C = vp.update_temporal(self.A, self.C, parameters)
            else:
                print('No temporal component to update')

# MainWindow handles all displays/interactions
class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.Button_name = [
            '', 'Deglow', 'Denoise', 'Remove Background',
            'Seeds Init', 'PNR Refine', 'KS Refine', 'Seeds Merge', 
            'Init A', 'Unit Merge', 'Get Noise FFT', 
            'First Spatial Update', 'First Temporal Update',
            'First Merge', 'Second Spatial Update', 'Second Temporal Update','Final merge', 'Save outputs'
        ]

        self.slider_name = [
            ['None'],  # Opening video
            ['None'],  # Deglow
            ['Kernel Size'],  # Denoise
            ['Kernel Size'],  # Remove Background
            ['Window size', 'Step size', 'Maximum window', 'Difference threshold'],  # Seeds Init
            ['Noise frequency', 'Threshold'],  # PNR Refine
            ['Significance Level'],  # KS Refine
            ['Noise frequency','Threshold correction', 'Threshold distance'],  # Seeds Merge (not applicable, placeholder)
            ['Noise frequency','Threshold correction','Spatial radius'],  # Init A +Init C
            ['Threshold correction'],  # Unit Merge 
            ['Noise threshold: Lower bound'],  # Get Noise FFT 
            ['Sparse penalty', 'Window size dilation', 'Size threshold: Lower bound'],  # Update Spatial
            ['Noise frequency', 'Jacobian threshold','Sparce penalty', 'AR order of P', 'Added lag'],  # Update Temporal
            ['Threshold correction'],  # first spacial,temporal merge
            ['Sparse penalty', 'Window size dilation', 'Size threshold: Lower bound'],  # Update Spatial
            ['Noise frequency', 'Jacobian threshold','Sparce penalty', 'AR order of P', 'Added lag'],  # Update Temporal
            ['None'], # second merge
            ['None'],
        ]

        self.Min_slider = [
            [0],  # Get optimal chunk
            [0],  # Deglow 
            [1],  # Denoise
            [1],  # Remove Background
            [100,50, 5,1] ,  # Seeds Init (e.g., threshold min)
            [1,5],  # PNR Refine (e.g., min noise frequency)    
            [1],  # KS Refine (e.g., significance level min)
            [2,2,5],  # Seeds Merge (not applicable, placeholder)
            [2,2,5],  # Init A (e.g., spatial radius min)
            [2],  # Unit Merge (not applicable, placeholder)
            [2],  # Get Noise FFT (not applicable, placeholder)
            [1,5,15],  # Update Spatial (e.g., update factor min) 
            [2,1,1,1, 10],  # Update Temporal
            [2],  # Update Merge
            [1,5,15],  # Update Spatial (e.g., update factor min) 
            [2,1,1,1,10],  # Update Temporal
            [0],
            [0]
        ]

        self.Max_slider = [
            [0],  # Get optimal chunk
            [0],  # Deglow 
            [10],  # Denoise
            [20],  # Remove Background
            [1100,500, 25,10] ,  # Seeds Init (e.g., threshold min)
            [80,20],  # PNR Refine (e.g., min noise frequency)    
            [20],  # KS Refine (e.g., significance level min)
            [10,20,50],  # Seeds Merge (not applicable, placeholder)
            [10,20,50],  # Init A (e.g., spatial radius min)
            [10],  # Unit Merge (not applicable, placeholder)
            [10],  # Get Noise FFT (not applicable, placeholder)
            [10,50,50],  # Update Spatial (e.g., update factor min) 
            [10,10,5,5, 50],  # Update Temporal
            [10],  # Merge
            [10,50,50],  # Update Spatial (e.g., update factor min) 
            [10,10,5,5,50],  # Update Temporal
            [0],
            [0]
        ]
 ### KS Refine, and all 'Update' functions need divide by 10
        self.init_slider = [
            [0],  # Get optimal chunk
            [0],  # Deglow 
            [7],  # Denoise
            [15],  # Remove Background
            [500,200, 10,2] ,  # Seeds Init (e.g., threshold min)
            [2,10],  # PNR Refine (e.g., min noise frequency)    
            [5],  # KS Refine (e.g., significance level min)
            [6,8,10],  # Seeds Merge (not applicable, placeholder)
            [6,8,10],  # Init A (e.g., spatial radius min)
            [8],  # Unit Merge (not applicable, placeholder)
            [5],  # Get Noise FFT (not applicable, placeholder)
            [1,10,25],  # Update Spatial (e.g., update factor min) 
            [6,2,1,1, 20],  # Update Temporal
            [8],  # Merge
            [1,10,25],  # Update Spatial (e.g., update factor min) 
            [6,2,1,1, 20],  # Update Temporal
            [0],
            [0]
        ]
        self.current_control = 0 
        self.intern_indx=None
        self.current_layo=None
        self.check_counter=0
        self.glow_check_counter=0
        self.current_widget = [
            'chnk_widget', 'remove_glow_widget', 'denoise_widget', 'remove_bck_widget',
            'seeds_init_widget', 'pnr_refine_widget', 'ks_refine_widget', 'seeds_merge_widget', 
            'initA_widget', 'unit_merge_widget', 'get_noise_fft_widget', 
            'update_spatial_widget', 'update_temporal_widget', 'first_merge_widget','update_spatial_widget_2', 'update_temporal_widget_2','Final_merge', 'Save_stuff'
        ]
        self.current_layout = [
            'chnk_layout', 'deglow_widget', 'denoise_layout', 'remove_bck_layout',  
            'seeds_init_layout', 'pnr_refine_layout', 'ks_refine_layout', 'seeds_merge_layout', 
            'initA_layout', 'unit_merge_layout', 'get_noise_fft_layout', 
            'update_spatial_layout', 'update_temporal_layout','first_merge_layout','update_spatial_layout2', 'update_temporal_layout2','Final_merge_layout', 'Save_stuff_layout'
        ]
        self.Method_drop_down=[["gaussian","anisotropic",'median', "bilateral"], ['uniform','tophat'], ['rolling', "random"]]
        self.numb_upper_bound=[['0.2','0.4','0.6','0.8','0.9'], ['None', '100', '300', '500', '700', '1000'], ['None', '100', '300', '500', '700', '1000']]
        self.numb_upper_bound_passing=[[0.2,0.4,0.6,0.8,0.9], [None, 100, 300, 500, 700, 1000], [None, 100, 300, 500, 700, 1000]]
        self.decimal_tenths=np.array([6,9,13])
        self.decimal_hundreth=np.array([5,7,8,10,11,12,14,15])
        self.decimal_tenths_2=np.array([5,6,7,8,12,15])
        self.method_index=[2,3,4] # gives indices for method selection
        self.bound_index=[10,11,14] # gives indices where there is an upper bound
        self.inter_counter=0 # initiates inner counter at zero
        self.inter_counter_2=0
        self.comboBox =None
        self.comboBox_2=None
  
        
        self.setWindowTitle("xarray_player")
        self.setGeometry(0, 0, 800, 500)
        outerLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        Button_layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.upload_button = QPushButton("Upload Video to select Parameters", self)
        self.upload_button.clicked.connect(self.open_file_dialog)
        self.csv_button = QPushButton("Upload Parameters to apply to all videos", self)
        self.csv_button.clicked.connect(self.open_csv_file)
        self.button_load_videos = QPushButton("Select folder of videos to upload", self)
        self.button_load_videos.clicked.connect(self.pass_video_folder)
        self.button_load_videos.setVisible(False)
        self.button_minian= QPushButton("Run minian", self)
        self.button_minian.clicked.connect(self.run_minian)
        self.button_minian.setVisible(False)

        self.checkBox = QCheckBox('Contours', self)
        self.checkBox.move(680, 400)
        self.checkBox.setVisible(False)
        self.checkBox.stateChanged.connect(self.countour_checkbox)
        self.button_play = QPushButton("Start")
        self.button_play.clicked.connect(self.start_thread)
        self.button_stop = QPushButton("Pause")
        self.button_stop.clicked.connect(self.stop_thread)
        self.button_restart = QPushButton("Replay Video")
        self.button_restart.clicked.connect(self.replay_thread)
        self.glow_checkBox = QCheckBox('Deglow', self)
        self.glow_checkBox.move(680, 460)
        self.glow_checkBox.setVisible(False)
        self.glow_checkBox.stateChanged.connect(self.deglow_checkbox)

        # Current control set index and Next Button setup
        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.next_control_set)
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)

        # Save video
        self.save_video_button = QPushButton("Save Video", self)
        self.save_video_button.setVisible(False)
        self.save_video_button.clicked.connect(self.save_xarray)  

                        # Save_parameters
        self.save_param_button = QPushButton("Save Parameters", self)
        self.save_param_button.setVisible(False)
        self.save_param_button.clicked.connect(self.save_parameters)         


        topLayout.addWidget(self.progress)
        topLayout.addWidget(self.label) 
        topLayout.addWidget(self.csv_button)
        topLayout.addWidget(self.button_load_videos)
        topLayout.addWidget(self.button_minian)       
        topLayout.addWidget(self.upload_button)

        # Stacked Widget for switching between control sets
        self.controlStack = QStackedWidget(self)
        topLayout.addWidget(self.controlStack)

        for i in range(len(self.current_widget)):
            self.current_widget[i]=QWidget()
            self.current_layout[i]=QVBoxLayout(self.current_widget[i])
        self.min_widget=QWidget()
        self.thresh_widget=QWidget()
        self.current_widget_2=[self.min_widget,self.thresh_widget]

        Button_layout.addWidget(self.button_restart, 2)
        Button_layout.addWidget(self.button_play,2)
        Button_layout.addWidget(self.button_stop,2)
        Button_layout.addWidget(self.save_video_button,2)
        Button_layout.addWidget(self.save_param_button,2)

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(Button_layout)
        
        if self.current_layo != None:
            outerLayout.addLayout(self.current_layo)
        self.setLayout(outerLayout)
    
        # Set the window's main layout

        self.thread= Processing_HUB()
        self.thread.updateFrame.connect(lambda image: self.displayFrame(image))
        self.thread.updateProgressBar.connect(lambda int: self.updateProgress(int))

#### Need break function for index over 17 ####
## Initiating controls for MiniAM
    # Next
    def next_control_set(self):
        self.next_btn.setVisible(False)
        self.progress.setValue(0)
        self.save_changes()
        self.current_control += 1 
        self.update_button_index()
        if self.current_control==1:
            self.glow_checkBox.setVisible(True)
        if self.current_control==2:
            self.glow_checkBox.setVisible(False)
            self.checkBox.setVisible(True)
        if self.current_control==3:
            self.checkBox.setVisible(False)
            self.save_video_button.setVisible(True) # Putting this here for now but might move
            self.save_param_button.setVisible(True)
        if self.current_control==5:
            self.comboBox.setVisible(False)
        if self.current_control >= 16:
            self.current_control=16 # when we finish we might replace this with a save button or something
        self.controlStack.setCurrentIndex(self.current_control)
        self.init_new_widget(self.current_control)
        self.send_init_slider_val()
        
    def init_new_widget(self, cur_index):
        if cur_index>=2:
            self.controlStack.removeWidget(self.current_widget[cur_index-1])
            self.current_layo.removeWidget(self.current_widget[cur_index-1])
        if 3>cur_index-self.Button_name.index('Seeds Init')>=1:
            self.controlStack.removeWidget(self.current_widget_2[cur_index-self.Button_name.index('Seeds Init')-1])
            self.current_layo.removeWidget(self.current_widget_2[cur_index-self.Button_name.index('Seeds Init')-1])

        if cur_index in self.method_index or cur_index == 5:
            if self.comboBox is not None:
                self.comboBox.setVisible(False)
                self.comboBox.deleteLater()
                self.comboBox = None

        if cur_index in self.method_index:
            self.comboBox = QComboBox(self)
            self.comboBox.move(680, 500)
            method_index = self.method_index.index(cur_index)  # Get the index of the current method drop-down
            self.comboBox.addItems(self.Method_drop_down[method_index])  # Use the correct method drop-down options
            Box_label = QLabel('Method', self.current_widget[cur_index])
            Box_label.move(600, 500)
            self.current_layo.addWidget(Box_label)
            slotLambda = lambda: self.current_drop_changed_lambda(
                self.Method_drop_down[method_index][self.comboBox.currentIndex()])
            self.comboBox.currentIndexChanged.connect(slotLambda)
            self.comboBox.setVisible(True)

        if cur_index in self.bound_index or cur_index in [12,15]:
            if self.comboBox_2 is not None:
                self.comboBox_2.setVisible(False)
                self.comboBox_2.deleteLater()
                self.comboBox_2 = None
                    
        if cur_index in self.bound_index:
            self.comboBox_2 = QComboBox(self)
            self.comboBox_2.move(680, 400)
            bound_index = self.bound_index.index(cur_index)
            self.comboBox_2.addItems(self.numb_upper_bound[bound_index])
            Box_label = QLabel('Upper bound', self.current_widget[cur_index])
            Box_label.move(680, 320)
            self.current_layo.addWidget(Box_label)
            slotLambda = lambda: self.current_drop_changed_lambda(self.numb_upper_bound_passing[bound_index][self.comboBox_2.currentIndex()])
            self.comboBox_2.currentIndexChanged.connect(slotLambda)  
            self.comboBox_2.setVisible(True) 
    

        self.progress.setVisible(False)
        self.current_layo=self.current_layout[cur_index]
        self.current_function_Label = QLabel('{}'.format(self.Button_name[cur_index]), self.current_widget[cur_index])
        self.current_layo.addWidget(self.current_function_Label)

        if self.slider_name[cur_index][0] != 'None':
            if cur_index in self.decimal_tenths:
                initial_slider=(self.init_slider[cur_index][0]/10)
                self.current_label = QLabel(self.slider_name[cur_index][0] + ': ' + str(initial_slider), self.current_widget[cur_index])
            elif cur_index in self.decimal_hundreth:
                initial_slider=(self.init_slider[cur_index][0]/100)
                self.current_label = QLabel(self.slider_name[cur_index][0] + ': ' + str(initial_slider), self.current_widget[cur_index])
            else:
                self.current_label = QLabel(self.slider_name[cur_index][0] + ': ' + str(self.init_slider[cur_index][0]), self.current_widget[cur_index])
            self.current_layo.addWidget(self.current_label) # Add label for displaying slider value
            self.current_slider = QSlider(Qt.Horizontal, self)
            self.current_slider.valueChanged[int].connect(self.on_slider_change)
            self.current_slider.setMinimum(self.Min_slider[cur_index][0])
            self.current_slider.setTickInterval(10)
            self.current_slider.setMaximum(self.Max_slider[cur_index][0])
            self.current_slider.setValue(self.init_slider[cur_index][0])
            self.current_layo.addWidget(self.current_slider)
            self.current_slider.setEnabled(True)
            self.current_label.setEnabled(True)
        self.controlStack.addWidget(self.current_widget[cur_index])

        if len(self.slider_name[cur_index])>1:
            if cur_index in self.decimal_tenths_2:
                initial_slider=(self.init_slider[cur_index][1]/10)
                self.current_label = QLabel(self.slider_name[cur_index][1] + ': ' + str(initial_slider), self.current_widget[cur_index])
            else:
                self.current_label_2 = QLabel(self.slider_name[cur_index][1] + ': ' + str(self.init_slider[cur_index][1]), self.current_widget[cur_index])
            self.current_layo.addWidget(self.current_label_2) # Add label for displaying slider value
            self.current_slider_2 = QSlider(Qt.Horizontal, self)
            self.current_slider_2.valueChanged[int].connect(self.on_slider_change_2)
            self.current_slider_2.setMinimum(self.Min_slider[cur_index][1])
            self.current_slider_2.setMaximum(self.Max_slider[cur_index][1])
            self.current_slider_2.setTickInterval(10)
            self.current_slider_2.setValue(self.init_slider[cur_index][1])
            self.current_layo.addWidget(self.current_slider_2)
            self.controlStack.addWidget(self.current_widget[cur_index])

        if len(self.slider_name[cur_index])>2:
            self.current_label_3 = QLabel(self.slider_name[cur_index][2] + ': ' + str(self.init_slider[cur_index][2]), self.current_widget[cur_index])
            self.current_layo.addWidget(self.current_label_3) # Add label for displaying slider value
            self.current_slider_3 = QSlider(Qt.Horizontal, self)
            self.current_slider_3.valueChanged[int].connect(self.on_slider_change_3)
            self.current_slider_3.setMinimum(self.Min_slider[cur_index][2])
            self.current_slider_3.setMaximum(self.Max_slider[cur_index][2])
            self.current_slider_3.setTickInterval(10)
            self.current_slider_3.setValue(self.init_slider[cur_index][2])
            self.current_layo.addWidget(self.current_slider_3)
            self.controlStack.addWidget(self.current_widget[cur_index])

        if len(self.slider_name[cur_index])>3:
            self.current_label_4 = QLabel(self.slider_name[cur_index][3] + ': ' + str(self.init_slider[cur_index][3]), self.current_widget[cur_index])
            self.current_layo.addWidget(self.current_label_4) # Add label for displaying slider value
            self.current_slider_4 = QSlider(Qt.Horizontal, self)
            self.current_slider_4.valueChanged[int].connect(self.on_slider_change_4)
            self.current_slider_4.setMinimum(self.Min_slider[cur_index][3])
            self.current_slider_4.setMaximum(self.Max_slider[cur_index][3])
            self.current_slider_4.setTickInterval(10)
            self.current_slider_4.setValue(self.init_slider[cur_index][3])
            self.current_layo.addWidget(self.current_slider_4)
            self.controlStack.addWidget(self.current_widget[cur_index])

        if len(self.slider_name[cur_index])>4:
            self.current_label_5 = QLabel(self.slider_name[cur_index][4] + ': ' + str(self.init_slider[cur_index][4]), self.current_widget[cur_index])
            self.current_layo.addWidget(self.current_label_5) # Add label for displaying slider value
            self.current_slider_5 = QSlider(Qt.Horizontal, self)
            self.current_slider_5.valueChanged[int].connect(self.on_slider_change_5)
            self.current_slider_5.setMinimum(self.Min_slider[cur_index][4])
            self.current_slider_5.setMaximum(self.Max_slider[cur_index][4])
            self.current_slider_5.setTickInterval(10)
            self.current_slider_5.setValue(self.init_slider[cur_index][4])
            self.current_layo.addWidget(self.current_slider_5)
            self.controlStack.addWidget(self.current_widget[cur_index])

        self.progress.setVisible(False)

    def switch_control_set(self, index):
        self.controlStack.setCurrentIndex(index)

    def end_of_first_video(self):
        #stuff for end of process
        print('')

    def closeEvent(self, event):
        # Ask for confirmation before closing
        confirmation = QMessageBox.question(self, "Confirmation", "Are you sure you want to close the application?", QMessageBox.Yes | QMessageBox.No)

        if confirmation == QMessageBox.Yes:
            event.accept()  # Close the app
        else:
            event.ignore()  # Don't close the app

### End of miniAM insert

    @pyqtSlot()
    def update_button_index(self):
        button_indx=self.current_control
        if 0 <= button_indx < 17:
            self.thread.update_button_indx(button_indx)
        else:
            print('Invalid function index:', button_indx)
            default_index = 17
            self.thread.update_button_indx(default_index)

    @pyqtSlot()
    def current_drop_changed_lambda(self,value):
        self.thread.get_method(value)

    @pyqtSlot()
    def start_thread(self):
        self.thread.resume()
        self.thread.play()
        self.upload_button.setVisible(False)

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def stop_thread(self):
        self.thread.stop_thread_play()

    @ pyqtSlot()
    def replay_thread(self):
        self.thread.restart_video() # Still working on this function

    @ pyqtSlot()
    def open_file_dialog(self): 
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
        if file_path:
            self.thread.set_file_path(file_path)
            self.upload_button.setVisible(False)
            self.csv_button.setVisible(False)
            self.send_init_slider_val()  
            self.thread.get_initial_video() 
    
    @ pyqtSlot()
    def open_csv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Get Parameters", "", "Parameter Files (*.csv)")
        if file_path:
            self.thread.set_csv_file_path(file_path)
            self.csv_button.setVisible(False)
            self.upload_button.setVisible(False)
            self.button_play.setVisible(False)
            self.button_stop.setVisible(False)
            self.button_restart.setVisible(False)
            self.button_load_videos.setVisible(True)

    @pyqtSlot()
    def pass_video_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder containing videos", "")
        if not folder_path:  
            return      
        self.button_load_videos.setVisible(False)
        self.button_minian.setVisible(True)
        self.thread.set_folder_file_path(folder_path)

    @ pyqtSlot()
    def run_minian(self):
        self.thread.call_VSC_minian()

    @pyqtSlot()
    def save_xarray(self):
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'NetCDF files (*.nc)')
        if save_path:
            passed_data=self.thread.get_xarray()
            passed_data.to_netcdf(save_path)

    @pyqtSlot()
    def save_changes(self):
        self.progress.setVisible(True)
        self.thread.stop_thread_play()
        self.thread.save_changes_2_xarray()
        self.progress.setVisible(False)
        self.next_btn.setVisible(True) 
        

    @pyqtSlot()
    def on_slider_change(self):
            current_value = self.current_slider.value()
            self.thread.temp_mod_frame(current_value)
            if self.current_control in self.decimal_tenths:
                current_value=(current_value/10)
            if self.current_control in self.decimal_hundreth:
                current_value=(current_value/100)          
            self.current_label.setText("{0}: {1}".format(self.slider_name[self.current_control][0], str(current_value))) 

    @pyqtSlot()
    def on_slider_change_2(self):
            current_value_2=self.current_slider_2.value()
            self.thread.temp_mod_frame_2(current_value_2)
            if self.current_control in self.decimal_tenths_2:
                current_value_2=(current_value_2/10)
            self.current_label_2.setText("{0}: {1}".format(self.slider_name[self.current_control][1], str(current_value_2)))  

    @pyqtSlot()
    def on_slider_change_3(self):
        current_value_3=self.current_slider_3.value()
        self.thread.temp_mod_frame_3(current_value_3)
        self.current_label_3.setText("{0}: {1}".format(self.slider_name[self.current_control][2], str(current_value_3))) 

    @pyqtSlot()
    def on_slider_change_4(self):
        current_value_4=self.current_slider_4.value()
        self.thread.temp_mod_frame_4(current_value_4)
        self.current_label_4.setText("{0}: {1}".format(self.slider_name[self.current_control][3], str(current_value_4))) 

    @pyqtSlot()
    def on_slider_change_5(self):
        current_value_5=self.current_slider_5.value()
        self.thread.temp_mod_frame_5(current_value_5)
        self.current_label_5.setText("{0}: {1}".format(self.slider_name[self.current_control][4], str(current_value_5))) 
    
    @pyqtSlot()
    def save_parameters(self):
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', '(*.csv)')
        if save_path:
            parameter_list=self.thread.get_parameters()
            parameter_list.to_csv(save_path, encoding='utf-8', index=True)
    
    @pyqtSlot()
    def send_init_slider_val(self):
        if self.init_slider[self.current_control][0] is not None:
            numb_of_sliders=len(self.init_slider[self.current_control])
            self.thread.get_init_val(self.init_slider[self.current_control][0])
        if self.current_control in self.method_index:
            method_index = self.method_index.index(self.current_control)
            self.thread.get_init_method(self.Method_drop_down[method_index][0])
        if self.current_control in self.bound_index:
            bound_index = self.bound_index.index(self.current_control)
            self.thread.get_init_method(self.numb_upper_bound_passing[bound_index][0])
        if numb_of_sliders>1:
            self.thread.get_init_val_2(self.init_slider[self.current_control][1])
        if numb_of_sliders>2:
            self.thread.get_init_val_3(self.init_slider[self.current_control][2])
        if numb_of_sliders>3:
            self.thread.get_init_val_4(self.init_slider[self.current_control][3])
        if numb_of_sliders>4:
            self.thread.get_init_val_5(self.init_slider[self.current_control][4])

    @pyqtSlot(int)
    def updateProgress(self, progress_val):
        self.progress.setValue(progress_val)

    @pyqtSlot()
    def countour_checkbox(self):
        if self.check_counter == 0:
            self.thread.contour_check()
            self.check_counter += 1
        else:
            self.thread.contour_unchecked()
            self.check_counter = 0

    @pyqtSlot()
    def deglow_checkbox(self):
        if self.glow_check_counter == 0:
            self.thread.glow_check()
            self.glow_check_counter += 1
        else:
            self.thread.glow_unchecked()
            self.glow_check_counter = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())