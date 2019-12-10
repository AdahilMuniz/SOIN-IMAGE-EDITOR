#Imports
import numpy   as np #Numpy
import tkinter as tk #TK GUI module
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tkinter import filedialog #File Dialog GUI
from PIL import Image, ImageTk #Image and Image Tk handle

from image_processor import image_processor
from soin_image import soin_image
from binary_file import binary_file


class soin_gui():

    #Attributes
    #Path
    top_path = None
    #Image
    soin_img_hand = None #Image Handle
    soin_img_hand_original = None #Original Image Handle
    #Image Processor
    img_proc_hand = None #Image Processor Handle
    #Binary File
    bin_file_hand = None

    #Top window
    wm            = None
    #Image Label
    wm_img        = None
    wm_img_label  = None
    #Menu Bar
    wm_menubar    = None
    wm_filemenu   = None
    wm_editmenu   = None
    wm_toolmenu   = None
    wm_helpmenu   = None
    #Log window
    log_factor    = None
    #Power Windows
    power_factor  = None
    power_exponent= None

    #Scacle 
    sacale_factor = None
    #Rotate
    rotate_factor = None

    #Free Hand Windows
    plot_wm = None
    fh_x1   = None
    fh_y1   = None
    fh_x2   = None
    fh_y2   = None
    #Histrogram
    hist_plot_wm = None
    #Steganography
    result_phrase = None
    steg_e = None
    #steg_v = None
    #Filters
    filter_dim = None
    filter_fac = None
    #Filters Free Hand
    f_fh_wm = None
    filter_fh_val = [[None, None, None], [None, None, None], [None, None, None]]
    f_fh_entries  = [[None, None, None], [None, None, None], [None, None, None]]
    f_fh_dim     = None
    #Spectrum Edit
    spec_wm = None
    spec_cv = None
    spec_wm_img = None
    spec_radius_val = None
    spec_width_val = None
    old_x = None
    old_y = None
    brush_size = 2;
    spec_filter_sel = None
    spec_cord_edit_buff = np.array([None, None, None])
    spec_result = None
    spec_result_abs = None
    #RGB <-> HSV
    rgb_hsv_red_val = None
    rgb_hsv_green_val = None
    rgb_hsv_blue_val = None
    rgb_hsv_hue_val = None
    rgb_hsv_sat_val = None
    rgb_hsv_value_val = None

    rgb_hsv_wm_scale_r = None
    rgb_hsv_wm_scale_g = None
    rgb_hsv_wm_scale_b = None
    rgb_hsv_wm_scale_h = None
    rgb_hsv_wm_scale_s = None
    rgb_hsv_wm_scale_v = None

    #Adjustment
    adj_h_factor = None
    adj_s_factor = None
    adj_v_factor = None

    #Chroma Key
    chk_img_entry = None
    chk_img_path  = None
    chk_radius    = None


    def __init__(self, soin_img_hand, img_proc_hand, bin_file_hand, top_path = ''):
        self.soin_img_hand = soin_img_hand
        self.img_proc_hand = img_proc_hand
        self.bin_file_hand = bin_file_hand
        self.top_path = top_path

    def build_wm(self):
        self.build_top()
        self.build_img_label()
        self.build_menubar()

    def build_top(self):
        #Top Windows
        self.wm = tk.Tk()
        icon_image = tk.PhotoImage(file = self.top_path+'/resource/logo/soin_logo_64X64.png') 
        self.wm.iconphoto(False, icon_image)
        self.wm.title("Soin Image Editor")
        if (self.soin_img_hand != None):
            self.wm.geometry(str(self.soin_img_hand.img_width)+"x"+str(self.soin_img_hand.img_height))
        else:
            self.wm.geometry("400x400")
        self.wm.bind_all("<Control-z>", self.restore)

    def build_img_label(self):
        #Image Label
        if (self.soin_img_hand != None):
            self.wm_img       = ImageTk.PhotoImage(self.soin_img_hand.img_handle)
            self.wm_img_label = tk.Label(self.wm, text = "Image Label", image =self.wm_img )
            self.wm_img_label.place(height=self.soin_img_hand.img_height, width=self.soin_img_hand.img_width, x=0, y=0)
        else:
            #img_init          = Image.open(self.top_path+'/resource/logo/soin_logo_bg_200X200.png')
            #wm_img_init       = ImageTk.PhotoImage(img_init)
            #wm_img_init_label = tk.Label(self.wm, text = "Image Label", image =wm_img_init )
            #wm_img_init_label.place(height=400, width=400, x=0, y=0)

            img_init          = Image.open(self.top_path+'/resource/logo/soin_logo_bg_200X200.png')
            self.wm_img       = ImageTk.PhotoImage(img_init)
            self.wm_img_label = tk.Label(self.wm, text = "Image Label", image =self.wm_img )
            self.wm_img_label.place(height=400, width=400, x=0, y=0)

    def build_menubar(self):
        #Menu
        self.wm_menubar  = tk.Menu(self.wm)
        self.wm_filemenu = tk.Menu(self.wm_menubar, tearoff=0)

        #File Menu Label
        self.wm_filemenu.add_command(label="Open", command=self.browse_open_file)
        self.wm_filemenu.add_command(label="Save", command=self.browse_save_file)
        self.wm_filemenu.add_separator()
        self.wm_filemenu.add_command(label="Exit", command=self.wm.quit)

        self.wm_menubar.add_cascade(label="File", menu=self.wm_filemenu)

        #Edit Menu Label
        self.wm_editmenu = tk.Menu(self.wm_menubar, tearoff=0)
        self.wm_editmenu.add_command(label="Log Function"          , command=self.build_log_wm)
        self.wm_editmenu.add_command(label="Power Function"        , command=self.build_power_wm)
        self.wm_editmenu.add_command(label="Negative"              , command=self.negative_call)
        self.wm_editmenu.add_command(label="Free Hand"             , command=self.build_free_hand_wm)
        self.wm_editmenu.add_command(label="Steganography"         , command=self.build_steganography_wm)
        self.wm_editmenu.add_command(label="Histogram Equalization", command=self.build_hist_eq_wm)
        self.wm_editmenu.add_command(label="Filter"                , command=self.build_filters_wm)
        self.wm_editmenu.add_command(label="Fourier Transform"     , command=self.fft_call)
        self.wm_editmenu.add_command(label="Scale"                 , command=self.build_scale_wm)
        self.wm_editmenu.add_command(label="Rotate"                , command=self.build_rotate_wm)

        self.wm_menubar.add_cascade(label="Edit", menu=self.wm_editmenu)

        #Tool Menu Label
        self.wm_toolmenu = tk.Menu(self.wm_menubar, tearoff=0)
        self.wm_toolmenu.add_command(label="RGB<->HSV"                          , command=self.build_rgb_hsv_wm)
        self.wm_toolmenu.add_command(label="RGB to Grayscale(Average)"          , command= lambda: self.rgb2grayscale_call(0))
        self.wm_toolmenu.add_command(label="RGB to Grayscale(Weighted Average)" , command= lambda: self.rgb2grayscale_call(1))
        self.wm_toolmenu.add_command(label="Sepia"                              , command=self.sepia_call)
        self.wm_toolmenu.add_command(label="Adjustment"                         , command=self.build_adjustment_wm)
        self.wm_toolmenu.add_command(label="Chroma Key"                         , command=self.build_chkey_wm)
        self.wm_toolmenu.add_command(label="Wavelet Haar"                       , command=self.build_haar_wm)


        self.wm_menubar.add_cascade(label="Tools", menu=self.wm_toolmenu)

        #Help Menu Label
        self.wm_helpmenu = tk.Menu(self.wm_menubar, tearoff=0)
        self.wm_helpmenu.add_command(label="Tools List", command=self.dummy)
        self.wm_helpmenu.add_command(label="About..."  , command=self.dummy)

        self.wm_menubar.add_cascade(label="Help", menu=self.wm_helpmenu)

        self.wm.config(menu=self.wm_menubar)

        if (self.soin_img_hand == None):
            self.wm_menubar.entryconfig("Edit", state="disable")
            self.wm_menubar.entryconfig("Tools", state="disable")

    def browse_open_file(self):
        filetypes=[('TIF files','*.tif'), ('PNG files','*.png'), ('JPEG files','*.jpeg *.jpg'), ('Soin Compress files','*.scimg'), ('All','*')]
        img_file = filedialog.askopenfile(parent=self.wm, mode='rb', title='Choose a image', filetypes=filetypes)
        try:
            file,ext = os.path.splitext(img_file.name)
            if (ext != '.scimg'):
                new_soin_img_hand = soin_image(img_file.name, self.img_proc_hand)
                new_soin_img_hand.open_image()
                self.update_image(new_soin_img_hand)
            else:
                new_soin_img_hand = soin_image(img_file.name, self.img_proc_hand)
                self.soin_img_hand = new_soin_img_hand
                #new_soin_img_hand.open_image()
                #self.update_image(new_soin_img_hand)
                #self.open_compressed_file(img_file.name)
                self.open_compressed_file(img_file.name)
        except AttributeError:
            print('> Any new image was selected.')

    def browse_save_file(self):
        filetypes=[('TIF files','*.tif'), ('PNG files','*.png'), ('JPEG files','*.jpeg'), ('BMP files','*.bmp'), ('Soin Compress files','*.scimg')]
        img_file = filedialog.asksaveasfile(parent=self.wm, mode='w', title='Choose a image', filetypes=filetypes)
        file,ext = os.path.splitext(img_file.name)
        try:
            if (ext != '.scimg'):
                self.soin_img_hand.save_image(img_file.name)
            else:
                #self.save_compressed_file(img_file.name)
                self.build_compress_wm(img_file.name)
        except AttributeError:
            print('> An error occurred trying to save the image.')

    def update_image(self, new_soin_img_hand):
        self.soin_img_hand = new_soin_img_hand
        self.wm.geometry(str(self.soin_img_hand.img_width)+"x"+str(self.soin_img_hand.img_height))
        img = ImageTk.PhotoImage(self.soin_img_hand.img_handle)
        self.wm_img_label.configure(image = img)
        self.wm_img_label.image = img
        self.wm_img_label.place(height=self.soin_img_hand.img_height, width=self.soin_img_hand.img_width, x=0, y=0)
        if (self.soin_img_hand != None):
            self.wm_menubar.entryconfig("Edit", state="normal")
            self.wm_menubar.entryconfig("Tools", state="normal")

    def build_log_wm(self):
        self.log_factor = tk.DoubleVar()
        log_wm = tk.Toplevel(self.wm)
        log_wm.title("Log Paramenters")
        log_wm.geometry("200x100")
        log_wm_label_text  = tk.Label(log_wm, text = "Factor")
        log_wm_label_text.place(x=10, y=25)
        log_wm_scale = tk.Scale(log_wm, variable=self.log_factor, from_=0.1, to=100, resolution = 0.1, orient=tk.HORIZONTAL)
        log_wm_scale.place(x=70, y=5)
        log_wm_bt    = tk.Button(log_wm, width = 3, height = 1, text = "Apply", command = self.log_call)
        log_wm_bt.place(x=85, y=60)

    def build_power_wm(self):
        self.power_factor   = tk.DoubleVar()
        self.power_exponent = tk.DoubleVar()
        power_wm = tk.Toplevel(self.wm)
        power_wm.title("Power Paramenters")
        power_wm.geometry("200x150")
        power_wm_factor_label_text  = tk.Label(power_wm, text = "Factor")
        power_wm_factor_label_text.place(x=10, y=25)
        power_wm_exponent_label_text  = tk.Label(power_wm, text = "Exponent")
        power_wm_exponent_label_text.place(x=10, y=65)

        power_wm_factor_scale = tk.Scale(power_wm, variable=self.power_factor, from_=0.1, to=100, resolution = 0.1, orient=tk.HORIZONTAL)
        power_wm_factor_scale.place(x=80, y=5)
        power_wm_exponent_scale = tk.Scale(power_wm, variable=self.power_exponent, from_=0.1, to=10, resolution = 0.1, orient=tk.HORIZONTAL)
        power_wm_exponent_scale.place(x=80, y=45)
        power_wm_bt    = tk.Button(power_wm, width = 3, height = 1, text = "Apply", command = self.power_call)
        power_wm_bt.place(x=85, y=100)


    def build_scale_wm(self):
        self.scale_factor = tk.DoubleVar()
        
        scale_mode = tk.IntVar()
        scale_wm = tk.Toplevel(self.wm)

        MODES = [
            ("Nearest Neighbor", 0),
            ("Bilinear"        , 1),
        ]

        scale_wm.title("Scaling")
        scale_wm.geometry("200x150")
        scale_wm_label_text  = tk.Label(scale_wm, text = "Factor")
        scale_wm_label_text.place(x=10, y=25)
        scale_wm_scale = tk.Scale(scale_wm, variable=self.scale_factor, from_=0.1, to=10, resolution = 0.1, orient=tk.HORIZONTAL)
        scale_wm_scale.place(x=70, y=5)
        
        for text, mode in MODES:
            r_bt = tk.Radiobutton(scale_wm, text=text, value=mode, variable=scale_mode)
            r_bt.place(x=10, y=60+mode*20)

        sacale_wm_bt    = tk.Button(scale_wm, width = 3, height = 1, text = "Apply", command = lambda: self.scale_call(scale_mode.get()))
        sacale_wm_bt.place(x=85, y=100)

    def build_rotate_wm(self):
        self.rotate_factor = tk.DoubleVar()
        
        rotate_mode = tk.IntVar()
        rotate_wm = tk.Toplevel(self.wm)

        rotate_wm.title("Rotating")
        rotate_wm.geometry("200x100")
        rotate_wm_label_text  = tk.Label(rotate_wm, text = "Angle")
        rotate_wm_label_text.place(x=10, y=25)
        rotate_wm_rotate = tk.Scale(rotate_wm, variable=self.rotate_factor, from_=-360, to=+360, orient=tk.HORIZONTAL)
        rotate_wm_rotate.set(0)
        rotate_wm_rotate.place(x=70, y=5)

        sacale_wm_bt    = tk.Button(rotate_wm, width = 3, height = 1, text = "Apply", command = self.rotate_call)
        sacale_wm_bt.place(x=85, y=60)
        
    def build_free_hand_wm(self):
        self.fh_x1 = tk.IntVar()
        self.fh_y1 = tk.IntVar()
        self.fh_x2 = tk.IntVar()
        self.fh_y2 = tk.IntVar()

        fh_wm = tk.Toplevel(self.wm)
        fh_wm.title("Free Hand Function")
        fh_wm.geometry("200x100")
        #Parameters
        #1
        fh_wm_x1_label_text  = tk.Label(fh_wm, text = "x1")
        fh_wm_x1_label_text.place(x=10, y=20)
        fh_wm_y1_label_text  = tk.Label(fh_wm, text = "y1")
        fh_wm_y1_label_text.place(x=70, y=20)
        fh_wm_x1_entry = tk.Entry(fh_wm, bd = 1, textvariable=self.fh_x1)
        fh_wm_x1_entry.place(x=30, y=20, height = 20, width = 30)
        fh_wm_y1_entry = tk.Entry(fh_wm, bd = 1, textvariable=self.fh_y1)
        fh_wm_y1_entry.place(x=90, y=20, height = 20, width = 30)
        #2
        fh_wm_x2_label_text  = tk.Label(fh_wm, text = "x2")
        fh_wm_x2_label_text.place(x=10, y=40)
        fh_wm_y2_label_text  = tk.Label(fh_wm, text = "y2")
        fh_wm_y2_label_text.place(x=70, y=40)
        fh_wm_x2_entry = tk.Entry(fh_wm, bd = 1, textvariable=self.fh_x2)
        fh_wm_x2_entry.place(x=30, y=40, height = 20, width = 30)
        fh_wm_y2_entry = tk.Entry(fh_wm, bd = 1, textvariable=self.fh_y2)
        fh_wm_y2_entry.place(x=90, y=40, height = 20, width = 30)
        #Buttons
        fh_wm_view_bt    = tk.Button(fh_wm, width = 3, height = 1, text = "View", command = self.plot)
        fh_wm_view_bt.place(x=130, y=20)
        fh_wm_apply_bt    = tk.Button(fh_wm, width = 3, height = 1, text = "Apply", command = self.free_hand_call)
        fh_wm_apply_bt.place(x=130, y=50)

    def build_steganography_wm(self): 
        self.log_factor = tk.IntVar()
        steg_wm = tk.Toplevel(self.wm)
        steg_wm.title("Steganography")
        steg_wm.geometry("500x100")
        steg_v = tk.StringVar()

        self.steg_e = tk.Entry(steg_wm, bd = 1, textvariable=steg_v)
        self.steg_e.place(x=10, y=40, height = 20, width = 480)

        steg_wm_enc_bt    = tk.Button(steg_wm, width = 3, height = 1, text = "Encode", command = lambda: self.steganography_enc_call(steg_v.get()))
        steg_wm_enc_bt.place(x=180, y=65)

        steg_wm_dec_bt    = tk.Button(steg_wm, width = 3, height = 1, text = "Decode", command = self.steganography_dec_call)
        steg_wm_dec_bt.place(x=250, y=65)

    def build_filters_wm(self):

        self.filter_dim = tk.IntVar()
        self.filter_fac = tk.DoubleVar()

        f_wm = tk.Toplevel(self.wm)
        f_wm.title("Filters")
        f_wm.geometry("280x210")
        #Buttons
        f_wm_ave_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Average", command = lambda: self.filter_call(1))
        f_wm_ave_bt.place(x=10, y=10)
        f_wm_gaus_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Gaussian", command = lambda: self.filter_call(2))
        f_wm_gaus_bt.place(x=100, y=10)
        f_wm_lapl_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Laplacian", command = lambda: self.filter_call(3))
        f_wm_lapl_bt.place(x=10, y=50)
        f_wm_hb_bt    = tk.Button(f_wm, width = 7, height = 1, text = "High-Boost", command = lambda: self.filter_call(4))
        f_wm_hb_bt.place(x=100, y=50)
        f_wm_med_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Median", command = lambda: self.filter_call(5))
        f_wm_med_bt.place(x=10, y=90)
        f_wm_fh_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Free Hand", command = self.build_filters_free_hand_wm)
        f_wm_fh_bt.place(x=100, y=90)
        f_wm_geo_m_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Geo. Mean", command = lambda: self.filter_call(7))
        f_wm_geo_m_bt.place(x=10, y=130)
        f_wm_harm_m_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Harm. Mean", command = lambda: self.filter_call(8))
        f_wm_harm_m_bt.place(x=100, y=130)
        f_wm_sobel_bt    = tk.Button(f_wm, width = 7, height = 1, text = "Sobel", command = lambda: self.filter_call(6))
        f_wm_sobel_bt.place(x=10, y=170)
        
        #Dimensions
        f_wm_dim_label_text  = tk.Label(f_wm, text = "Dimension")
        f_wm_dim_label_text.place(x=200, y=10)
        f_wm_dim_entry = tk.Entry(f_wm, bd = 1, textvariable=self.filter_dim)
        #Replace 0 with 3
        f_wm_dim_entry.delete(0, last = 1)
        f_wm_dim_entry.insert(0,"3")
        f_wm_dim_entry.place(x=200, y=30, height = 20, width = 30)

        #Factor
        f_wm_fac_label_text  = tk.Label(f_wm, text = "Factor")
        f_wm_fac_label_text.place(x=200, y=60)
        f_wm_fac_entry = tk.Entry(f_wm, bd = 1, textvariable=self.filter_fac)
        f_wm_fac_entry.delete(0, last = 3)
        f_wm_fac_entry.insert(0,"1")
        f_wm_fac_entry.place(x=200, y=80, height = 20, width = 30)

    def build_filters_free_hand_wm(self):
        self.f_fh_wm = tk.Toplevel(self.wm)
        self.f_fh_wm.title("Free Hand Mask")
        self.f_fh_wm.geometry("300x300")

        f_fh_ent_dx  = 30
        f_fh_ent_dy  = 20
        f_fh_ent_x   = 100
        f_fh_ent_y   = 100
        
        self.f_fh_dim = 3 

        for i in range(0, self.f_fh_dim):
            for j in range(0, self.f_fh_dim):
                self.filter_fh_val[i][j] = tk.DoubleVar()
                f_fh_entry = tk.Entry(self.f_fh_wm, bd = 1, textvariable=self.filter_fh_val[i][j])
                f_fh_entry.place(x=i*f_fh_ent_dx+f_fh_ent_x, y=j*f_fh_ent_dy+f_fh_ent_y, height = 20, width = 30)
                self.f_fh_entries[i][j] = f_fh_entry

            
        f_fh_plus_bt    = tk.Button(self.f_fh_wm, width = 1, height = 1, text = "+", command = self.add_filters_dim)
        f_fh_plus_bt.place(x=10, y=260)
        f_fh_minus_bt    = tk.Button(self.f_fh_wm, width = 1, height = 1, text = "-", command = self.sub_filters_dim)
        f_fh_minus_bt.place(x=50, y=260)
        f_fh_aplly_bt    = tk.Button(self.f_fh_wm, width = 5, height = 1, text = "Apply", command = self.conv_free_hand_call)
        f_fh_aplly_bt.place(x=200, y=260)

    def build_freq_wm(self):

        self.filter_dim = tk.IntVar()
        self.filter_fac = tk.DoubleVar()

        freq_wm = tk.Toplevel(self.wm)
        freq_wm.title("Frquency Domain")
        freq_wm.geometry("280x200")
        #Buttons
        freq_wm_ave_bt    = tk.Button(freq_wm, width = 9, height = 1, text = "Fourier Transform", command = self.fft_call)
        freq_wm_ave_bt.place(x=10, y=10)

    def build_spectrum_wm(self, spec_img):
        self.spec_wm = tk.Toplevel(self.wm)
        self.spec_wm.title("Spectrum")
        self.spec_radius_val = tk.IntVar()
        self.spec_width_val = tk.IntVar()

        spec_dif_height = 500 - spec_img.shape[0]

        if (spec_dif_height < 0):
            spec_dif_height = 0

        spec_wm_height = spec_img.shape[0]+spec_dif_height

        self.spec_wm.geometry(str(spec_img.shape[1]+250)+"x"+str(spec_wm_height))

        self.spec_filter_sel = tk.IntVar()

        #Radio Buttons Modes
        MODES = [
            ("Free Hand"     , 0),
            ("Low-Pass"      , 1),
            ("High-Pass"     , 2),
            ("Bandpass"      , 3),
            ("Bandreject"    , 4),
            ("Low-Pass (G)"  , 5),
            ("High-Pass (G)" , 6),
            ("Bandpass (G)"  , 7),
            ("Bandreject (G)", 8),
        ]

        #Image Label
        self.spec_wm_img       = Image.fromarray(spec_img)
        self.spec_wm_img       = ImageTk.PhotoImage(self.spec_wm_img)

        #Canvas
        self.spec_cv = tk.Canvas(self.spec_wm, height=spec_img.shape[0] ,width=spec_img.shape[1], bg="black")
        self.spec_cv.place(height=spec_img.shape[0] , width=spec_img.shape[1], x=0, y=0)
        self.spec_cv.image = self.spec_wm_img
        self.spec_cv.create_image(0,0, image=self.spec_cv.image, anchor = 'nw')

        self.spec_cv.focus_set()

        #Label
        spec_radius_label  = tk.Label(self.spec_wm, text = "Radius")
        spec_radius_label.place(x=spec_img.shape[1]+150, y=10)
        spec_width_label  = tk.Label(self.spec_wm, text = "Width")
        spec_width_label.place(x=spec_img.shape[1]+150, y=70)
        #Entry 
        #self.spec_filter_sel.trace("w", self.freq_filter_sel)
        spec_radius_entry = tk.Entry(self.spec_wm, bd = 1, textvariable=self.spec_radius_val)
        spec_radius_entry.place(x=spec_img.shape[1]+150, y=40, height = 20, width = 50)
        self.spec_radius_val.set(1)

        spec_width_entry = tk.Entry(self.spec_wm, bd = 1, textvariable=self.spec_width_val)
        spec_width_entry.place(x=spec_img.shape[1]+150, y=90, height = 20, width = 50)
        self.spec_width_val.set(1)
            

        #Buttons

        for text, mode in MODES:
            r_bt = tk.Radiobutton(self.spec_wm, text=text, value=mode, variable=self.spec_filter_sel, command = self.freq_filter_sel)
            r_bt.place(x=spec_img.shape[1]+20, y=10+mode*50)

        apply_bt    = tk.Button(self.spec_wm, width = 7, height = 1, text = "Apply", command = self.freq_filter_sel_call)
        apply_bt.place(x=spec_img.shape[1]+50, y=spec_wm_height-50)

    def build_haar_wm(self):
        haar_wm = tk.Toplevel(self.wm)
        haar_wm.title("Wavelet Haar")
        haar_depth = tk.IntVar()

        haar_wm.geometry(str(self.soin_img_hand.img_width+250)+"x"+str(self.soin_img_hand.img_height))

        haar_wm_img       = Image.fromarray(self.soin_img_hand.img_array)
        haar_wm_img       = ImageTk.PhotoImage(haar_wm_img)


    def build_rgb_hsv_wm(self):
        rgb_hsv_wm = tk.Toplevel(self.wm)
        rgb_hsv_wm.title("RGB <-> HSV")
        rgb_hsv_wm.geometry("400x200")

        self.rgb_hsv_red_val = tk.IntVar()
        self.rgb_hsv_green_val = tk.IntVar()
        self.rgb_hsv_blue_val = tk.IntVar()
        self.rgb_hsv_hue_val = tk.IntVar()
        self.rgb_hsv_sat_val = tk.DoubleVar()
        self.rgb_hsv_value_val = tk.DoubleVar()

        ##RGB##
        rgb_hsv_wm_label_rgb  = tk.Label(rgb_hsv_wm, text = "RGB")
        rgb_hsv_wm_label_rgb.place(x=100, y=5)

        rgb_hsv_wm_label_r  = tk.Label(rgb_hsv_wm, text = "Red")
        rgb_hsv_wm_label_r.place(x=10, y=45)
        self.rgb_hsv_wm_scale_r = tk.Scale(rgb_hsv_wm, variable=self.rgb_hsv_red_val, command=self.rgb_callback, fg="red", from_=0, to=255, orient=tk.HORIZONTAL)
        self.rgb_hsv_wm_scale_r.place(x=70, y=25)

        rgb_hsv_wm_label_g  = tk.Label(rgb_hsv_wm, text = "Green")
        rgb_hsv_wm_label_g.place(x=10, y=85)
        self.rgb_hsv_wm_scale_g = tk.Scale(rgb_hsv_wm, variable=self.rgb_hsv_green_val, command=self.rgb_callback, fg="green", from_=0, to=255, orient=tk.HORIZONTAL)
        self.rgb_hsv_wm_scale_g.place(x=70, y=65)

        rgb_hsv_wm_label_b  = tk.Label(rgb_hsv_wm, text = "Blue")
        rgb_hsv_wm_label_b.place(x=10, y=125)
        self.rgb_hsv_wm_scale_b = tk.Scale(rgb_hsv_wm, variable=self.rgb_hsv_blue_val, command=self.rgb_callback, fg="blue", from_=0, to=255, orient=tk.HORIZONTAL)
        self.rgb_hsv_wm_scale_b.place(x=70, y=105)

        ##HSV##
        rgb_hsv_wm_label_HSV  = tk.Label(rgb_hsv_wm, text = "HSV")
        rgb_hsv_wm_label_HSV.place(x=300, y=5)

        rgb_hsv_wm_label_h  = tk.Label(rgb_hsv_wm, text = "Hue")
        rgb_hsv_wm_label_h.place(x=200, y=45)
        self.rgb_hsv_wm_scale_h = tk.Scale(rgb_hsv_wm, variable=self.rgb_hsv_hue_val, command=self.hsv_callback, from_=0, to=360, orient=tk.HORIZONTAL)
        self.rgb_hsv_wm_scale_h.place(x=260, y=25)

        rgb_hsv_wm_label_s  = tk.Label(rgb_hsv_wm, text = "Sat.")
        rgb_hsv_wm_label_s.place(x=200, y=85)
        self.rgb_hsv_wm_scale_s = tk.Scale(rgb_hsv_wm, variable=self.rgb_hsv_sat_val, command=self.hsv_callback, from_=0, to=100, resolution=0.1, orient=tk.HORIZONTAL)
        self.rgb_hsv_wm_scale_s.place(x=260, y=65)

        rgb_hsv_wm_label_v  = tk.Label(rgb_hsv_wm, text = "Value")
        rgb_hsv_wm_label_v.place(x=200, y=125)
        self.rgb_hsv_wm_scale_v = tk.Scale(rgb_hsv_wm, variable=self.rgb_hsv_value_val, command=self.hsv_callback, from_=0, to=100, resolution=0.1, orient=tk.HORIZONTAL)
        self.rgb_hsv_wm_scale_v.place(x=260, y=105)

    
    def rgb_callback(self, value):
        hsv = self.img_proc_hand.rgb2hsv([self.rgb_hsv_red_val.get(), self.rgb_hsv_green_val.get(), self.rgb_hsv_blue_val.get()])
        self.rgb_hsv_wm_scale_h.set(hsv[0])
        self.rgb_hsv_wm_scale_s.set(hsv[1])
        self.rgb_hsv_wm_scale_v.set(hsv[2])

    def hsv_callback(self, value):
        rgb = self.img_proc_hand.hsv2rgb([self.rgb_hsv_hue_val.get(), self.rgb_hsv_sat_val.get(), self.rgb_hsv_value_val.get()])
        self.rgb_hsv_wm_scale_r.set(rgb[0])
        self.rgb_hsv_wm_scale_g.set(rgb[1])
        self.rgb_hsv_wm_scale_b.set(rgb[2])

    def build_adjustment_wm(self):
        self.adj_h_factor = tk.DoubleVar()
        self.adj_s_factor = tk.DoubleVar()
        self.adj_v_factor = tk.DoubleVar()

        adj_wm = tk.Toplevel(self.wm)
        adj_wm.title("Adjustment Paramenters")
        adj_wm.geometry("200x200")
        
        adj_wm_label_h_text  = tk.Label(adj_wm, text = "Hue")
        adj_wm_label_h_text.place(x=15, y=25)
        adj_wm_h_scale = tk.Scale(adj_wm, variable=self.adj_h_factor, from_=0.1, to=1, resolution = 0.1, orient=tk.HORIZONTAL)
        adj_wm_h_scale.place(x=85, y=5)
        adj_wm_h_scale.set(1)

        adj_wm_label_s_text  = tk.Label(adj_wm, text = "Saturation")
        adj_wm_label_s_text.place(x=15, y=65)
        adj_wm_s_scale = tk.Scale(adj_wm, variable=self.adj_s_factor, from_=0.1, to=1, resolution = 0.1, orient=tk.HORIZONTAL)
        adj_wm_s_scale.place(x=85, y=45)
        adj_wm_s_scale.set(1)

        adj_wm_label_v_text  = tk.Label(adj_wm, text = "Value")
        adj_wm_label_v_text.place(x=15, y=105)
        adj_wm_v_scale = tk.Scale(adj_wm, variable=self.adj_v_factor, from_=0.1, to=1, resolution = 0.1, orient=tk.HORIZONTAL)
        adj_wm_v_scale.place(x=85, y=85)
        adj_wm_v_scale.set(1)

        adj_wm_bt    = tk.Button(adj_wm, width = 3, height = 1, text = "Apply", command = self.adj_call)
        adj_wm_bt.place(x=85, y=140)

    def build_chkey_wm(self):
        chk_wm = tk.Toplevel(self.wm)
        chk_wm.title("Chroma Key")
        chk_wm.geometry("250x150")

        self.chk_img_path = tk.StringVar()
        self.chk_radius   = tk.IntVar()

        #Image
        chk_img_label = tk.Label(chk_wm, bd = 1, text="Image:")
        chk_img_label.place(x=20, y=25)
        self.chk_img_entry = tk.Entry(chk_wm, bd = 1, textvariable=self.chk_img_path)
        self.chk_img_entry.place(x=80, y=25, height = 20, width = 100)
        chk_img_bt    = tk.Button(chk_wm, width = 3, height = 1, text = "Browse", command = self.open_image_background)
        chk_img_bt.place(x=190, y=20)

        #Radius
        chk_img_label_radius  = tk.Label(chk_wm, text = "Radius:")
        chk_img_label_radius.place(x=20, y=70)
        chk_img_scale = tk.Scale(chk_wm, variable=self.chk_radius, from_=0.1, to=1000, orient=tk.HORIZONTAL)
        chk_img_scale.place(x=80, y=50)
            
        chk_apply_bt    = tk.Button(chk_wm, width = 3, height = 1, text = "Apply", command = self.chroma_key_call)
        chk_apply_bt.place(x=85, y=100)

    def build_compress_wm(self, path):
        comp_wm = tk.Toplevel(self.wm)
        comp_wm.title("Compression")
        comp_wm.geometry("250x160")

        comp_method = tk.IntVar()
        haar_comp_depth = tk.IntVar()

        #Radio Buttons Modes
        MODES = [
            ("Huffman"        , 0),
            ("Haar + Huffman" , 1),
            ("Haar (Grayscale) + Huffman" , 2),
        ]


        #Buttons

        for text, mode in MODES:
            r_bt = tk.Radiobutton(comp_wm, text=text, value=mode, variable=comp_method)
            r_bt.place(x=20, y=10+mode*20)

        #Radius
        comp_haar_label_depth = tk.Label(comp_wm, text = "Haar Depth:")
        comp_haar_label_depth.place(x=10, y=90)
        comp_haar_scale_depth = tk.Scale(comp_wm, variable=haar_comp_depth, from_=1, to=8, orient=tk.HORIZONTAL)
        comp_haar_scale_depth.place(x=120, y=70)
            
        comp_save_bt    = tk.Button(comp_wm, width = 3, height = 1, text = "Save", command = lambda: self.save_compressed_file(path, comp_method.get(), haar_comp_depth.get()))
        comp_save_bt.place(x=85, y=115)

    def build_decompress_wm(self, path):
        decomp_wm = tk.Toplevel(self.wm)
        decomp_wm.title("Compression")
        decomp_wm.geometry("250x150")

        decomp_method = tk.IntVar()

        #Radio Buttons Modes
        MODES = [
            ("Huffman"        , 0),
            ("Haar + Huffman" , 1),
            ("Haar (Grayscale) + Huffman" , 2),
        ]

        #Buttons

        for text, mode in MODES:
            r_bt = tk.Radiobutton(decomp_wm, text=text, value=mode, variable=decomp_method)
            r_bt.place(x=20, y=10+mode*20)
            
        decomp_open_bt    = tk.Button(decomp_wm, width = 3, height = 1, text = "Open", command = lambda: self.open_compressed_file(path))
        decomp_open_bt.place(x=85, y=100)



    def open_image_background(self):
        filetypes=[('TIF files','*.tif'), ('PNG files','*.png'), ('JPEG files','*.jpeg *.jpg'), ('All','*')]
        img_file = filedialog.askopenfile(parent=self.wm, mode='rb', title='Choose a image', filetypes=filetypes)
        try:
            #new_soin_img_hand = soin_image(img_file.name, self.img_proc_hand)
            #new_soin_img_hand.open_image()
            #self.update_image(new_soin_img_hand)
            self.chk_img_entry.insert(0, img_file.name)
        except AttributeError:
            print('> Any new image was selected.')

    def chroma_key_call(self):
        new_soin_img_hand = soin_image(self.chk_img_path.get(), self.img_proc_hand)
        new_soin_img_hand.open_image()

        self.soin_img_hand.img_array = self.img_proc_hand.chroma_key(self.soin_img_hand.img_array, new_soin_img_hand.img_array, [0, 255, 0] , self.chk_radius.get() )
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image


    def freq_filter_sel(self):
        sel = self.spec_filter_sel.get()

        #Free Hand
        if (sel == 0):
            self.spec_cv.bind('<Button-1>', self.paint_mouse_press)
            self.spec_cv.bind('<B1-Motion>', self.paint_mouse_motion)
            self.spec_cv.bind('<ButtonRelease-1>', self.paint_mouse_release)
            self.spec_cv.bind('<Key>', self.paint_key_press)
        else:
            self.spec_cv.unbind('<Button-1>')
            self.spec_cv.unbind('<B1-Motion>')
            self.spec_cv.unbind('<ButtonRelease-1>')
            self.spec_cv.unbind('<Key>')
            self.spec_cv.delete("all")
            self.spec_cv.image = self.spec_wm_img
            self.spec_cv.create_image(0,0, image=self.spec_cv.image, anchor = 'nw')

            self.spec_cord_edit_buff = np.array([None, None, None])

        if (sel >= 1):
            if (sel == 1):
                img_result = self.img_proc_hand.fourier_hand.low_pass(self.spec_result_abs, self.spec_radius_val.get(), 0)
            elif (sel == 2):
                img_result = self.img_proc_hand.fourier_hand.high_pass(self.spec_result_abs, self.spec_radius_val.get(), 0)
            elif (sel == 3):
                img_result = self.img_proc_hand.fourier_hand.band_pass(self.spec_result_abs, self.spec_radius_val.get(), self.spec_width_val.get(), 0)
            elif (sel == 4):
                img_result = self.img_proc_hand.fourier_hand.band_reject(self.spec_result_abs, self.spec_radius_val.get(), self.spec_width_val.get(), 0)
            elif (sel == 5):
                img_result = self.img_proc_hand.fourier_hand.low_pass(self.spec_result_abs, self.spec_radius_val.get(), 1)
            elif (sel == 6):
                img_result = self.img_proc_hand.fourier_hand.high_pass(self.spec_result_abs, self.spec_radius_val.get(), 1)
            elif (sel == 7):
                img_result = self.img_proc_hand.fourier_hand.band_pass(self.spec_result_abs, self.spec_radius_val.get(), self.spec_width_val.get(), 1)
            elif (sel == 8):
                img_result = self.img_proc_hand.fourier_hand.band_reject(self.spec_result_abs, self.spec_radius_val.get(), self.spec_width_val.get(), 1)
            else:
                img_result = self.spec_result_abs

            img_result = Image.fromarray(img_result)
            img_result = ImageTk.PhotoImage(img_result)
            self.spec_cv.image = img_result
            self.spec_cv.create_image(0,0, image=self.spec_cv.image, anchor = 'nw')
        else:
            self.spec_cv.image = self.spec_wm_img
            self.spec_cv.create_image(0,0, image=self.spec_cv.image, anchor = 'nw')


    def freq_filter_sel_call(self):
        sel = self.spec_filter_sel.get()
        if(sel == 0):
            self.free_hand_ifft_call()
        elif(sel >= 1):
            self.freq_filter_func_call(sel-1)

        self.spec_wm.destroy()
        self.spec_wm = None

    #Paint Methods
    def paint_mouse_motion(self, event):
        x, y = (event.x), (event.y)
        cord = np.array([x,y, self.brush_size])
        self.spec_cord_edit_buff = np.vstack((self.spec_cord_edit_buff, cord))

        if (self.old_x and self.old_y):
            self.spec_cv.create_rectangle((x-self.brush_size, y-self.brush_size, x+self.brush_size, y+self.brush_size), fill="black",  outline="")

        self.old_x = event.x
        self.old_y = event.y

    def paint_mouse_release(self, event):
        self.old_x = None
        self.old_y = None
        #print(self.spec_cord_edit_buff)

    def paint_mouse_press(self, event):
        x, y = (event.x), (event.y)
        cord = np.array([x,y, self.brush_size])
        self.spec_cord_edit_buff = np.vstack((self.spec_cord_edit_buff, cord))
        self.spec_cv.create_rectangle((x-self.brush_size, y-self.brush_size, x+self.brush_size, y+self.brush_size), fill="black",  outline="")


    def paint_key_press(self, event):
        if (event.char == '+'):
            if (self.brush_size < 100):
                self.brush_size += 1
            else:
                print("> Maximum brush size is 100.")
        elif (event.char == '-'):
            if (self.brush_size > 1):
                self.brush_size -= 1
            else:
                print("> Minimum brush size is 1.")
            
        
    def add_filters_dim(self):

        f_fh_ent_dx  = 30
        f_fh_ent_dy  = 20
        f_fh_ent_x   = 100
        f_fh_ent_y   = 100

        if (self.f_fh_dim >= 9):
            print("> ERROR: Max dimension is 9")
            return -1

        #Append New dimensions to the value list
        self.filter_fh_val.insert(0, [None] * (self.f_fh_dim)) #First Row
        self.filter_fh_val.append([None] * self.f_fh_dim) #Last Row
        self.filter_fh_val = [x + [None] for x in self.filter_fh_val] #Last Column
        self.filter_fh_val = [[None] + x for x in self.filter_fh_val] #First Column

        self.f_fh_entries.insert(0, [None] * self.f_fh_dim) #First Row
        self.f_fh_entries.append([None] * self.f_fh_dim) #Last Row
        self.f_fh_entries = [x + [None] for x in self.f_fh_entries] #Last Column
        self.f_fh_entries = [[None] + x for x in self.f_fh_entries] #First Column

        self.f_fh_dim = self.f_fh_dim + 2

        dim_interator = int((self.f_fh_dim-3)/2)

        #TODO: Implement it in a more elegant way
        #First Line
        for i in range(-1*(dim_interator-1), 3+(dim_interator-1)):
            self.filter_fh_val[0][i+(dim_interator)] = tk.DoubleVar();
            f_fh_entry = tk.Entry(self.f_fh_wm, bd = 1, textvariable=self.filter_fh_val[0][i+(dim_interator)])
            f_fh_entry.place(x=i*f_fh_ent_dx+f_fh_ent_x, y=f_fh_ent_y-(((self.f_fh_dim-3)/2)*f_fh_ent_dy), height = 20, width = 30)
            self.f_fh_entries[0][i+(dim_interator)] = f_fh_entry
        #Last Line
        for i in range(-1*(dim_interator-1), 3+(dim_interator-1)):
            self.filter_fh_val[self.f_fh_dim-1][i+(dim_interator)] = tk.DoubleVar();
            f_fh_entry = tk.Entry(self.f_fh_wm, bd = 1, textvariable=self.filter_fh_val[self.f_fh_dim-1][i+(dim_interator)])
            f_fh_entry.place(x=i*f_fh_ent_dx+f_fh_ent_x, y=(2+((self.f_fh_dim-3)/2))*f_fh_ent_dy+f_fh_ent_y, height = 20, width = 30)
            self.f_fh_entries[self.f_fh_dim-1][i+(dim_interator)] = f_fh_entry

        #First Column
        for j in range(-1*dim_interator, self.f_fh_dim-1*dim_interator):
            self.filter_fh_val[j+dim_interator][0] = tk.DoubleVar();
            f_fh_entry = tk.Entry(self.f_fh_wm, bd = 1, textvariable=self.filter_fh_val[j+dim_interator][0])
            f_fh_entry.place(x=f_fh_ent_x-(((self.f_fh_dim-3)/2)*f_fh_ent_dx), y=j*f_fh_ent_dy+f_fh_ent_y, height = 20, width = 30)
            self.f_fh_entries[j+dim_interator][0] = f_fh_entry
        #Last Column
        for j in range(-1*dim_interator, self.f_fh_dim-1*dim_interator):
            self.filter_fh_val[j+dim_interator][self.f_fh_dim-1] = tk.DoubleVar();
            f_fh_entry = tk.Entry(self.f_fh_wm, bd = 1, textvariable=self.filter_fh_val[j+dim_interator][self.f_fh_dim-1])
            f_fh_entry.place(x=(2+((self.f_fh_dim-3)/2))*f_fh_ent_dx+f_fh_ent_x, y=j*f_fh_ent_dy+f_fh_ent_y, height = 20, width = 30)
            self.f_fh_entries[j+dim_interator][self.f_fh_dim-1] = f_fh_entry

    def sub_filters_dim(self):

        f_fh_ent_dx  = 30
        f_fh_ent_dy  = 20
        f_fh_ent_x   = 100
        f_fh_ent_y   = 100

        if (self.f_fh_dim <= 3):
            print("> ERROR: Min dimension is 3")
            return -1

        #Delete New dimensions to the value list
        self.filter_fh_val.remove(self.filter_fh_val[self.f_fh_dim-1])
        self.filter_fh_val.remove(self.filter_fh_val[0])
        [x.pop(self.f_fh_dim-1) for x in self.filter_fh_val]
        [x.pop(0) for x in self.filter_fh_val]


        #Last Line
        for i in self.f_fh_entries[self.f_fh_dim-1]:
            i.destroy();

        #First Line
        for i in self.f_fh_entries[0]:
            i.destroy();

        self.f_fh_entries.remove(self.f_fh_entries[self.f_fh_dim-1])
        self.f_fh_entries.remove(self.f_fh_entries[0])

        #Columns
        for i in self.f_fh_entries:
            i[0].destroy();
            i[self.f_fh_dim-1].destroy();

        [x.pop(self.f_fh_dim-1) for x in self.f_fh_entries]
        [x.pop(0) for x in self.f_fh_entries]

        self.f_fh_dim = self.f_fh_dim - 2


    def build_hist_eq_wm(self):

        he_wm = tk.Toplevel(self.wm)
        he_wm.title("Histogram Equalization")
        he_wm.geometry("70x50")

        if (self.soin_img_hand.is_rgb):
            result = np.array([None, None, None])
            for i in range(0, self.soin_img_hand.img_array.shape[2]):
                result[i] = self.img_proc_hand.hist_func(self.soin_img_hand.img_array[:,:,i]) #Call Processor function
        else:
            result = self.img_proc_hand.hist_func(self.soin_img_hand.img_array)
        
        self.hist_plot(result)
        he_wm_apply_bt    = tk.Button(he_wm, width = 3, height = 1, text = "Apply", command = self.hist_eq_call)
        he_wm_apply_bt.place(x=10, y=10)

    def plot(self):
        if (self.plot_wm != None):
            self.plot_wm.destroy()
            self.plot_wm = None

        self.plot_wm = tk.Toplevel(self.wm)
        self.plot_wm.title("Free Hand Function View")

        point1 = [int(self.fh_x1.get()), int(self.fh_y1.get())]
        point2 = [int(self.fh_x2.get()), int(self.fh_y2.get())]
        
        points = np.array([point1, point2])
        
        xnew = np.linspace(0, 255, num=256, endpoint=True)
        xnew = xnew.astype(int)

        result = self.img_proc_hand.free_func([xnew], points)

        fig = Figure(figsize=(6,4))
        a = fig.add_subplot(111)
        a.plot(xnew, result, color='blue')

        a.set_title ("Free Hand Function View", fontsize=16)
        a.set_ylabel("Output Image", fontsize=14)
        a.set_xlabel("Input Image", fontsize=14)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_wm)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def hist_plot(self, img_hist):
        #if (self.hist_plot_wm != None):
        #    self.hist_plot_wm.withdraw()

        self.hist_plot_wm = tk.Toplevel(self.wm)
        self.hist_plot_wm.title("Histogram Plot View")
        #self.hist_plot_wm.geometry("900x900")
        
        fig = Figure(figsize=(10,6))

        if (self.soin_img_hand.is_rgb):
            
            r = fig.add_subplot(311)
            #r.hist(img_hist[0], bins = 256, color = "red")
            y_pos = np.arange(len(img_hist[0]))
            r.bar(y_pos,img_hist[0], color="red")
            r.set_title ("Red Histogram", fontsize=4)
    
            g = fig.add_subplot(312)
            #g.hist(img_hist[1], bins = 256, color = "green")
            y_pos = np.arange(len(img_hist[1]))
            g.bar(y_pos,img_hist[1], color="green")
            g.set_title ("Gree Histogram", fontsize=4)
    
            b = fig.add_subplot(313)
            #b.hist(img_hist[2], bins = 256, color = "blue")
            y_pos = np.arange(len(img_hist[2]))
            b.bar(y_pos,img_hist[2], color="blue")
            b.set_title ("Blue Histogram", fontsize=4)
        else:
            y_pos = np.arange(len(img_hist))
            gr = fig.add_subplot(111)
            gr.bar(y_pos,img_hist)
            gr.set_title ("Histogram Plot View", fontsize=16)

        canvas = FigureCanvasTkAgg(fig, master=self.hist_plot_wm)
        canvas.get_tk_widget().pack()
        canvas.draw()



    def filter_call (self, filter_sel):
        if (self.soin_img_hand.is_rgb):
            for i in range(0, self.soin_img_hand.img_array.shape[2]):
                #self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.free_func(self.soin_img_hand.img_array[:,:,i], points) #Call Processor function
                if (filter_sel == 1):
                    self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.average_smothing_func(self.soin_img_hand.img_array[:,:,i], self.filter_dim.get())
                elif (filter_sel == 2):
                    self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.gaussian_smothing_func(self.soin_img_hand.img_array[:,:,i], self.filter_dim.get())
                elif (filter_sel == 3):
                    #self.soin_img_hand.img_array = self.img_proc_hand.laplacian_sharpening_func(self.soin_img_hand.img_array[:,:,i])
                    print("> This feature is not implemented for colorized images.")
                elif (filter_sel == 4):
                    self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.highboost_sharpening_func(self.soin_img_hand.img_array[:,:,i], self.filter_dim.get(), self.filter_fac.get())
                elif (filter_sel == 5):
                    self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.median_smothing_func(self.soin_img_hand.img_array[:,:,i], self.filter_dim.get())
                elif (filter_sel == 6):
                    self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.sobel_edge_detec_func(self.soin_img_hand.img_array[:,:,i])
                elif (filter_sel == 7):
                    self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.geometric_mean_smothing_func(self.soin_img_hand.img_array[:,:,i], self.filter_dim.get())
                elif (filter_sel == 8):
                    self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.harmonic_mean_smothing_func(self.soin_img_hand.img_array[:,:,i], self.filter_dim.get())
        else:

            if (filter_sel == 1):
                self.soin_img_hand.img_array = self.img_proc_hand.average_smothing_func(self.soin_img_hand.img_array, self.filter_dim.get())
            elif (filter_sel == 2):
                self.soin_img_hand.img_array = self.img_proc_hand.gaussian_smothing_func(self.soin_img_hand.img_array, self.filter_dim.get())
            elif (filter_sel == 3):
                self.soin_img_hand.img_array = self.img_proc_hand.laplacian_sharpening_func(self.soin_img_hand.img_array)
            elif (filter_sel == 4):
                self.soin_img_hand.img_array = self.img_proc_hand.highboost_sharpening_func(self.soin_img_hand.img_array, self.filter_dim.get(), self.filter_fac.get())
            elif (filter_sel == 5):
                self.soin_img_hand.img_array = self.img_proc_hand.median_smothing_func(self.soin_img_hand.img_array, self.filter_dim.get())
            elif (filter_sel == 6):
                self.soin_img_hand.img_array = self.img_proc_hand.sobel_edge_detec_func(self.soin_img_hand.img_array)
            elif (filter_sel == 7):
                self.soin_img_hand.img_array = self.img_proc_hand.geometric_mean_smothing_func(self.soin_img_hand.img_array, self.filter_dim.get())
            elif (filter_sel == 8):
                self.soin_img_hand.img_array = self.img_proc_hand.harmonic_mean_smothing_func(self.soin_img_hand.img_array, self.filter_dim.get())


        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def hist_eq_call(self):

        if (self.soin_img_hand.is_rgb):
            self.soin_img_hand.hsv_update()
            self.soin_img_hand.img_hsv[:,:,2] = self.img_proc_hand.hist_eq_func(self.soin_img_hand.img_hsv[:,:,2])
            self.soin_img_hand.rgb_update()
            result = [None, None, None]
            for i in range(0, self.soin_img_hand.img_array.shape[2]):
                result[i] = self.img_proc_hand.hist_func(self.soin_img_hand.img_array[:,:,i])

        else:
            self.soin_img_hand.img_array = self.img_proc_hand.hist_eq_func(self.soin_img_hand.img_array)
            result = self.img_proc_hand.hist_func(self.soin_img_hand.img_array)
        

        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image
        self.hist_plot_wm.destroy()
        self.hist_plot_wm = None
        self.hist_plot(result)
        

    def log_call(self):
        self.soin_img_hand.img_array = self.img_proc_hand.log_func(self.soin_img_hand.img_array, self.log_factor.get()) #Call Processor function
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def power_call(self):
        self.soin_img_hand.img_array = self.img_proc_hand.power_func(self.soin_img_hand.img_array, self.power_factor.get(), self.power_exponent.get()) #Call Processor function
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image


    def scale_call(self, mode=0):
        if (self.soin_img_hand.is_rgb):
            result = None
            for i in range(0, self.soin_img_hand.img_array.shape[2]):
                temp_result = self.img_proc_hand.scale(self.soin_img_hand.img_array[:,:,i], self.scale_factor.get(), mode) #Call Processor function
                if (type(result).__module__ != np.__name__):#Find a better way to do that
                    result = temp_result 
                else:
                    result = np.dstack((result, temp_result))
                 #self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.scale(self.soin_img_hand.img_array[:,:,i], self.scale_factor.get(), mode) #Call Processor function
            self.soin_img_hand.img_array = result
        else:
            self.soin_img_hand.img_array = self.img_proc_hand.scale(self.soin_img_hand.img_array, self.scale_factor.get(), mode) #Call Processor function
        
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image


    def rotate_call(self, mode=0):
        if (self.soin_img_hand.is_rgb):
            result = None
            for i in range(0, self.soin_img_hand.img_array.shape[2]):
                temp_result = self.img_proc_hand.rotate(self.soin_img_hand.img_array[:,:,i], self.rotate_factor.get()) #Call Processor function
                if (type(result).__module__ != np.__name__):#Find a better way to do that
                    result = temp_result 
                else:
                    result = np.dstack((result, temp_result))
                 #self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.rotate(self.soin_img_hand.img_array[:,:,i], self.rotate_factor.get(), mode) #Call Processor function
            self.soin_img_hand.img_array = result
        else:
            self.soin_img_hand.img_array = self.img_proc_hand.rotate(self.soin_img_hand.img_array, self.rotate_factor.get()) #Call Processor function
        
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def negative_call(self):
        self.soin_img_hand.img_array = self.img_proc_hand.negative_func(self.soin_img_hand.img_array) #Call Processor function
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def free_hand_call(self):
        point1 = [int(self.fh_x1.get()), int(self.fh_y1.get())]
        point2 = [int(self.fh_x2.get()), int(self.fh_y2.get())]
        points = np.array([point1, point2])

        if (self.soin_img_hand.is_rgb):
            for i in range(0, self.soin_img_hand.img_array.shape[2]):
                self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.free_func(self.soin_img_hand.img_array[:,:,i], points) #Call Processor function
        else:
            self.soin_img_hand.img_array = self.img_proc_hand.free_func(self.soin_img_hand.img_array, points) #Call Processor function
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def steganography_enc_call(self, phrase):
        if (self.soin_img_hand.is_rgb):
            self.soin_img_hand.img_array[:,:,0] = self.img_proc_hand.steganography_encode(self.soin_img_hand.img_array[:,:,0], phrase)
        else:
            self.soin_img_hand.img_array = self.img_proc_hand.steganography_encode(self.soin_img_hand.img_array, phrase)
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def steganography_dec_call(self):
        if (self.soin_img_hand.is_rgb):
            phrase = self.img_proc_hand.steganography_decode(self.soin_img_hand.img_array[:,:,0])
        else:
            phrase = self.img_proc_hand.steganography_decode(self.soin_img_hand.img_array)
        self.steg_e.insert(0, phrase)

    def conv_free_hand_call(self):
        filter = np.zeros([self.f_fh_dim, self.f_fh_dim])

        for i in range(0,self.f_fh_dim):
            for j in range(0, self.f_fh_dim):
                filter[i][j] = self.filter_fh_val[i][j].get()

        if (self.soin_img_hand.is_rgb):
            for i in range(0, self.soin_img_hand.img_array.shape[2]):
                self.soin_img_hand.img_array[:,:,i] = self.img_proc_hand.cov_func(self.soin_img_hand.img_array[:,:,i], filter) #Call Processor function
        else:
            self.soin_img_hand.img_array = self.img_proc_hand.cov_func(self.soin_img_hand.img_array, filter) #Call Processor function
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def fft_call(self):
        if (self.soin_img_hand.is_rgb):
            print("> This features is not implemented for colorized image.")
            return -1
        self.spec_result = np.fft.fftshift(self.img_proc_hand.fourier_hand.FFT_2D(self.soin_img_hand.img_array))#Call Processor function
        self.spec_result_abs = abs(self.spec_result) 
        self.spec_result_abs = self.img_proc_hand.log_func(self.spec_result_abs, 1)
        self.spec_result_abs = self.img_proc_hand.power_func(self.spec_result_abs, 1, 2)
        self.build_spectrum_wm(self.spec_result_abs)

    def free_hand_ifft_call(self):

        print(self.spec_cord_edit_buff.shape)
        if (len(self.spec_cord_edit_buff) <= 3):
            print("> Edit Buffer empty.")
            return -1

        pixels = np.array([None, None])

        for point in self.spec_cord_edit_buff:

            if (point[0] !=None and point[1] != None and point[2] != None):
                for r0 in range(point[1]-point[2],point[1]+point[2]):
                    for r1 in range(point[0]-point[2], point[0]+point[2]):
                        cord = np.array([r1, r0])
                        pixels = np.vstack((pixels, cord))
    
        img_result = self.img_proc_hand.free_hand_ifft(pixels, self.spec_result) #Call Processor function

        self.soin_img_hand.img_array = img_result
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def freq_filter_func_call(self, sel=0):
        img_result = self.img_proc_hand.freq_filter_func(self.spec_result, self.spec_radius_val.get(), self.spec_width_val.get(), sel)

        self.soin_img_hand.img_array = img_result
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

    def rgb2grayscale_call(self, method):

        if (self.soin_img_hand.is_rgb):
            img_result = self.img_proc_hand.rgb2gray(self.soin_img_hand.img_array, method)
    
            if (type(img_result).__module__ == np.__name__):#Find a better way to do that
                self.soin_img_hand.is_rgb = 0
                self.soin_img_hand.img_array = img_result
                self.soin_img_hand.update_image() #Update Soin Image 
                self.update_image(self.soin_img_hand) #Update Gui Image
        else:
            print("> The image is not a RGB one.")

    def sepia_call(self):

        if (self.soin_img_hand.is_rgb):
            img_result = self.img_proc_hand.sepia(self.soin_img_hand.img_array)
    
            if (type(img_result).__module__ == np.__name__):#Find a better way to do that
                self.soin_img_hand.img_array = img_result
                self.soin_img_hand.update_image() #Update Soin Image 
                self.update_image(self.soin_img_hand) #Update Gui Image
        else:
            print("> The image is not a RGB one.")

    def adj_call(self):
        if (self.soin_img_hand.is_rgb):
            self.soin_img_hand.hsv_update()
            img_result = self.img_proc_hand.adjustment(self.soin_img_hand.img_hsv, self.adj_h_factor.get(), self.adj_s_factor.get(), self.adj_v_factor.get())
            self.soin_img_hand.hsv_update()
    
            self.soin_img_hand.img_array = img_result
            self.soin_img_hand.update_image() #Update Soin Image 
            self.update_image(self.soin_img_hand) #Update Gui Image
        else:
            print("> The image is not a RGB one.")


    def save_compressed_file(self, path, method, depth = 1):
        
        if (self.soin_img_hand.img_height >= 65536 or self.soin_img_hand.img_width >= 65536):
            print("> Dimensions are bigger than the suported.")
            return -1


        print("> Compressing image ...")
        
        low_height = self.soin_img_hand.img_height
        high_height = self.soin_img_hand.img_height >> 8

        low_width = self.soin_img_hand.img_width
        high_width = self.soin_img_hand.img_width >> 8

        low_h_bin = self.bin_file_hand.get_bin_byte(low_height)
        high_h_bin = self.bin_file_hand.get_bin_byte(high_height)

        low_w_bin = self.bin_file_hand.get_bin_byte(low_width)
        high_w_bin = self.bin_file_hand.get_bin_byte(high_width)

        rgb   = str(self.soin_img_hand.is_rgb)

        depth_bin  = self.bin_file_hand.get_bin_byte(depth)[4:8]
        method_bin = self.bin_file_hand.get_bin_byte(method)[6:8]

        if (method == 0):
            img_bit_string = self.img_proc_hand.huffman_compress(self.soin_img_hand.img_array)
            final_bit_string = method_bin+high_h_bin + low_h_bin + high_w_bin + low_w_bin + rgb + img_bit_string
        elif(method == 1):
            img_bit_string = self.img_proc_hand.huffman_haar_compress(self.soin_img_hand.img_array, depth, 0)
            final_bit_string = method_bin+depth_bin + high_h_bin + low_h_bin + high_w_bin + low_w_bin + rgb + img_bit_string
        else:
            img_bit_string = self.img_proc_hand.huffman_haar_compress(self.soin_img_hand.img_array, depth, 1)
            final_bit_string = method_bin+depth_bin + high_h_bin + low_h_bin + high_w_bin + low_w_bin + rgb + img_bit_string

        self.bin_file_hand.write_file(path, final_bit_string)

        print("> Compression finished.")

    def open_compressed_file(self, path):
        print("> Decompressing image ...")
        
        enc_file = self.bin_file_hand.read_file(path)
        
        method_bin = '000000'+enc_file[0:2]
        method  = self.bin_file_hand.to_byte(method_bin)[0]

        start = 2

        if (method == 0):
            start += 0
        else:
            haar_depth_bin = '0000'+enc_file[start:start+4]
            haar_depth_val = self.bin_file_hand.to_byte(haar_depth_bin)[0]
            start += 4

        dimensions_bin = enc_file[start:start+33]
        height_bin     = dimensions_bin[0:16]
        width_bin      = dimensions_bin[16:32]
        rgb            = dimensions_bin[32]

        enc_huff_img   = enc_file[start+33:len(enc_file)]

        height_vec = self.bin_file_hand.to_byte(height_bin)
        width_vec  = self.bin_file_hand.to_byte(width_bin)

        height = (height_vec[0] << 8) | height_vec[1]
        width  = (width_vec[0] << 8) | width_vec[1]

        if (method == 0):
            self.soin_img_hand.img_array = self.img_proc_hand.huffman_decompress(enc_huff_img, height, width, rgb)
        elif (method == 1):
            self.soin_img_hand.img_array = self.img_proc_hand.huffman_haar_decompress(enc_huff_img, height, width, rgb, haar_depth_val, 0)
        else:
            self.soin_img_hand.img_array = self.img_proc_hand.huffman_haar_decompress(enc_huff_img, height, width, rgb, haar_depth_val, 1)
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image

        print("> Decompression finished.")


    def restore(self, event=None):
        self.soin_img_hand.img_array = self.soin_img_hand.img_orig
        self.soin_img_hand.update_image() #Update Soin Image 
        self.update_image(self.soin_img_hand) #Update Gui Image


    def dummy(self):
        print("I am a dummy method")

    def run_wm(self):
        self.wm.mainloop()