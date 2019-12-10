#TODO: - Is it possible extend laplacian sharpening?
#      - Fiz index problem from median smothing

#Imports
import numpy   as np #Numpy

class image_processor:

    filter_proc_hand = None
    fourier_hand = None
    huffman_hand = None
    haar_hand    = None

    def __init__(self, filter_proc_hand, fourier_hand, huffman_hand, haar_hand):
        self.filter_proc_hand = filter_proc_hand
        self.fourier_hand = fourier_hand
        self.huffman_hand = huffman_hand
        self.haar_hand = haar_hand

    def log_func(self, img_array, k):
        hand_array = img_array.astype(int) # Converting array to int type
        img_array[img_array == 0] = 1 # Replace 0s with 1s
        log_result  = k*np.log(img_array) # Log
        #norm_result = self.normalize(log_result) #Normalize
        return log_result.astype('uint8') #Return as byte

    def power_func(self, img_array, k, x):
        hand_array  = img_array.astype(int) # Converting array to int type
        pw_result   = k*(hand_array**x) # Power
        norm_result = self.normalize(pw_result) #Normalize
        return norm_result.astype('uint8') #Return as byte

    def negative_func(self, img_array, L=255):
        hand_array  = img_array.astype(int) # Converting array to int type
        neg_result = L - hand_array #Negative
        img_array[img_array < 0] = 0 # Replace negative values with 0s
        norm_result = self.normalize(neg_result) #Normalize
        return norm_result.astype('uint8') #Return as byte

    def steganography_encode(self, img_array, phrase):
        
        i_init = 0
        i_end = 8
        j = 0
        column_change = 0

        if((img_array.shape[0]*img_array.shape[1])/8 < len(phrase)+1):
            print("> Image has no enough space to store the sentence.")
            return -1

        phrase = phrase + "\n"

        for k in phrase:
            for i in range(i_init, i_end):
                img_array[j][i] = (img_array[j][i]&0xFE) | ((ord(k)>>(i%8)) & 1)
                if (i>=(img_array.shape[1]-1)):
                    j += 1
                    i_init = 0
                    i_end = 8
                    column_change = 1
            
            if (column_change != 1):
                i_init = i_end
                i_end = i_end + 8
            else:
                column_change = 0

        return img_array

    def steganography_decode(self, img_array):
        i_init = 0
        i_end = 8
        j = 0
        count = 0

        phrase = ""
        character = 0

        for i in img_array:
            for j in i:
                character = character | ((j & 1) << count)
                count += 1

                if (count > 7):
                    count = 0
                    if (chr(character) == '\n'):
                        return phrase
                    
                    phrase = phrase + chr(character)
                    character = 0

        return -1



    def free_func(self, img_array, points):
        hand_array     = img_array#.astype(int) # Converting array to int type
        flag0          = 0
        flag1          = 0
        line_result    = None
        pre_result     = None
        min_points_x   = np.amin(points[:,0])
        max_points_x   = np.amax(points[:,0])
    
        min_points_y   = np.amin(points[:,1])
        max_points_y   = np.amax(points[:,1])
    
        if (min_points_x == max_points_x):
            min_cordinates = np.where(points[:,1] == min_points_y)
            max_cordinates = np.where(points[:,1] == max_points_y)
        else:
            min_cordinates = np.where(points[:,0] == min_points_x)
            max_cordinates = np.where(points[:,0] == max_points_x)
    
        min_points_y_in= points [min_cordinates[0], 1][0]
        max_points_y_in= points [max_cordinates[0], 1][0]
    
        min_val = np.amin(hand_array)
        max_val = np.amax(hand_array)
    
        for i in hand_array:
            for j in i:
                if((j >= min_points_x) and (j <= max_points_x)):
                    pre_result = self.line_eq(j, points)
                elif((j > max_points_x) and (j <= max_val)):
                    pre_result = self.line_eq(j, np.array([[max_points_x, max_points_y_in], [max_val, max_val]]))
                else:
                    pre_result = self.line_eq(j, np.array([[min_val, min_val], [min_points_x, min_points_y_in]]))
        
                if (flag0 != 0):
                    line_result = np.append(line_result, pre_result)
                else:
                    line_result = pre_result
                    flag0 = 1

            flag0 = 0
            if (flag1 != 0):
                result = np.vstack((result, line_result))
            else:
                result = line_result
                flag1 = 1
    
        norm_result = self.normalize(result) #Normalize
        return norm_result.astype('uint8') #Return as byte

    def hist_func(self, img_array):
        hand_array  = img_array.astype(int) # Converting array to int type
        hist        = np.zeros(np.max(hand_array)+1)
        for i in hand_array:
            for j in i:
                hist[j] = hist[j] + 1;
        norm_result = self.normalize(hist) #Normalize
        return hist.astype('uint8')

    def hist_eq_func(self, img_array):
        hand_array  = img_array.astype(int) # Converting array to int type
        hist = self.hist_func(img_array)
        prob = hist/(np.sum(hist)) #Probability
        ac_prob = np.zeros(np.max(hand_array)+1)#Accumulate Probability

        ac_prob[0] = prob[0]
        for i in range(1, len(prob)):
            ac_prob[i] = prob[i] + ac_prob[i-1] 

        
        ac_prob_result = self.normalize(ac_prob, np.max(hand_array)).astype('uint8') #Normalize

        for i in range(0, len(hand_array)):
            for j in range(0, len(hand_array[i])):
                hand_array[i][j] = ac_prob_result[hand_array[i][j]]

        return self.normalize(hand_array, np.max(hand_array)).astype('uint8') #Normalize

    def average_smothing_func(self, img_array, dim):
        filter     = self.filter_proc_hand.average_smothing_mask(dim)
        img_result = self.filter_proc_hand.filter_cov_apply(filter, img_array)

        return self.normalize(img_result).astype('uint8') #Normalize

    def gaussian_smothing_func(self, img_array, dim):
        filter     = self.filter_proc_hand.gaussian_smothing_mask(dim)
        img_result = self.filter_proc_hand.filter_cov_apply(filter, img_array)
        
        return self.normalize(img_result).astype('uint8') #Normalize

    def laplacian_sharpening_func(self, img_array):
        filter     = self.filter_proc_hand.laplacian_mask
        img_result = self.filter_proc_hand.filter_cov_apply(filter, img_array)
        
        return self.normalize(img_result).astype('uint8') #Normalize

    def highboost_sharpening_func(self, img_array, dim, factor):
        img_result = self.filter_proc_hand.highboost_sharpening(dim, factor, img_array)

        return self.normalize(img_result).astype('uint8') #Normalize

    def median_smothing_func(self, img_array, dim):
        img_result = self.filter_proc_hand.median_smothing(dim, img_array)

        return self.normalize(img_result).astype('uint8') #Normalize

    def sobel_edge_detec_func(self, img_array):
        img_result = self.filter_proc_hand.sobel_edge_detec(img_array)

        return self.normalize(img_result).astype('uint8') #Normalize

    def geometric_mean_smothing_func(self, img_array, dim):
        img_result = self.filter_proc_hand.geometric_mean_smothing(img_array, dim)

        return self.normalize(img_result).astype('uint8') #Normalize

    def harmonic_mean_smothing_func(self, img_array, dim):
        img_result = self.filter_proc_hand.harmonic_mean_smothing(img_array, dim)

        return self.normalize(img_result).astype('uint8') #Normalize

    def cov_func(self, filter, img_array):
        img_result = self.filter_proc_hand.filter_cov_apply(img_array, filter)

        return self.normalize(img_result).astype('uint8') #Normalize

    def ift_func(self, img_array):
        img_result = abs(self.fourier_hand.IFFT_2D(img_array))
        return self.normalize(img_result).astype('uint8') #Normalize

    def free_hand_ifft(self, pixels, img_array):

        for pixel in pixels:
            try:
                if (pixel[0]<img_array.shape[0] and pixel[1]<img_array.shape[1]):
                    img_array[pixel[0]][pixel[1]] = 0 +0j
            except TypeError:
                pass

        img_result = abs(self.fourier_hand.IFFT_2D(img_array))
        return self.normalize(img_result).astype('uint8') #Normalize

    def freq_filter_func(self, img_array, radius = 1, width = 1, sel = 0):
        if (sel ==0):
            img_result = self.fourier_hand.low_pass(img_array, radius, 0)
        elif(sel == 1):
            img_result = self.fourier_hand.high_pass(img_array, radius, 0)
        elif(sel == 2):
            img_result = self.fourier_hand.band_pass(img_array, radius, width, 0)
        elif(sel == 3):
            img_result = self.fourier_hand.band_reject(img_array, radius, width, 0)
        elif(sel == 4):
            img_result = self.fourier_hand.low_pass(img_array, radius, 1)
        elif(sel == 5):
            img_result = self.fourier_hand.high_pass(img_array, radius, 1)
        elif(sel == 6):
            img_result = self.fourier_hand.band_pass(img_array, radius, width, 1)
        elif(sel == 7):
            img_result = self.fourier_hand.band_reject(img_array, radius, width, 1)
        else:
            img_result = img_array

        img_result = abs(self.fourier_hand.IFFT_2D(img_result))
        return self.normalize(img_result).astype('uint8') #Normalize

    def rgb2hsv(self, rgb):
        r, g, b = rgb
        r, g, b = r / 255.0, g / 255.0, b / 255.0

        max_color = np.max([r, g, b])    # maximum 
        min_color = np.min([r, g, b])    # minimum
        diff = max_color-min_color
    
        #if max_color and max_color are qual there is no collor difference, so hue = 0 
        if max_color == min_color:  
            h = 0
        elif max_color == r:  # Red
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_color == g: # Green
            h = (60 * ((b - r) / diff) + 120) % 360
        elif max_color == b: # Blue
            h = (60 * ((r - g) / diff) + 240) % 360
    
        if max_color == 0: 
            s = 0
        else: 
            s = (diff / max_color) * 100
    
        # Velu: 
        v = max_color * 100
        
        return [h, s, v]

    def dhwt_func(self, img_array, depth = 1):

        if (len(img_array.shape) != 3):
            result = self.haar_hand.DHWT_2D(img_array, depth)
        else:
            result = np.zeros((img_array.shape[0], img_array.shape[1], img_array.shape[2]))
            for i in range(0, img_array.shape[2]):
                result[:,:,i] = self.haar_hand.DHWT_2D(img_array[:,:,i], depth)
        #return self.normalize(result).astype('uint8')
        return result


    def idhwt_func(self, img_array, depth = 1):

        #result = img_array

        #print(img_array.shape)
        if (len(img_array.shape) != 3):
            result = self.haar_hand.IDHWT_2D(img_array, depth)
        else:
            result = np.zeros((img_array.shape[0], img_array.shape[1], img_array.shape[2]))
            for i in range(0, img_array.shape[2]):
                result[:,:,i] = self.haar_hand.IDHWT_2D(img_array[:,:,i], depth)

            #result = np.reshape(result, (img_array.shape[0], img_array.shape[1], 3))

        print(np.min(result))
        return self.normalize(result).astype('uint8')



    def hsv2rgb(self, hsv):
        h,s,v = hsv
        r,g,b = 0,0,0

        #Normalize h, s, v
        s = s/100
        v = v/100

        c = v*s
        x = c*(1-abs((h/60)%2-1))
        m = v - c

        if   (h>=0 and h<60):
            (r, g, b) = (c, x, 0)
        elif (h>=60 and h<120):
            (r, g, b) = (x, c, 0)
        elif (h>=120 and h<180):
            (r, g, b) = (0, c, x)
        elif (h>=180 and h<240):
            (r, g, b) = (0, x, c)
        elif (h>=240 and h<300):
            (r, g, b) = (x, 0, c)
        elif (h>=300 and h<360):
            (r, g, b) = (c, 0, x)

        r, g, b = (r+m)*255.0, (g+m)*255.0, (b+m)*255.0

        return [r, g, b]

    def rgb2gray(self, img_array, method = 0):

        if (len(img_array.shape) != 3):
            print("> The image is not a RGB one.")
            return None

        hand_array = np.zeros([img_array.shape[0], img_array.shape[1]])

        for i in range(0, img_array.shape[0]):
            for j in range(0, img_array.shape[1]):
                if (method == 0):
                    hand_array[i][j] = np.sum(img_array[i,j,:])/3
                else:
                    hand_array[i][j] = int(img_array[i][j][0]*0.21 + img_array[i][j][0]*0.72 + img_array[i][j][0]*0.07)

        return self.normalize(hand_array).astype('uint8') #Normalize

    #Dirty way
    def gray2rgb(self, img_array):

        if (len(img_array.shape) != 2):
            print("> The image is not a Graysacale one.")
            return None

        hand_array = np.zeros([img_array.shape[0], img_array.shape[1],3])

        for i in range(0, img_array.shape[0]):
            for j in range(0, img_array.shape[1]):
                hand_array[i,j,:] = img_array[i,j]

        return self.normalize(hand_array).astype('uint8') #Normalize


    def sepia(self, img_array):
        
        if (len(img_array.shape) != 3):
            print("> The image is not a RGB one.")
            return None

        hand_array = np.zeros([img_array.shape[0], img_array.shape[1], img_array.shape[2]])

        for i in range(0, img_array.shape[0]):
            for j in range(0, img_array.shape[1]):
                tr = int(img_array[i][j][0]*0.393 + img_array[i][j][0]*0.769 + img_array[i][j][0]*0.769)
                tg = int(img_array[i][j][0]*0.349 + img_array[i][j][0]*0.686 + img_array[i][j][0]*0.168)
                tb = int(img_array[i][j][0]*0.272 + img_array[i][j][0]*0.534 + img_array[i][j][0]*0.131)

                if (tr > 255):
                    hand_array[i][j][0] = 255
                else:
                    hand_array[i][j][0] = tr

                if (tg > 255):
                    hand_array[i][j][1] = 255
                else:
                    hand_array[i][j][1] = tg

                if (tb > 255):
                    hand_array[i][j][2] = 255
                else:
                    hand_array[i][j][2] = tb
                    
        return self.normalize(hand_array).astype('uint8') #Normalize


    def adjustment(self, img_array, factor_h=1, factor_s=1, factor_v=1):
        
        if (len(img_array.shape) != 3):
            print("> The image is not a HSV one.")
            return None

        img_array[:,:,0] = factor_h*img_array[:,:,0]
        img_array[:,:,1] = factor_s*img_array[:,:,1]
        img_array[:,:,2] = factor_v*img_array[:,:,2]

        img_hand = np.zeros([img_array.shape[0], img_array.shape[1], 3])

        for i in range(0, img_array.shape[0]):
            for j in range(0, img_array.shape[1]):
                img_hand[i,j,:] = self.hsv2rgb(img_array[i,j,:])

        return self.normalize(img_hand).astype('uint8') #Normalize

    def chroma_key(self, img_array, background_array, reference, radius):
        if (len(img_array.shape) != 3):
            print("> The image is not a RGB one.")
            return None

        if (background_array.shape[0] < img_array.shape[0] or background_array.shape[1] < img_array.shape[1]):
            print("> One of the background dimensions is inferior to the front image.")
            return None

        for i in range(0, img_array.shape[0]):
            for j in range(0, img_array.shape[1]):
                d = self.distance(img_array[i,j,:], reference)
                if(d<radius):
                    img_array[i,j,:] = background_array[i,j,:]

        return self.normalize(img_array).astype('uint8') #Normalize

    def scale(self, img_array, factor, mode=0):

        hand_array = np.zeros([int(np.floor(img_array.shape[0]*factor)), int(np.floor(img_array.shape[1]*factor))])

        if (mode == 0 or factor < 1):
            for i in range(0,hand_array.shape[0]):
                for j in range(0, hand_array.shape[1]):
                     hand_array[i][j] = img_array[int(i/factor)][int(j/factor)]
        else:
            factor = int(np.ceil(factor))
            for i in range(0,hand_array.shape[0], int(factor)):
                for j in range(0, hand_array.shape[1], int(factor)):
                    hand_array[i][j] = img_array[int(i/factor)][int(j/factor)]
            
            for i in range(0, hand_array.shape[0], int(factor)):
                for j in range(0, hand_array.shape[1]):

                    y0 = int(int(j/factor)*factor)
                    y1 = int(y0 + factor)

                    if (y1 >= hand_array.shape[1]):
                        hand_array[i][j] = hand_array[i][y0] + ((j - y0)/(y1 - y0))*(0 - hand_array[i][y0])
                    else:
                        hand_array[i][j] = hand_array[i][y0] + ((j - y0)/(y1 - y0))*(hand_array[i][y1] - hand_array[i][y0])

            for j in range(0, hand_array.shape[1]):
                for i in range(0, hand_array.shape[0]):
                
                    x0 = int(int(i/factor)*factor)
                    x1 = int(x0 + factor)

                    if (x1 >= hand_array.shape[0]):
                        hand_array[i][j] = hand_array[x0][j] + ((i - x0)/(x1 - x0))*(0 - hand_array[x0][j])
                    else:
                        hand_array[i][j] = hand_array[x0][j] + ((i - x0)/(x1 - x0))*(hand_array[x1][j] - hand_array[x0][j])


        return self.normalize(hand_array).astype('uint8') #Normalize

    def rotate(self, img_array, angle):

        angle_radians = float(angle*(np.pi/180))
        width, height = img_array.shape

        if (angle<0 and angle>=-90):
            n_height = abs(int(height*np.sin((np.pi/2)-angle_radians) - width*np.sin(angle_radians)))
            n_width  = abs(int(width*np.cos(angle_radians) - height*np.cos((np.pi/2)-angle_radians)))
        elif(angle<-180):
            n_height = abs(int(height*np.sin((np.pi/2)-angle_radians) - width*np.sin(angle_radians)))
            n_width  = abs(int(width*np.cos(angle_radians) - height*np.cos((np.pi/2)-angle_radians)))
        elif(angle>=-180 and angle<-90):
            n_height = abs(int(height*np.sin((np.pi/2)-angle_radians) + width*np.sin(angle_radians)))
            n_width  = abs(int(width*np.cos(angle_radians) + height*np.cos((np.pi/2)-angle_radians)))
        elif (angle>=0 and angle<=90):
            n_height = abs(int(height*np.sin((np.pi/2)-angle_radians) + width*np.sin(angle_radians)))
            n_width  = abs(int(width*np.cos(angle_radians) + height*np.cos((np.pi/2)-angle_radians)))
        elif (angle>90 and angle<=180):
            n_height = abs(int(height*np.sin((np.pi/2)-angle_radians) - width*np.sin(angle_radians)))
            n_width  = abs(int(width*np.cos(angle_radians) - height*np.cos((np.pi/2)-angle_radians)))
        else:
            n_height = abs(int(height*np.sin((np.pi/2)-angle_radians) + width*np.sin(angle_radians)))
            n_width  = abs(int(width*np.cos(angle_radians) + height*np.cos((np.pi/2)-angle_radians)))

        hand_array = np.zeros([n_width, n_height])

        for i in range(0, n_width):
            for j in range(0, n_height):
                nx = int((i-(n_width/2))*np.cos(angle_radians) + (j-(n_height/2))*np.sin(angle_radians) + width/2)
                ny = int((j-(n_height/2))*np.cos(angle_radians) - (i-(n_width/2))*np.sin(angle_radians) + height/2)
                if (nx > 0 and nx < width and ny > 0 and ny < height ):
                    hand_array[i][j] = img_array[nx][ny]

        return self.normalize(hand_array).astype('uint8') #Normalize


    def huffman_haar_compress(self, img_array, depth=1, to_gray=0):
        hand_image_array = self.normalize(self.dhwt_func(img_array, depth)).astype('uint8')
        
        if (to_gray == 1):
            if (len(img_array.shape) != 3):
                print("> This options is available only for RGB images.")
                return -1

            origin, comp = self.haar_separate_components(hand_image_array, depth)
            enc_word,root,n_leaves = self.huffman_haar_to_gray_encode(comp, origin)
        else:
            enc_word,root,n_leaves = self.huffman_encode(hand_image_array)

        complete_word_enc = self.huffman_hand.huff2bit(enc_word, root, n_leaves)
        return complete_word_enc

    def huffman_compress(self, img_array):
        enc_word,root,n_leaves = self.huffman_encode(img_array)
        complete_word_enc = self.huffman_hand.huff2bit(enc_word, root, n_leaves)
        return complete_word_enc

    def huffman_haar_decompress(self, encode_array, height, width, rgb, depth=1, to_gray=0):
        
        if (rgb != '1'):
            img_hand = np.zeros([height, width])
        else:
            img_hand = np.zeros([height, width, 3])

        dec_word = self.huffman_decode(encode_array)

        #TODO: Use the same loop
        if (rgb != '1'):
            for i in range(0,len(dec_word)):
                if (int(i/width) > (height-1) or int(i%width) > (width-1)):
                    break
                img_hand[int(i/width)][int(i%width)] = dec_word[i]

        else:
            if (to_gray == 0):  
                for k in range(0,3):
                    for i in range(0,height):
                        for j in range(0,width):
                            if ((j+i*width+(k*width*height)) < len(dec_word)): #HACK: (Something is going wrong, the writed file has the wrong size sometimes)
                                img_hand[i][j][k] = dec_word[j+i*width+(k*width*height)]
            
            #Haar with grayscale:
            else:
                origin_height = int(height/2**depth)
                origin_width = int(width/2**depth)

                origin_img_hand = np.zeros([origin_height, origin_width, 3])
                componets_img_hand = np.zeros([height, width])
                for k in range(0,3):
                    for i in range(0,origin_height):
                        for j in range(0,origin_width):
                            origin_img_hand[i][j][k] = dec_word[j+i*origin_width+(k*origin_width*origin_height)]

                for i in range(origin_height*origin_width*3,len(dec_word)):
                    if (int(i/width) > (height-1) or int(i%width) > (width-1)):
                        break
                    componets_img_hand[int(i/width)][int(i%width)] = dec_word[i]

                img_hand = self.haar_join_components(origin_img_hand, componets_img_hand, depth)


        img_hand = self.idhwt_func(img_hand, depth)

        return self.normalize(img_hand).astype('uint8') #Normalize


    def huffman_decompress(self, encode_array, height, width, rgb):
        
        if (rgb != '1'):
            img_hand = np.zeros([height, width])
        else:
            img_hand = np.zeros([height, width, 3])

        dec_word = self.huffman_decode(encode_array)

        #TODO: Use the same loop
        if (rgb != '1'):
            for i in range(0,len(dec_word)):
                if (int(i/width) > (height-1) or int(i%width) > (width-1)):
                    break
                img_hand[int(i/width)][int(i%width)] = dec_word[i]

        else:
            for k in range(0,3):
                for i in range(0,height):
                    for j in range(0,width):
                        img_hand[i][j][k] = dec_word[j+i*width+(k*width*height)]

        #print(img_hand)

        return self.normalize(img_hand).astype('uint8') #Normalize

    def huffman_encode(self, img_array):

        if (len(img_array.shape) != 3):
            height, width = img_array.shape
            img_flat_array = img_array.reshape((1, height*width))[0]
        else:
            height, width, depth = img_array.shape
            img_flat_array = np.zeros(height*width*depth)
            #HACK: The reshape was not working
            for k in range(0, depth):
                for i in range(0, height):
                    for j in range(0, width):
                        img_flat_array[j + i*width + k*height*width] = int(img_array[i][j][k])

        return self.huffman_hand.encode(img_flat_array.astype('uint8'))

    def huffman_haar_to_gray_encode(self, components, original):

        height, width, depth = original.shape

        origin_img_flat_array = np.zeros(height*width*depth)
        components_img_flat_array = components.reshape((1, components.shape[0]*components.shape[1]))[0]

        for k in range(0, depth):
                for i in range(0, height):
                    for j in range(0, width):
                        origin_img_flat_array[j + i*width + k*height*width] = int(original[i][j][k])

        img_flat_array = np.hstack((origin_img_flat_array, components_img_flat_array))


        return self.huffman_hand.encode(img_flat_array.astype('uint8'))

    def huffman_decode(self, encode_array):
        enc_array, root = self.huffman_hand.bit2huff(encode_array)
        dec = self.huffman_hand.decode(enc_array, root)
        return dec

    def haar_separate_components(self, img_array, depth = 1):
        height = img_array.shape[0]
        width  = img_array.shape[1]
        #Original
        original   = img_array[0:int(height/2**depth), 0:int(width/2**depth)]
        #Components
        components = np.zeros((height, width, 3))
        
        components[0:int(height/2**depth), int(width/2**depth):width, :]      = img_array[0:int(height/2**depth), int(width/2**depth):width, :] #Top-Right
        components[int(height/2**depth):height, int(width/2**depth):width, :] = img_array[int(height/2**depth):height, int(width/2**depth):width, :] #Bottom-Right
        components[int(height/2**depth):height, 0:int(width/2**depth), :]     = img_array[int(height/2**depth):height, 0:int(width/2**depth), :] #Bottom-Left

        components = self.rgb2gray(components, 1)

        return (original, components)

    def haar_join_components(self, original, components, depth = 1):
        height = components.shape[0]
        width  = components.shape[1]

        result = np.zeros((height,width, 3))

        for i in range(0,3):
            result[:,:,i] = components[:,:]

        result[0:int(height/2**depth), 0:int(width/2**depth)] = original


        return result


    #Aux methods
    def distance(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**(1/2)

    def line_eq(self, val, points):
        if (points[1,0] != points[0,0]):
            ang_coef = np.absolute(points[1,1] - points[0,1])/np.absolute(points[1,0] - points[0,0])
            result = (ang_coef*val) -(ang_coef*points[0,0]) + points[0,1]
        else:
            if (val == points[1,0]):
                result = np.amax(points[:,1])
        return result.astype(int)

    def normalize(self, img_array, top=255):

        #print("Max: ", np.max(img_array))
        #print("Min: ", np.min(img_array))

        if (np.max(img_array) > 0):
            #if (np.min(img_array) < 0):
            #    img_array = img_array + (np.abs(np.min(img_array)))
            img_array[img_array < 0] = 0 # Replace negative with 0s
            return top*(img_array/np.max(img_array))
        return img_array
     

class filter_processor:
    
    laplacian_mask = np.array([[-1,-1,-1],[-1, 8, -1],[-1, -1, -1]])
    sobel_mask_y   = np.array([[-1,-2,-1],[0, 0, 0],[1, 2, 1]])
    sobel_mask_x   = np.array([[-1,0,1],[-2, 0, 2],[-1, 0, 1]])

    def __init__(self):
        pass

    def average_smothing_mask(self, dim):
        if (dim % 2 == 0):
            print("> ERROR: Filter has even dimension")
            return -1

        return np.ones([dim, dim])/(dim*dim)

    def gaussian_smothing_mask(self, dim):
        if (dim % 2 == 0):
            print("> ERROR: Filter has even dimension")
            return -1

        filter   = np.zeros([dim, dim])
        rest_dim = int((dim-1)/2)
        std_dev  = 1.0 #TODO: User argument? 

        #Buildig Filter
        for i in range(-rest_dim, rest_dim+1):
            for j in range(-rest_dim, rest_dim+1):
                filter[i+rest_dim][j+rest_dim] = (1/(2*np.pi*std_dev))*np.exp(-1*((i**2)+(j**2))/(2*(std_dev**2)))

        return filter

    #TODO: Choose gaussian dimension?
    def highboost_sharpening(self, dim, factor, img_array):
        if (dim % 2 == 0):
            print("> ERROR: Filter has even dimension")
            return -1

        filter     = self.gaussian_smothing_mask(dim)
        img_blur   = self.filter_cov_apply(filter, img_array)
        mask       = img_array - img_blur
        img_result = img_array + factor*mask

        return img_result

    #TODO: Choose dimension?
    def median_smothing(self, dim, img_array):
        if (dim % 2 == 0):
            print("> ERROR: Filter has even dimension")
            return -1

        hand_array = img_array.astype('uint8') # Converting array to int type
        rest_dim = int((dim-1)/2)

        vec_mask = np.zeros(dim*dim)

        #Adding extra dimensions
        #Columns
        hand_array = np.column_stack((hand_array, np.zeros([len(hand_array), rest_dim])))
        hand_array = np.column_stack((np.zeros([len(hand_array), rest_dim]), hand_array))
        #Rows
        hand_array = np.vstack((hand_array, np.zeros([rest_dim, len(hand_array[0])])))
        hand_array = np.vstack((np.zeros([rest_dim, len(hand_array[0])]), hand_array))

        #Image Result
        img_result = np.zeros([len(hand_array), len(hand_array[0])])

        for i in range(rest_dim, len(hand_array)-rest_dim):
            for j in range(rest_dim, len(hand_array[0])-rest_dim):
                for i_l in range(0, dim):
                    for j_l in range(0, dim):
                        vec_mask[i_l*dim + j_l] = hand_array[i-(rest_dim-i_l)][j-(rest_dim-j_l)]
                img_result[i][j] = np.median(vec_mask)

        return img_result[rest_dim:len(hand_array)-rest_dim, rest_dim:len(hand_array)-rest_dim]

    def geometric_mean_smothing(self, img_array, dim):
        if (dim % 2 == 0):
            print("> ERROR: Filter has even dimension")
            return -1

        hand_array = img_array.astype('uint8') # Converting array to int type
        rest_dim = int((dim-1)/2)

        vec_mask = np.zeros(dim*dim)

        #Adding extra dimensions
        #Columns
        hand_array = np.column_stack((hand_array, np.zeros([len(hand_array), rest_dim])))
        hand_array = np.column_stack((np.zeros([len(hand_array), rest_dim]), hand_array))
        #Rows
        hand_array = np.vstack((hand_array, np.zeros([rest_dim, len(hand_array[0])])))
        hand_array = np.vstack((np.zeros([rest_dim, len(hand_array[0])]), hand_array))

        #Image Result
        img_result = np.zeros([len(hand_array), len(hand_array[0])])

        for i in range(rest_dim, len(hand_array)-rest_dim):
            for j in range(rest_dim, len(hand_array[0])-rest_dim):
                for i_l in range(0, dim):
                    for j_l in range(0, dim):
                        vec_mask[i_l*dim + j_l] = hand_array[i-(rest_dim-i_l)][j-(rest_dim-j_l)]
                img_result[i][j] = ((np.prod(vec_mask))**(1/(dim*dim)))

        return img_result[rest_dim:len(hand_array)-rest_dim, rest_dim:len(hand_array)-rest_dim]

    def harmonic_mean_smothing(self, img_array, dim):
        if (dim % 2 == 0):
            print("> ERROR: Filter has even dimension")
            return -1

        hand_array = img_array.astype('uint8') # Converting array to int type
        rest_dim = int((dim-1)/2)

        vec_mask = np.zeros(dim*dim)

        #Adding extra dimensions
        #Columns
        hand_array = np.column_stack((hand_array, np.zeros([len(hand_array), rest_dim])))
        hand_array = np.column_stack((np.zeros([len(hand_array), rest_dim]), hand_array))
        #Rows
        hand_array = np.vstack((hand_array, np.zeros([rest_dim, len(hand_array[0])])))
        hand_array = np.vstack((np.zeros([rest_dim, len(hand_array[0])]), hand_array))

        #Image Result
        img_result = np.zeros([len(hand_array), len(hand_array[0])])

        for i in range(rest_dim, len(hand_array)-rest_dim):
            for j in range(rest_dim, len(hand_array[0])-rest_dim):
                for i_l in range(0, dim):
                    for j_l in range(0, dim):
                        if ((hand_array[i-(rest_dim-i_l)][j-(rest_dim-j_l)]) != 0):
                            vec_mask[i_l*dim + j_l] = 1/(hand_array[i-(rest_dim-i_l)][j-(rest_dim-j_l)])
                
                img_result[i][j] = len(vec_mask)/(np.sum(vec_mask))

        return img_result[rest_dim:len(hand_array)-rest_dim, rest_dim:len(hand_array)-rest_dim]


    def sobel_edge_detec(self, img_array):
        
        hand_array = img_array.astype('uint8') # Converting array to int type
        rest_dim = 1

        gx_result = 0
        gy_result = 0

        #Adding extra dimensions
        #Columns
        hand_array = np.column_stack((hand_array, np.zeros([len(hand_array), rest_dim])))
        hand_array = np.column_stack((np.zeros([len(hand_array), rest_dim]), hand_array))
        #Rows
        hand_array = np.vstack((hand_array, np.zeros([rest_dim, len(hand_array[0])])))
        hand_array = np.vstack((np.zeros([rest_dim, len(hand_array[0])]), hand_array))

        #Image Result
        img_result = np.zeros([len(hand_array), len(hand_array[0])])

        for i in range(rest_dim, len(hand_array)-rest_dim):
            for j in range(rest_dim, len(hand_array[0])-rest_dim):
                gx_result = (hand_array[i+1][j-1] + 2*hand_array[i+1][j] + hand_array[i+1][j+1]) - (hand_array[i-1][j-1] + 2*hand_array[i-1][j] + hand_array[i-1][j+1])
                gy_result = (hand_array[i-1][j+1] + 2*hand_array[i][j+1] + hand_array[i+1][j+1]) - (hand_array[i-1][j-1] + 2*hand_array[i][j-1] + hand_array[i+1][j-1])
                img_result[i][j] = np.sqrt((gx_result**2)+(gy_result**2))

        return img_result[rest_dim:len(hand_array)-rest_dim, rest_dim:len(hand_array[0])-rest_dim]


    def filter_cov_apply(self, filter, img_array):
        if (len(filter) != len(filter[0])):
            print("> ERROR: Filter has not the same dimensions!")
            return -1
        elif (len(filter) % 2 == 0):
            print("> ERROR: Filter has even dimension")
            return -1

        filter_dim = len(filter)
        
        hand_array = img_array.astype('uint8') # Converting array to int type
        rest_dim = int((filter_dim-1)/2)

        #Adding extra dimensions
        #Columns
        hand_array = np.column_stack((hand_array, np.zeros([len(hand_array), rest_dim])))
        hand_array = np.column_stack((np.zeros([len(hand_array), rest_dim]), hand_array))
        #Rows
        hand_array = np.vstack((hand_array, np.zeros([rest_dim, len(hand_array[0])])))
        hand_array = np.vstack((np.zeros([rest_dim, len(hand_array[0])]), hand_array))

        #Image Result
        img_result = np.zeros([len(hand_array), len(hand_array[0])])

        for i in range(rest_dim, len(hand_array)-rest_dim):
            for j in range(rest_dim, len(hand_array[0])-rest_dim):
                for i_l in range(0, len(filter)):
                    for j_l in range(0, len(filter[0])):
                        img_result[i][j] = img_result[i][j] + hand_array[i-(rest_dim-i_l)][j-(rest_dim-j_l)] * filter[i_l][j_l]

        return img_result[rest_dim:len(hand_array)-rest_dim, rest_dim:len(hand_array[0])-rest_dim]


class fourier():

    def __init__(self):
        pass
        
    def DFT(self, x):
        
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)
    
    def DFT_2D(self, f):
        
        f = np.asarray(f, dtype=complex)
        M = f.shape[0]
        N = f[0].shape[0]
        #B = np.exp(-2j * np.pi * ((v*y/N)+(u*x/M)))
    
        result = [[0.0 for k in range(M)] for l in range(N)]
        sum_result = 0
        for i in range(M):
            for j in range(N):
                sum_result = 0
                for k in range(M):
                    for l in range(N):
                        sum_result = sum_result + f[k,l]*np.exp(-2j * np.pi * ((l*j/N)+(k*i/M)))
                result[i][j] = sum_result
    
        return result
    
    
    def FFT(self, x):
        
        x = np.asarray(x, dtype=complex)

        N = x.shape[0]

        if (N <=2):
            return self.DFT(x)
        else:
            X_even = self.FFT(x[::2])
            X_odd  = self.FFT(x[1::2])

            factor = np.exp(-2j * np.pi * np.arange(N) / N)

            return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
                                   X_even + factor[int(N / 2):] * X_odd])

    def FFT_2D(self, f):
        
        f = np.asarray(f, dtype=float)
        M = f.shape[0]
        N = f[0].shape[0]

        if (np.log2(N) % 1 > 0 or np.log2(M) % 1 > 0):
            print("> The dimensions must be a power of 2, using DFT ...")
            return self.DFT_2D(f)


        result_row = np.asarray([[0.0 for k in range(N)] for l in range(M)], dtype=complex)
        result     = np.asarray([[0.0 for k in range(N)] for l in range(M)], dtype=complex)

        for i in range(M):
            #pass
            #print(f[i])
            x = f[i]
            result_row [i] = self.FFT(x)[0:N]

        result_row = (np.array(result_row))

        for j in range(N):
            y = result_row[:,j]
            result [:,j] = self.FFT(y)[0:M]

        return result
    

    def IDFT(self, y):
        
        y = np.asarray(y, dtype=complex)
        N = y.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(2j * np.pi * k * n / N)
        return (np.dot(M, y)/N)
    
    def IDFT_2D(self, f):
        
        f = np.asarray(f, dtype=complex)
        M = f.shape[0]
        N = f[0].shape[0]
                        
        result = [[0.0 for k in range(M)] for l in range(N)]
        sum_result = 0
        for i in range(M):
            for j in range(N):
                sum_result = 0
                for k in range(M):
                    for l in range(N):
                        sum_result = sum_result + f[k,l]*np.exp(2j * np.pi * ((l*j/N)+(k*i/M)))/(M*N)
                result[i][j] = sum_result
    
        return np.asarray(result)
    
    
    def IFFT(self, y):
        
        y = np.asarray(y, dtype=complex)
        N = y.shape[0]
    
        if N <=2:  # this cutoff should be optimized
            return self.IDFT(y)
        else:
            Y_even = self.IFFT(y[::2])
            Y_odd = self.IFFT(y[1::2])
            factor = np.exp(2j * np.pi * np.arange(N) / N)
            result_a = (Y_even + (factor[:int(N / 2)] * Y_odd))
            result_b = (Y_even + (factor[int(N / 2):] * Y_odd))
            result = np.concatenate([result_a, result_b])
            return result   

    def IFFT_2D(self, f):

        f = np.asarray(f, dtype=complex)
        M = f.shape[0]
        N = f[0].shape[0]

        if (np.log2(N) % 1 > 0 or np.log2(M) % 1 > 0):
            print("> The dimensions must be a power of 2, using DFT ...")
            return self.IDFT_2D(f)
        
        result_row = np.asarray([[0.0 for k in range(N)] for l in range(M)], dtype=complex)
        result     = np.asarray([[0.0 for k in range(N)] for l in range(M)], dtype=complex)
        for i in range(M):
            x = f[i];
            result_row [i] = self.IFFT(x)[0:N]

        result_row = np.array(result_row)

        for j in range(N):
            y = result_row[:,j]
            result [:,j] = self.IFFT(y)[0:M]

        return result

    #Filters
    def low_pass(self, f, cutoff, gauss=0):
        row, col = f.shape
        crow = int(row/2)
        ccol = int(col/2)

        hand_array = np.zeros([row, col])
        #n = np.hstack((np.arange(row).reshape(row, 1), np.array(row*[crow]).reshape(row, 1)))
        #m = np.hstack((np.arange(col).reshape(col, 1), np.array(col*[ccol]).reshape(col, 1)))

        for i in range(0,row):
            for j in range(0,col):
                d = self.distance([i,j], [crow, ccol])
                if (gauss == 0):
                    if (d <= cutoff):
                        hand_array[i][j] = 1;
                    else:
                        hand_array[i][j] = 0;
                else:
                    if(cutoff>0):
                        hand_array[i][j] = np.exp(-1*((d**2)/(2*(cutoff**2))))
                    else:
                        print("> Cutoff can not be 0.")

        return f * hand_array

    def high_pass(self, f, cutoff, gauss=0):
        row, col = f.shape
        crow = int(row/2)
        ccol = int(col/2)

        hand_array = np.zeros([row, col])
        #n = np.hstack((np.arange(row).reshape(row, 1), np.array(row*[crow]).reshape(row, 1)))
        #m = np.hstack((np.arange(col).reshape(col, 1), np.array(col*[ccol]).reshape(col, 1)))

        for i in range(0,row):
            for j in range(0,col):
                d = self.distance([i,j], [crow, ccol])
                if (gauss == 0):
                    if (d <= cutoff):
                        hand_array[i][j] = 0;
                    else:
                        hand_array[i][j] = 1;
                else:
                    if(cutoff>0):
                        hand_array[i][j] = 1 - np.exp(-1*((d**2)/(2*(cutoff**2))))
                    else:
                        print("> Cutoff can not be 0.")

        return f * hand_array


    def band_pass(self, f, cutoff, width, gauss=0):
        row, col = f.shape
        crow = int(row/2)
        ccol = int(col/2)

        hand_array = np.zeros([row, col])
        #n = np.hstack((np.arange(row).reshape(row, 1), np.array(row*[crow]).reshape(row, 1)))
        #m = np.hstack((np.arange(col).reshape(col, 1), np.array(col*[ccol]).reshape(col, 1)))

        for i in range(0,row):
            for j in range(0,col):
                d = self.distance([i,j], [crow, ccol])
                if (gauss == 0):
                    if (((cutoff-(width/2)) <= d) and ((cutoff+(width/2)) >= d)):
                        hand_array[i][j] = 1;
                    else:
                        hand_array[i][j] = 0;
                else:
                    if(d>0):
                        if (width>0):
                            hand_array[i][j] = np.exp(-1*((((d**2)-(cutoff**2))/(d*width))**2))
                        else:
                            print("> Width can not be 0.")

        return f * hand_array

    def band_reject(self, f, cutoff, width, gauss=0):
        row, col = f.shape
        crow = int(row/2)
        ccol = int(col/2)

        hand_array = np.zeros([row, col])
        #n = np.hstack((np.arange(row).reshape(row, 1), np.array(row*[crow]).reshape(row, 1)))
        #m = np.hstack((np.arange(col).reshape(col, 1), np.array(col*[ccol]).reshape(col, 1)))

        for i in range(0,row):
            for j in range(0,col):
                d = self.distance([i,j], [crow, ccol])
                if (gauss == 0):
                    if (((cutoff-(width/2)) <= d) and ((cutoff+(width/2)) >= d)):
                        hand_array[i][j] = 0;
                    else:
                        hand_array[i][j] = 1;
                else:
                    if(d>0):
                        if (width>0):
                            hand_array[i][j] = 1 - np.exp(-1*((((d**2)-(cutoff**2))/(d*width))**2))
                        else:
                            print("> Width can not be 0.")

        return f * hand_array
                

    #Auxiliary methods
    def distance(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)


    def nextPowerOf2(self, n): 
        count = 0; 
    
        if (n and not(n & (n - 1))): 
            return n 
          
        while( n != 0): 
            n >>= 1
            count += 1
          
        return 1 << count; 