#Imports
import numpy as np #Numpy

class haar():

    def __init__(self):
        pass


    def DHWT(self, f):
        F = np.asarray(f, dtype=float)
        N = F.shape[0]
        half_size = int(N/2)

        result = np.zeros(N)

        #if (np.log2(N) % 1 > 0):
        #    print("> The dimensions must be a power of 2.")

        for i in range(0, half_size):
            l = i*2

            a = (F[l] + F[l+1])/2
            d = (F[l] - F[l+1])/2
            
            result[i] = a
            result[i+half_size] = d

        return result

    def DHWT_2D(self, f, depth=1):
        result = np.asarray(f, dtype=float)
        N = f.shape[0]
        M = f.shape[1]

        #r_half_size = N #int(N/2)
        #c_half_size = M #int(N/2)

        temp = np.zeros(N)

        #if (np.log2(N) % 1 > 0 or np.log2(M) % 1 > 0):
        #    print("> The dimensions must be a power of 2.")


        for k in range(0,depth):
            
            r_half_size = int(N/(2**k))
            c_half_size = int(M/(2**k))

            for i in range(0, r_half_size):
                if (i == 0):
                    row_result = self.DHWT(result[i,0:c_half_size])
                else:
                    row_result = np.vstack((row_result, self.DHWT(result[i,0:c_half_size])))
    
            result[0:r_half_size, 0:c_half_size] = row_result[:, 0:c_half_size]
    
            for j in range(0, c_half_size):
                if (j == 0):
                    col_result = self.DHWT(result[0:r_half_size,j])
                else:
                    col_result = np.column_stack((col_result, self.DHWT(result[0:r_half_size,j])))

            result[0:r_half_size,0:c_half_size] = col_result[0:r_half_size,:]

        return result


    def IDHWT(self, y):
        Y = np.asarray(y, dtype=float)
        N = Y.shape[0]
        half_size = int(N/2)
    
        result = np.zeros(N)
    
        #if (np.log2(N) % 1 > 0):
        #    print("> The dimensions must be a power of 2.")
    
        for i in range(0, half_size):
            l = i*2
    
            x1 = Y[i] + Y[i+half_size]
            x2 = Y[i] - Y[i+half_size]
            
            result[l] = x1
            result[l+1] = x2
    
        return result

    def IDHWT_2D(self, y, depth=1):
        result = np.asarray(y, dtype=float)
        N = y.shape[0]
        M = y.shape[1]
    
        #r_half_size = N #int(N/2)
        #c_half_size = M #int(N/2)

        temp = np.zeros(N)
    
        #if (np.log2(N) % 1 > 0 or np.log2(M) % 1 > 0):
        #    print("> The dimensions must be a power of 2.")


        for k in range(depth-1, -1, -1):
            
            r_half_size = int(N/(2**k))
            c_half_size = int(M/(2**k))
    
            for j in range(0, c_half_size):
                if (j == 0):
                    col_result = self.IDHWT(result[0:r_half_size,j])
                else:
                    col_result = np.column_stack((col_result, self.IDHWT(result[0:r_half_size,j])))
    
            result[0:r_half_size,0:c_half_size] = col_result[0:r_half_size,:]
    
            for i in range(0, r_half_size):
                if (i == 0):
                    row_result = self.IDHWT(result[i,0:c_half_size])
                else:
                    row_result = np.vstack((row_result, self.IDHWT(result[i,0:c_half_size])))
    
            result[0:r_half_size, 0:c_half_size] = row_result[:, 0:c_half_size]
    
        return result




#array = [[100, 50, 60, 150],[20, 60, 40, 30],[50, 90, 70, 82],[74, 66, 90, 58]]
#wh = haar()
#
#print(wh.DWHT_2D(array))
