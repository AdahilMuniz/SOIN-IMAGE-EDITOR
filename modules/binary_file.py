#Imports
import numpy   as np #Numpy

class binary_file:
    
    def __init__(self):
        pass

    def write_file(self, file_path, bit_string):
        f = open(file_path,"wb")
        to_write = self.to_byte(bit_string)
        f.write(bytearray(to_write))
        f.close()

    def read_file(self, file_path):
        f = open(file_path,"rb")
        byte_array = f.read()
        return self.to_bit(byte_array)

    def to_byte(self, bit_string):
        #size = self.nextEightMultiple(len(bit_string))
        size_bytes = int(np.ceil(len(bit_string)/8))
        size_bits  = self.nextEightMultiple(len(bit_string))
        #to_complement = 8 - (len(bit_string)%8)
        byte_vector = [0]*size_bytes

        #Complete with 0s        
        for i in range(0, (size_bits-len(bit_string))):
            bit_string += '0'

        for i in range(0, size_bits):
            byte_vector[int(i/8)] = (byte_vector[int(i/8)]<<1) | int(bit_string[i])
        
        return byte_vector

    def to_bit(self, byte_array):
        bit_string = ''
        for byte_val in byte_array:
            bit_string += self.get_bin_byte(byte_val)
        
        return bit_string


    def get_bin_byte(self, data):
        bit_string = ''

        #if (data > 255):
        #    print("> Data must be one byte")
        #    return -1

        for i in range(7,-1, -1):
            bit_string += str((data  >> i) & 0x01)

        return bit_string


    def nextEightMultiple(self, value):
        return ((value + 7) & (-8))


#arr = [2,3,5,8,0,1,3,2,3,2,2,2,3,2,10,10,10,10,10]
#bf = binary_file()
#orig = '0000011100100000011001000001011000010000100000000100000001010000101010000001011000100010101100111001100111111001110101010101'
#bf.write_file("test.simg", orig)
#bf.read_file("test.simg")