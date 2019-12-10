#TODO: Do I need a identifier for the dict elements?

#Imports
import numpy   as np #Numpy
import time

from joblib import Parallel, delayed

from binary_tree import binNode
from binary_file import binary_file

class huffman():

    bin_file = binary_file()

    def __init__(self):
        pass

    def encode(self, array):
        result         = ""
        hist_result    = self.histogram(array)
        dicts          = self.build_dicts(hist_result)
        non_zero_dicts = self.non_zero(dicts)
        tree           = self.build_tree(non_zero_dicts)
        root           = tree[0]
        n_nodes        = tree[1]
        n_leaves       = tree[2]

        l_ways = root.travers()


        cur = time.time()
        #for i in array:
        #    for j in l_ways:
        #        if (i == j["data"]["value"]):
        #            result += j["way"]

        num_cores = 4#multiprocessing.cpu_count()
        tmp_result = Parallel(n_jobs=num_cores)(delayed(self.find_symbol)(i, l_ways) for i in array)
#
        for i in tmp_result:
            result += i
#
        print("Time : ", time.time() - cur)

        return (result, root, n_leaves)


    def find_symbol(self, value, l_ways):
        result = ''
        for j in l_ways:
            if (value == j["data"]["value"]):
                result += j["way"]
        return result

    #OBS.: On decode tree the node data must already the symbol
    def decode(self, array, root_tree):
        count = 0
        start = 0
        end = 1

        result = []

        l_way = root_tree.travers()
        while(end < len(array)):
            way = array[start:end]
            for i in l_way:
                if (way == i["way"]):
                    start = end
                    result.append(i["data"])
            end += 1

        return result

    def histogram(self, array):
        hist        = np.zeros(np.max(array)+1)
        for i in array:
            hist[i] = hist[i] + 1
        return hist

    def non_zero(self, dicts):
        non_zero_dict = [];
        for i in dicts:
            if (i["freq"]>0):
                non_zero_dict.append(i)
        return non_zero_dict

    #def sort_dicts(self, dicts):
    #    x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    #    sorted_dicts = sorted(dict.items(), key=lambda kv: kv[1])

    def min_dict(self, dicts):
        return min(dicts, key=lambda x:x['freq'])

    def max_dict(self, dicts):
        return max(dicts, key=lambda x:x['freq'])

    def build_dicts(self, hist_array):
        dicts = []
        for i in range(0, len(hist_array)):
            element = {
                "value" : i,
                "freq"  : hist_array[i],
            }
            dicts.append(element)
        return dicts

    #TODO: Do a better analyze about the tree correctness
    def build_tree(self, dicts):
        node_list = [] #List of tree nodes
        root  = None #Root node
        nodes_number = len(dicts)
        leaves_number = len(dicts)

        #Create leaf nodes
        for i in dicts:
            node_list.append(binNode(i))

        while (len(dicts) > 1):
            #Get the two least probable
            low_freq_1 = self.min_dict(dicts)#First one
            dicts.remove(low_freq_1)# Remove from list
            low_freq_2 = self.min_dict(dicts)#Second one
            dicts.remove(low_freq_2)# Remove from list

            #Internal Freq
            sum_freq = low_freq_1["freq"] + low_freq_2["freq"]
            #Create element dict to internal
            element = {
                "value" : None,
                "freq"  : sum_freq,
            }
            #Append to the dict list
            dicts.append(element)
    
            #Search the child nodes in the node list
            node_l = None
            node_r = None
            for i in node_list:
                if (low_freq_1["value"] == i.data["value"] and low_freq_1["freq"] == i.data["freq"]):
                    node_l = i
            node_list.remove(node_l)
            for i in node_list:
                if (low_freq_2["value"] == i.data["value"] and low_freq_2["freq"] == i.data["freq"]):
                    node_r = i
            node_list.remove(node_r)
    
            #Create new nodes if they were not already created
            #if (node_l == None):
            #    node_l = binNode(low_frq_1)
            #if (node_r == None):
            #    node_r = binNode(low_frq_2)
    
            #Create internal node
            root = binNode(element, node_l, node_r)
            node_list.append(root)
            nodes_number +=1

        return (root, nodes_number, leaves_number)

    #Methods to prepare the data to be saved on files
    def huff2bit(self, enc_word, root, n_leaves):
        tree_enc = self.tree2bit(root, n_leaves)
        return tree_enc + enc_word

    def tree2bit(self, root, n_leaves):
        #tree_encode_size = 8*n_leaves + n_nodes
        #tree_array = np.zeros(1+tree_encode_size) #Number of leaves + Tree Encoded Size
        #tree_array[0] = n_leaves;
        n_leaves_low  = n_leaves
        n_leaves_high = n_leaves >> 8
        bit_n_leaves_l   = self.bin_file.get_bin_byte(n_leaves_low)
        bit_n_leaves_h   = self.bin_file.get_bin_byte(n_leaves_high)
        tree_serialized = self.nodes_serialize(root)
        return bit_n_leaves_h + bit_n_leaves_l + tree_serialized


    # Leaves    -> 1 + Bytes
    # Internals -> 0 
    def nodes_serialize(self, node, bit_string = ''):

        if (node.left == None and node.right == None):
            bit_string += '1'
            bit_string += self.bin_file.get_bin_byte(node.data["value"])
        else:
            bit_string += '0'
            bit_string = self.nodes_serialize(node.left , bit_string)
            bit_string = self.nodes_serialize(node.right, bit_string)
        return bit_string


    #Methods to decode the string from file
    def bit2huff(self, bit_string):
        n_leaves = int(bit_string[0:16], 2)
        n_nodes = 2*n_leaves - 1
        tree_string_size = n_leaves*8 + n_nodes
        garbage, root = self.nodes_deserialize(bit_string[16:tree_string_size+16])
        array = bit_string[tree_string_size+16:len(bit_string)]

        return (array, root)


    def nodes_deserialize(self, bit_string = ''):
        if (bit_string[0] == '1'):
            byte_value = int(bit_string[1:9], 2)
            bit_string = bit_string[9:len(bit_string)]
            return (bit_string, binNode(byte_value, None, None))
        else:
            left  = None
            right = None
            bit_string = bit_string[1:len(bit_string)]
            bit_string, left  = self.nodes_deserialize(bit_string)
            bit_string, right = self.nodes_deserialize(bit_string)
            return (bit_string, binNode(None, left, right))



#arr = [2,3,5,8,0,1,3,2,3,2]
#arr = [2,3,5,8,0,1,3,2,3,2,2,2,3,2,10,10,10,10,10]
#hff = huffman()
#
#
#enc = hff.encode(arr)
#complete_word_enc = hff.huff2bit(enc[0], enc[1], enc[2])
#enc_array, root = hff.bit2huff(complete_word_enc)
#dec = hff.decode(enc_array, root)
#
#print("Complete Encoded Word: ", complete_word_enc)
#print("Complete Decoded Array: ", dec)
#
#
#print("Complete Encoded Word Size (bits): ", len(complete_word_enc))
#print("Complete Decoded Array Size (bits): ", len(dec)*8)