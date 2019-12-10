#This is a simple and biased to application binary tree

class binNode:

    data = None
    left = None
    right = None

    #current_way = ''
    #leaves_way = []

    def __init__(self, data = None, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right

    def insert(self, node = None, side = None):
        if (side == 'L' or side == 'l'):
            self.left = node
        elif(side == 'R' or side == 'r'):
            self.right = node
        else:
            print("> ERROR: side must be 'L'/'l' or 'R'/'r'.")

    def printTree(self, depth = 0):
        if (depth != 0):
            print((depth*"   ") + "|" + "->" + str(self.data))
        else:
            print("   " + str(self.data))
        next_deepth = depth + 1

        if (self.left != None):
            self.left.printTree(next_deepth)
        if (self.right != None):
            self.right.printTree(next_deepth)

    def travers(self, current_way = '', leaves_way = []):

        if (self.left != None):
            current_way += '0' 
            leaves_way = self.left.travers(current_way, leaves_way)
            current_way = current_way[0:len(current_way)-1]
        if(self.right != None):
            current_way += '1' 
            leaves_way = self.right.travers(current_way, leaves_way)
            current_way = current_way[0:len(current_way)-1]

        if (self.left == None and self.right == None):
            leaf =  {
                "data": self.data,
                "way": current_way,
            }
            return (leaves_way + [leaf])

        return leaves_way

        #if (current_way == ''):
        #    result = self.leaves_way
        #    self.leaves_way = []
        #    return result


#l1 = binNode(1)
#l2 = binNode(2)
#
#l11 = binNode(11)
#l12 = binNode(12)
#
#l21 = binNode(21)
#l22 = binNode(22)
#
#l221 = binNode(221)
#
#
#l22.insert(l221, 'L')
#
#l1.insert(l11, 'L')
#l1.insert(l12, 'R')
#
#l2.insert(l21, 'L')
#l2.insert(l22, 'R')
#
#root = binNode(0, l1, l2)
#
#root.printTree()
#
#a = root.travers()
#
#print(root.leaves_way)