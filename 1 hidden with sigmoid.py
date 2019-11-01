#Alhamdulillah! ANN diajar XOR dengan no-hidden layer SGD dan no bias sudah berjalan!
#Tetapi cost tidak akan maks, alias hasil akan 0 melulu; kenapa?
#karena tidak ada hidden layer dan bias, kasus X(0,0) bakal anulir semua weight
#ketika di sigmoidkan, istiqomah 0.5!

#Plus bias, ternyata tidak bisa diselesaikan! Berarti mesti ada hidden layer
#alhamdulillah sudah bisa multi input multi output,, tapi kita akan melihat cost function yang
#masih diatas satu lantaran (1,1,1) -> (0,0,0) tidak bisa diselesaikan dengan struktur yang
#masih amat sederhana ini

#HIdden layer 1: Ada masalah, kita sering nyangkut di minimum lokal
#Taktik: kita bikin superepoch buat cari minimum yang seminimum-minimumnya (superepoch == step)
#alhamdulillah ini udah dimasukin yang Cost:  4.838387113974794e-05; paling kecil ditemukan tadi di 2.sekiane-05
#sudah pintar bray, sudah diajar lulus!

import struct
import numpy as np
import matplotlib
from matplotlib import pyplot

def get_data_and_labels(images_filename, labels_filename):
    print("Opening files ...")
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        print("Reading files ...")
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
        labels = []
        for item in range(0,10): #num_of_items
            print("Current image number: %7d" % item)
            for value in range(num_of_image_values):
                data[item][value] = int.from_bytes(images_file.read(1),
                                                   byteorder="big")
            labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
        return data, labels
    finally:
        images_file.close()
        labels_file.close()
        print("Files closed.")


class NeuralNetwork:
    def cost(self, y,out):
        cost = 0
        for i in range(0, self.y.shape[0]):
            errind = self.y[i][0] - self.output[i] 
            errind = np.array(errind)
            costind = np.dot(errind,errind)
            cost = cost + costind
        
        cost = cost / self.y.shape[0]
        return cost
    

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function, its sure after penurunan matematik
        return x * (1 - x)

    def __init__(self, x, y):
        self.h1          = np.array([np.zeros(6)]).T                  #hidden layer 1, ada 6 neuron
        self.input      = x                                                               #horizontal m, in vertical sample            
        self.y          = y                                                               #horizontal n, in vertical sample
        
		# 2|6|1
        self.weights2   = np.random.random( (self.y.shape[1], self.h1.shape[0]) )      #horizontal per input m, verticaled as y n
        #self.weights2   = np.array([[0.80614695,-0.14111434,2.46709216,2.32656736,1.5757561,2.4043752,]])
        self.bias2      = np.array([np.random.random(self.y.shape[1])]).T
        #self.bias2      = np.array([[-6.23008396]])

        self.bias1      = np.array([np.random.random(self.h1.shape[0])]).T
        #self.bias1      = np.array([[-8.05478534],[-6.96230537],[-5.86651519],[-6.98042039],[-7.56557186],[-7.91246433]])
        self.weights1   = np.random.random( (self.h1.shape[0], self.input.shape[1]) )        #horizontal aligned
        #self.weights1   = np.array([[8.5073284,7.86768107,34.15892027],[8.54998366,8.11889863,26.61332107],[4.64966991,4.74634817,31.58175057],[6.10580688,5.89890652,35.31839077],[7.550106,6.91320036,37.25279925],[7.05964822,7.7838479,42.83578178]])
        
        # eta adalah learning rate
        self.eta = 1
        self.output     = np.zeros(y.shape)
        self.error      = self.y - self.output
        self.wat        = self.sigmoid_derivative(self.output)

    def randominit(self):
        self.weights2   = np.random.random( (self.y.shape[1], self.h1.shape[0]) )      #horizontal per input m, verticaled as y n
        self.bias2      = np.array([np.random.random(self.y.shape[1])]).T

        self.bias1      = np.array([np.random.random(self.h1.shape[0])]).T
        self.weights1   = np.random.random( (self.h1.shape[0], self.input.shape[1]) )
    def feedind(self,x):
        #jadi fungsi ini ngefeedfwd semua sampel... padahal ga perlu
        #nah makanya ini akan urg bikin indivnya perindex sample x
        sumh1 = np.matrix(self.weights1) * np.matrix(self.input[x]).T
        outh1 = sumh1 + self.bias1
        outh1 = np.array(outh1)
        self.h1 = self.sigmoid(outh1)

        sumir = np.matrix(self.weights2) * np.matrix(self.h1)
        out   = sumir + self.bias2
        out   = np.array(out).T
        self.output[x] = self.sigmoid(out)              #horizontal
        self.output[x] = np.array(self.output[x])
    
    def test(self,x):
        #jadi fungsi ini ngefeedfwd masukan (bukan sampel)
        sumh1 = np.matrix(self.weights1) * np.matrix(x).T
        outh1 = sumh1 + self.bias1
        outh1 = np.array(outh1)
        h1 = self.sigmoid(outh1)

        sumir = np.matrix(self.weights2) * np.matrix(h1)
        out   = sumir + self.bias2
        out   = np.array(out).T
        out   = self.sigmoid(out)

        return out
    
    def backprop(self,x):
        ### stochastic ####/
        # x berupa index random
        """
        kali = 2 * (self.y[x] - self.output[x]) * self.sigmoid_derivative(self.output[x])
        kali = np.matrix(kali).T

        d_bias2 = kali
        d_weights2 = kali * np.matrix(self.h1).T    #jadi gepeng
        d_bias2 = np.array(d_bias2)
        d_weights2 = np.array(d_weights2)

        d_bias1 = self.sigmoid_derivative(self.h1) * np.array( self.weights2.T * kali ) 
        d_weights1 = np.matrix(self.sigmoid_derivative(self.h1) * self.weights2.T) * kali * np.matrix(self.input[x])      #input sudah gepeng
        #d_weights1 = np.array(np.matrix(self.weights2).T * kali) * np.array( self.sigmoid_derivative(self.h1) * np.matrix(self.input[x]))      #input sudah gepeng
        d_bias1 = np.array(d_bias1)
        d_weights1 = np.array(d_weights1)
            #harusnya di(-), tapi ini udah y-output bukan output-y lagi; jadi aman
		"""
        ### mini batch ###
        # x diabaykan
        d_weights1 = 0
        d_weights2 = 0
        d_bias1    = 0
        d_bias2    = 0
        m = self.y.shape[0]
        for e in range(0,m):
            self.feedind(e)                 #ulah hilap, semua output dan h mesti diambil per sampel!
            kali = 2 * (self.y[e] - self.output[e]) * self.sigmoid_derivative(self.output[e])
            kali = np.matrix(kali).T

            ds_bias2 = kali
            ds_weights2 = kali * np.matrix(self.h1).T    #jadi gepeng
            ds_bias2 = np.array(ds_bias2)
            ds_weights2 = np.array(ds_weights2)

            ds_bias1 = self.sigmoid_derivative(self.h1) * np.array( self.weights2.T * kali ) 
            ds_weights1 = np.matrix(self.sigmoid_derivative(self.h1) * self.weights2.T) * kali * np.matrix(self.input[e])      #input sudah gepeng
            ds_bias1 = np.array(ds_bias1)
            ds_weights1 = np.array(ds_weights1)

            d_weights1 += ds_weights1
            d_bias1    += ds_bias1
            d_weights2 += ds_weights2
            d_bias2    += ds_bias2
        
        d_weights1 = d_weights1/m
        d_bias1    = d_bias1/m
        d_weights2 = d_weights2/m
        d_bias2    = d_bias2/m
        #print(i)"""

        # update the weights with the derivative (slope) of the loss function
        
        self.weights1 += self.eta*d_weights1
        self.bias1    += self.eta*d_bias1

        self.weights2 += self.eta*d_weights2
        self.bias2    += self.eta*d_bias2

        self.feedind(x)
        #self.weights2 += d_weights2

if __name__ == "__main__":
    x = np.array([[0,0,0,0],
                  [0,1,0,0],
                  [1,0,0,0],
                  [1,1,0,0]])
    y = np.array([[0,0],[1,1],[1,1],[0,0]])

    binar = NeuralNetwork(x,y)

    for i in range(0, y.shape[0]):
        binar.feedind(i)
        yt = binar.output[i]
        print(yt)
    cost = binar.cost(y,binar.output)
    print("Cost: ",cost)

    #sudah ditraining ya om, otw minimum global brow!, Cost:  4.838387113974794e-05
    epoch = 1000
    nepoch= np.array([0])
    ncost = np.array([cost])
    binar.eta = 5
    for i in range(0,epoch):
        v = np.random.randint(0,4)
        binar.backprop(v)
        for j in range(0, y.shape[0]):
            binar.feedind(j)
        cost = binar.cost(y,binar.output)
        print("Cost: ",cost)
        ncost = np.append(ncost, cost)
        nepoch= np.append(nepoch,i+1)
    
    pyplot.plot(nepoch,ncost)
    pyplot.show()    
    
    for i in range(0, y.shape[0]):
        binar.feedind(i)
        yt = binar.test(x[i])
        print(yt)
    
    print()

    yt = binar.test(np.array([0,1]))
    print(yt)