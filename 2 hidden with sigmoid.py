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

import numpy as np
import matplotlib
from matplotlib import pyplot
class NeuralNetwork:
    def cost(self, y,out):
        err = y - out
        err = err*err
        dip = np.matrix(np.ones(err.shape[0])) * np.matrix(err)
        dip = np.array(dip)
        dip = dip[0]
        cost= np.dot(dip,dip)
        return cost

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function, its sure after penurunan matematik
        return x * (1 - x)

    def __init__(self, x, y):
        self.h2         = np.array(np.zeros(6)).T
        self.h1         = np.array([np.zeros(6)]).T                  #hidden layer 1, ada 6 neuron
        self.input      = x                                                               #horizontal m, in vertical sample            
        self.y          = y                                                               #horizontal n, in vertical sample
        
        self.weights3   = np.random.random( (self.y.shape[1], self.h2.shape[0]) )      #horizontal per input m, verticaled as y n
        self.bias3      = np.array([np.random.random(self.y.shape[1])]).T

        self.weights2   = np.random.random( (self.h2.shape[0], self.h1.shape[0]) )      #horizontal per input m, verticaled as y n
        self.bias2      = np.array([np.random.random(self.h2.shape[0])]).T

        self.bias1      = np.array([np.random.random(self.h1.shape[0])]).T
        self.weights1   = np.random.random( (self.h1.shape[0], self.input.shape[1]) )        #horizontal aligned
        
        # eta adalah learning rate
        self.eta = 0.5
        self.output     = np.zeros(y.shape)
        self.error      = self.y - self.output
        self.wat        = self.sigmoid_derivative(self.output)

    def randominit(self):
        self.weights3   = np.random.random( (self.y.shape[1], self.h2.shape[0]) )      #horizontal per input m, verticaled as y n
        self.bias3      = np.array([np.random.random(self.y.shape[1])]).T

        self.weights2   = np.random.random( (self.h2.shape[0], self.h1.shape[0]) )      #horizontal per input m, verticaled as y n
        self.bias2      = np.array([np.random.random(self.h2.shape[0])]).T

        self.bias1      = np.array([np.random.random(self.h1.shape[0])]).T
        self.weights1   = np.random.random( (self.h1.shape[0], self.input.shape[1]) )
    def feedind(self,x):
        #jadi fungsi ini ngefeedfwd semua sampel... padahal ga perlu
        #nah makanya ini akan urg bikin indivnya perindex sample x
        sumh1 = np.matrix(self.weights1) * np.matrix(self.input[x]).T
        outh1 = sumh1 + self.bias1
        outh1 = np.array(outh1)
        self.h1 = self.sigmoid(outh1)

        sumh2 = np.matrix(self.weights2) * np.matrix(self.h1)
        outh2 = sumh2 + self.bias2
        outh2 = np.array(outh2)
        self.h2 = self.sigmoid(outh2)

        sumir = np.matrix(self.weights3) * np.matrix(self.h2)
        out = sumir + self.bias3
        out = np.array(out).T

        self.output[x] = self.sigmoid(out)              #horizontal
        self.output[x] = np.array(self.output[x])
    
    def test(self,x):
        #jadi fungsi ini ngefeedfwd masukan (bukan sampel)
        sumh1 = np.matrix(self.weights1) * np.matrix(x).T
        outh1 = sumh1 + self.bias1
        outh1 = np.array(outh1)
        self.h1 = self.sigmoid(outh1)

        sumh2 = np.matrix(self.weights2) * np.matrix(self.h1)
        outh2 = sumh2 + self.bias2
        outh2 = np.array(outh2)
        self.h2 = self.sigmoid(outh2)

        sumir = np.matrix(self.weights3) * np.matrix(self.h2)
        out = sumir + self.bias3
        out = np.array(out).T

        out = self.sigmoid(out)
        #
        """
        for i in range(0,out.shape[1]):
            if out[0][i] <0.15:
                out[0][i] = 0
            elif out[0][i] >0.9:
                out[0][i] = 1
            else:
                out[0][i] = -1"""
        return out

    
    def backprop(self,x):
        ### stochastic ####/
        # x berupa index random
        """
        kali = 2 * (self.y[x] - self.output[x]) * self.sigmoid_derivative(self.output[x])
        #harusnya di(-), tapi ini udah y-output bukan output-y lagi; jadi aman
        kali = np.matrix(kali).T

        d_bias3 = kali
        d_weights3 = kali * np.matrix(self.h2).T    #jadi gepeng
        d_bias3 = np.array(d_bias3)
        d_weights3 = np.array(d_weights3)

        d_bias2 = self.sigmoid_derivative(self.h2) * np.array( np.matrix(self.weights3).T * kali ) 
        d_weights2 = np.matrix(self.sigmoid_derivative(self.h2) * np.array( np.matrix(self.weights3).T * kali ) ) * kali * np.matrix(self.h1).T      #input sudah gepeng
        d_bias2 = np.array(d_bias2)
        d_weights2 = np.array(d_weights2)

        d_bias1 = np.matrix(self.sigmoid_derivative(self.h1) * self.weights2.T ) * np.matrix(self.sigmoid_derivative(self.h2) * self.weights3.T) * kali
        d_weights1 = np.matrix( self.sigmoid_derivative(self.h1) * self.weights2.T ) * np.matrix(self.sigmoid_derivative(self.h2) * self.weights3.T) * kali * np.matrix(self.input[x])
        d_bias1 = np.array(d_bias1)
        d_weights1 = np.array(d_weights1)
        

        ### mini batch ###
        # x berupa array 1 dimensi index random
        """
        d_weights1 = 0
        d_weights2 = 0
        d_weights3 = 0
        d_bias1    = 0
        d_bias2    = 0
        d_bias3    = 0
        m = self.input.shape[0]
        for e in range(0,m):
            self.feedind(e)                 #ulah hilap, semua output dan h mesti diambil per sampel!
            kali =  (self.y[e] - self.output[e]) * self.sigmoid_derivative(self.output[e])
            #harusnya di(-), tapi ini udah y-output bukan output-y lagi; jadi aman
            kali = np.matrix(kali).T

            ds_bias3 = kali
            ds_weights3 = kali * np.matrix(self.h2).T    #jadi gepeng
            ds_bias3 = np.array(ds_bias3)
            ds_weights3 = np.array(ds_weights3)

            ds_bias2 = self.sigmoid_derivative(self.h2) * np.array( np.matrix(self.weights3).T * ds_bias3 ) 
            ds_weights2 = np.matrix(ds_bias2) * np.matrix(self.h1).T      #jadi gepeng
            ds_bias2 = np.array(ds_bias2)
            ds_weights2 = np.array(ds_weights2)

            ds_bias1 =  self.sigmoid_derivative(self.h1) * np.array( np.matrix(self.weights2).T * ds_bias2 )                #input sudah gepeng
            ds_weights1 =  np.matrix(ds_bias1) * np.matrix(self.input[e])
            ds_bias1 = np.array(ds_bias1)
            ds_weights1 = np.array(ds_weights1)

            d_weights1 += ds_weights1
            d_bias1    += ds_bias1
            d_weights2 += ds_weights2
            d_bias2    += ds_bias2
            d_weights3 += ds_weights3
            d_bias3    += ds_bias3            
        
        d_weights1 = d_weights1/m
        d_bias1    = d_bias1/m
        d_weights2 = d_weights2/m
        d_bias2    = d_bias2/m
        d_weights3 = d_weights3/m
        d_bias3    = d_bias3/m
        #print(i)"""

        # update the weights with the derivative (slope) of the loss function
        
        self.weights1 += self.eta*d_weights1
        self.bias1    += self.eta*d_bias1

        self.weights2 += self.eta*d_weights2
        self.bias2    += self.eta*d_bias2

        self.weights3 += self.eta*d_weights3
        self.bias3    += self.eta*d_bias3
        self.feedind(x)
        #self.weights2 += d_weights2

if __name__ == "__main__":
    x = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    y = np.array([[0],[1],[1],[0]])
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
    binar.eta = 10
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
