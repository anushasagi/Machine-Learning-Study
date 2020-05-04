#The commented variables are suggestions so change them as appropriate,
#However, do not change the _init_(), train(), or predict(x=[]) function headers
#You may create additional functions as you see fit

import numpy as np
np.random.seed(100)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    p = sigmoid(p)
    return p * (1 - p)

class NeuralNetwork:

    #Do not change this function header
    def __init__(self,x=[[]],y=[],numLayers=2,numNodes=2,eta=0.001,maxIter=10000):
        self.data = x
        self.labels = y
        self.nLayers = numLayers
        self.nNodes = numNodes
        self.eta = eta
        self.maxIt = maxIter

        #self.weights = [np.random.rand(len(x[i]),numNodes)] #create the weights from the inputs to the first layer
        #for each of the layers
            #self.weights.append(np.random.rand(numNodes,numNodes)) #create the random weights between internal layers
        #self.weights.append(np.random.rand(numNodes,1)) #create weights from final layer to output node
        #self.outputs = np.zeros(y.shape)
        #self.train()
        #self.a = 1 #this is how you define a non-static variable
        

        #Weights initialization
        self.weights = [np.random.rand(len(x[0]),self.nNodes)] #create the weights from the inputs to the first layer
        for i in range(self.nLayers-1):
            self.weights.append(np.random.rand(self.nNodes,self.nNodes))
        self.weights.append(np.random.rand(self.nNodes,1))


        #self.outputs = np.zeros(y.shape)
        #self.train()
        #self.predict()


        #Training the data
        for d_i, l_i in zip(self.data, self.labels):
            self.data_i = d_i
            self.label_i = l_i
            self.train()

    def train(self):
        #Do not change this function header
            self.data_i = np.array(self.data_i).reshape(len(self.data[0]), 1)
            a2, cache = self.feedforward(self.data_i)
            predicted = cache[len(cache)-1]
            error = predicted-self.label_i
            delta_i = self.backprop(a2, error)
            for i in range(len(self.weights)):
                wt = self.weights[i]
                change = self.eta * np.dot(self.data_i, (delta_i[i].transpose()))
                self.weights[i] = wt-change
                self.data_i = cache.pop(0)

    def predict(self, x=[]):
        #Do not change this function header
        x = np.array(x)
        return self.feedforward(x)[1][-1]

    def feedforward(self, x):
        #This function is likely to be very helpful, but is not necessary
        point = x
        activation = []
        cache = []
        for weight in self.weights:
            mm = np.dot(weight.transpose(), point)
            activation.append(mm)
            cache.append(sigmoid(mm))
            point = sigmoid(mm)
        return activation, cache

    def backprop(self, a2, error):
        #This function is likely to be very helpful, but is not necessary
        current_delta = []
        for i in reversed(range(len(self.weights))):
            i = i+1
            if i < len(self.weights):
                wtl = self.weights[i]
            else:
                wtl = np.array([[1]])
            err_lw = np.dot(wtl, error)
            delta = err_lw * sigmoid_derivative(a2[i-1])
            current_delta.insert(0, delta)
            error = delta
        return current_delta


if __name__ == "__main__":
    x = [[1,0],[0,1],[1,1],[0,0]]
    y = [1,1,0,0]
    nn = NeuralNetwork(x, y)
    print(x)
    print(y)
    for i in x:
        predicted_value = nn.predict(i)
        print(predicted_value)