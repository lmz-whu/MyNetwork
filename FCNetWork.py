
# coding: utf-8

import numpy as np

class Network():
    def __init__(self, size, alpha = 0.01, activition = 'sigmoid', cost = 'crossEntropy'):
        self.num_layers = len(size) - 1
        #there are three ways to initialize the w
        # 1. for relu : scala = np.np.sqrt(2/j) 
        # 2. for tanh : scala = np.sqrt(1/j)
        # 3. for others : scala = np.sqrt( 2/(i+j) )
        self.w = [0] + [np.random.rand(i,j)*np.sqrt(2/j) for i,j in zip(size[1:], size[:-1])]
        self.b = [0] + [np.random.rand(i,1) for i in size[1:]]
        
        #define the dict of activition function, according to the parameter to choose the actFunc
        activition_functions = {'sigmoid':self.sigmoid, 'tanh':self.tanh, 'relu':self.ReLU}
        activition_functions_derivative = {'sigmoid':self.sigmoid_derivative, 'tanh':self.tanh_derivative, 'relu':self.ReLU_devirative}
        if activition in activition_functions:
            self.actFunc = activition_functions[activition]
            self.actFunc_derivative = activition_functions_derivative[activition]
        #define the dict of cost function, according to the parameter to choose the costFunc
        cost_functions = {'crossEntropy':self.costFunction_crossEntropy}
        cost_functions_derivative = {'crossEntropy':self.costFunction_crossEntropy_derivative}
        if cost in cost_functions:
            self.costFunc = cost_functions[cost]
            self.costFunc_derivative = cost_functions_derivative[cost]
    
    def train(self, trainDatas, nx, loop = 10, alpha = 0.01, testDatas = None, batchSize = 10):
        '''
        trainDatas : the dataset for training the network, it's shape is (m, nx+ny)
        nx : the dims of features, m denotes the number of all examples, ny : the dims of output
        loop : the number of iteration
        alpha : learning rate
        testDatas : the datasets for test/evaluate the network
        batchSize : 
        '''
        m_examples = trainDatas.shape[0]
        for _ in range(loop):
            np.random.shuffle(trainDatas)
            minibatches = [trainDatas[k:k+batchSize] for k in range(0, m_examples, batchSize)]
            
            for minibatch in minibatches:
                x = minibatch[:, :nx].T  #dims of x = (nx, batchSize)
                y = minibatch[:, nx:].T  #dims of y = (ny, batchSize)
                #calculating the dw db by backpropagation
                dw, db = self.backprop_L2_regularization(x, y)  #---------------------------------------------#
                #update the w,b
                self.w = [w - alpha * delta_w for w,delta_w in zip(self.w, dw)]
                self.b = [b - alpha * delta_b for b,delta_b in zip(self.b, db)]
            
            if (_) % 200 == 0 or True:
                #calculate the training error / cost
                X = trainDatas[:, :nx].T
                y = trainDatas[:, nx:].T
                a = self.predict(X)
                train_accuracy = np.mean(np.argmax(a, axis=0) == np.argmax(y, axis=0))
                cost = self.costFunc(a, y)
                cost = np.mean(np.sum(cost, axis = 0))
                if testDatas is not None:
                    #calculate the test error
                    X_test = testDatas[:, :-10].T
                    y_test = testDatas[:, -10:].T
                    test = self.predict(X_test)
                    test_accuracy = np.mean(np.argmax(test, axis=0) == np.argmax(y_test, axis=0))
                    print(_, "cost: ",cost, "train accurecy: ", train_accuacy, "test accurecy: ", test_accuracy)
                else:
                    print(_, "cost: ", cost)
    
    
    
    
    def train_Adam(self, trainDatas, nx, loop=10, alpha=0.01, testDatas=None, batchSize=10,                    lambd=1, belta_1=0.9, belta_2=0.999, epsilon=1e-8):
        '''
        trainDatas : the dataset for training the network, it's shape is (m, nx+ny)
        nx : the dims of features, m denotes the number of all examples, ny : the dims of output
        loop : the number of iteration
        alpha : learning rate
        testDatas : the datasets for test/evaluate the network
        batchSize : 
        '''
        m_examples = trainDatas.shape[0]
        t = 0
        Vdw = [np.zeros_like(i) for i in self.w]  # momentum
        Vdb = [np.zeros_like(i) for i in self.b]
        Sdw = [np.zeros_like(i) for i in self.w]  # Rmsprop
        Sdb = [np.zeros_like(i) for i in self.b]
        for _ in range(loop):
            np.random.shuffle(trainDatas)
            minibatches = [trainDatas[k:k+batchSize] for k in range(0, m_examples, batchSize)]
            alpha = alpha * 0.95**_
            
            for minibatch in minibatches:
                t += 1
                x = minibatch[:, :nx].T  #dims of x = (nx, batchSize)
                y = minibatch[:, nx:].T  #dims of y = (ny, batchSize)
                #calculating the dw db by backpropagation
                #forward propagation
                Z = [0]
                A = [x]
                m = x.shape[1]  # in fact, m = batchSize
                for i in range(1, self.num_layers + 1):  # i = {1, 2, ..., L}, L = self.num_layers
                    z_i = np.dot(self.w[i], x) + self.b[i]  #dims = (ni, m)
                    Z.append(z_i)
                    if i == self.num_layers:    #activation function of last layer is sigmoid
                        x = self.sigmoid(z_i)
                    else:
                        x = self.actFunc(z_i)    #dims = (ni, m)
                    A.append(x)
                
                dw = [np.zeros_like(i) for i in self.w]
                db = [np.zeros_like(i) for i in self.b]
                Vdw_corrected = [np.zeros_like(i) for i in self.w]  # momentum
                Vdb_corrected = [np.zeros_like(i) for i in self.b]
                Sdw_corrected = [np.zeros_like(i) for i in self.w]  # Rmsprop
                Sdb_corrected = [np.zeros_like(i) for i in self.b]
#                 dz = self.costFunc_derivative(A[-1], y) * self.actFunc_derivative(Z[-1]) # the last layer's dz
                dz = A[-1] - y   #this means that the last activition function is sigmoid
                dw[-1] = np.dot(dz, A[-2].T) / m  + lambd*self.w[-1] / m
                db[-1] = np.mean(dz, axis = 1, keepdims=True)
                Vdw[-1] = belta_1 * Vdw[-1] + (1-belta_1) * dw[-1]
                Vdb[-1] = belta_1 * Vdb[-1] + (1-belta_1) * db[-1]
                Sdw[-1] = belta_1 * Sdw[-1] + (1-belta_1) * dw[-1]**2
                Sdb[-1] = belta_1 * Sdb[-1] + (1-belta_1) * db[-1]**2
                Vdw_corrected[-1] = Vdw[-1] / (1-belta_1**t)
                Vdb_corrected[-1] = Vdb[-1] / (1-belta_1**t)
                Sdw_corrected[-1] = Sdw[-1] / (1-belta_2**t)
                Sdb_corrected[-1] = Sdb[-1] / (1-belta_2**t)
                for i in range(self.num_layers-1, 0, -1):
                    dz = self.actFunc_derivative(Z[i]) * np.dot(self.w[i+1].T, dz)   #Given (i+1)th dz to calculate ith dz
                    dw[i] = np.dot(dz, A[i-1].T) / m  + lambd*self.w[i] / m
                    db[i] = np.mean(dz, axis = 1, keepdims=True)
                    Vdw[i] = belta_1 * Vdw[i] + (1-belta_1) * dw[i]
                    Vdb[i] = belta_1 * Vdb[i] + (1-belta_1) * db[i]
                    Sdw[i] = belta_1 * Sdw[i] + (1-belta_1) * dw[i]**2
                    Sdb[i] = belta_1 * Sdb[i] + (1-belta_1) * db[i]**2
                    Vdw_corrected[i] = Vdw[i] / (1-belta_1**t)
                    Vdb_corrected[i] = Vdb[i] / (1-belta_1**t)
                    Sdw_corrected[i] = Sdw[i] / (1-belta_2**t)
                    Sdb_corrected[i] = Sdb[i] / (1-belta_2**t)
                    
                #update the w,b
                self.w = [w - alpha * vw/(np.sqrt(sw + epsilon)) for w,vw, sw in zip(self.w, Vdw_corrected, Sdw_corrected)]
                self.b = [b - alpha * vb/(np.sqrt(sb + epsilon)) for b,vb, sb in zip(self.b, Vdb_corrected, Sdb_corrected)]
            
            if (_) % 200 == 0 or True:
                #calculate the training error / cost
                X = trainDatas[:, :nx].T
                y = trainDatas[:, nx:].T
                a = self.predict(X)
                train_accuracy = np.mean(np.argmax(a, axis=0) == np.argmax(y, axis=0))
                cost = self.costFunc(a, y)
                cost = np.mean(np.sum(cost, axis = 0))
                if testDatas is not None:
                    #calculate the test error
                    X_test = testDatas[:, :-10].T
                    y_test = testDatas[:, -10:].T
                    test = self.predict(X_test)
                    test_accuracy = np.mean(np.argmax(test, axis=0) == np.argmax(y_test, axis=0))
                    print(_, "cost: ",cost, "train accuracy: ", train_accuracy, "test accuracy: ", test_accuracy)
                else:
                    print(_, "cost: ", cost)
        
        
    def predict(self, X):
        a = X
        for i in range(1, self.num_layers + 1):
            z_i = np.dot(self.w[i], a) + self.b[i]
            if i == self.num_layers:
                a = self.sigmoid(z_i)
            else:
                a = self.actFunc(z_i)
        return a
    
    
    
    #---------------- define backpropagation function here ---------------------------#
    def backprop(self, x, y):
        '''calculate the dw and db through the back propagation algorithm'''
        #forward propagation
        Z = [0]
        A = [x]
        m = x.shape[1]  # in fact, m = batchSize
        for i in range(1, self.num_layers + 1):  # i = {1, 2, ..., L}, L = self.num_layers
            z_i = np.dot(self.w[i], x) + self.b[i]  #dims = (ni, m)
            Z.append(z_i)
            if i == self.num_layers:    #activation function of last layer is sigmoid
                x = self.sigmoid(z_i)
            else:
                x = self.actFunc(z_i)    #dims = (ni, m)
            A.append(x)

        #back propagation
        dw = [np.zeros_like(i) for i in self.w]
        db = [np.zeros_like(i) for i in self.b]
#         dz = self.costFunc_derivative(A[-1], y) * self.actFunc_derivative(Z[-1]) # the last layer's dz
        dz = A[-1] - y   #this means that the last activition function is sigmoid
        dw[-1] = np.dot(dz, A[-2].T) / m  
        db[-1] = np.mean(dz, axis = 1, keepdims=True)
        for i in range(self.num_layers-1, 0, -1):
            dz = self.actFunc_derivative(Z[i]) * np.dot(self.w[i+1].T, dz)   #Given (i+1)th dz to calculate ith dz
            dw[i] = np.dot(dz, A[i-1].T) / m
            db[i] = np.mean(dz, axis = 1, keepdims=True)
            
        return dw,db
    
    def backprop_L2_regularization(self, x, y, lambd = 1):
        #forward propagation
        Z = [0]
        A = [x]
        m = x.shape[1]  # in fact, m = batchSize
        for i in range(1, self.num_layers + 1):  # i = {1, 2, ..., L}, L = self.num_layers
            z_i = np.dot(self.w[i], x) + self.b[i]  #dims = (ni, m)
            Z.append(z_i)
            if i == self.num_layers:    #activation function of last layer is sigmoid
                x = self.sigmoid(z_i)
            else:
                x = self.actFunc(z_i)    #dims = (ni, m)
            A.append(x)
            
        #back propagation
        dw = [np.zeros_like(i) for i in self.w]
        db = [np.zeros_like(i) for i in self.b]
#         dz = self.costFunc_derivative(A[-1], y) * self.actFunc_derivative(Z[-1]) # the last layer's dz
        dz = A[-1] - y   #this means that the last activition function is sigmoid
        dw[-1] = np.dot(dz, A[-2].T) / m  + lambd*self.w[-1] / m
        db[-1] = np.mean(dz, axis = 1, keepdims=True)
        for i in range(self.num_layers-1, 0, -1):
            dz = self.actFunc_derivative(Z[i]) * np.dot(self.w[i+1].T, dz)   #Given (i+1)th dz to calculate ith dz
            dw[i] = np.dot(dz, A[i-1].T) / m  + lambd*self.w[i] / m
            db[i] = np.mean(dz, axis = 1, keepdims=True)
            
        return dw,db
        
    def backprop_dropout_regularization(self, x, y, keepprop = 0.8):
         #forward propagation
        Z = [0]
        A = [x]   #keepprop of first layer is equal to 1 
        m = x.shape[1]  # in fact, m = batchSize
        for i in range(1, self.num_layers + 1):  # i = {1, 2, ..., L}, L = self.num_layers
            z_i = np.dot(self.w[i], x) + self.b[i]  #dims = (ni, m)
            Z.append(z_i)
            if i == self.num_layers:    #activation function of last layer is sigmoid
                x = self.sigmoid(z_i)
            else:
                x = self.actFunc(z_i)    #dims = (ni, m)
#             if i < self.num_layers :   # to gurantee the keepprop of the last layer is equal to 1
                d = np.random.rand(x.shape[0], x.shape[1]) < keepprop
                x *= d
                x /= keepprop
            A.append(x)

        #back propagation
        dw = [np.zeros_like(i) for i in self.w]
        db = [np.zeros_like(i) for i in self.b]
#         dz = self.costFunc_derivative(A[-1], y) * self.actFunc_derivative(Z[-1]) # the last layer's dz
        dz = A[-1] - y   #this means that the last activition function is sigmoid
        dw[-1] = np.dot(dz, A[-2].T) / m  
        db[-1] = np.mean(dz, axis = 1, keepdims=True)
        for i in range(self.num_layers-1, 0, -1):
            dz = self.actFunc_derivative(Z[i]) * np.dot(self.w[i+1].T, dz)   #Given (i+1)th dz to calculate ith dz
            dw[i] = np.dot(dz, A[i-1].T) / m
            db[i] = np.mean(dz, axis = 1, keepdims=True)
            
        return dw,db
    
    
    
    #-------------------- define cost function here ------------------------------#
    def costFunction_crossEntropy(self, a, y):
        return -(y*np.nan_to_num(np.log(a)) + (1-y)*np.nan_to_num(np.log(1-a)))
    def costFunction_crossEntropy_derivative(self, a, y):
        return np.nan_to_num(-y/a + (1-y)/(1-a))
    
    
    
    #---------------- define activition function here ---------------------------#
    def sigmoid(self, z):
        return np.nan_to_num(1 / (1 + np.exp(-z)))
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def tanh(self, z):
        '''when using the tanh activition , the last activition of neural network should be sigmoid
        because the cost function need the input that is positive'''
        return np.tanh(z)
    def tanh_derivative(self, z):
        return 1 - self.tanh(z)**2
    
    def ReLU(self, z):
#         return np.maximum(np.zeros_like(z), z)
        return np.maximum(0.01*z, z)   #leaky ReLU
    def ReLU_devirative(self, z):
        return np.where(z >= 0, 1, 0.01)



if __name__ == "__main__":
    print(__doc__)

