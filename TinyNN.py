
import numpy as np

np.random.seed(2)
def sigmoid(z):
    return 1./(1+np.exp(-z))

def relu(z):
    s = np.maximum(0,z)
    return s


class Optimizer(object):
    def __init__(self,activation='relu',cost_function='cross_entropy',learning_rate=0.1,dropout=True, keep_prob=1):
        self.activation = activation
        self.cost_func = cost_function
        self.dropout = dropout
        self.keep_prob = keep_prob
        if dropout==False:
            self.keep_prob=1.0
        self.lr = learning_rate


class TinyNN(object):
    def __init__(self,layer_dims=[],optimizer=None):
        '''
        简易神经网络类实例化

        :param layer_dims: 神经元个数
        :param optimizer:  优化器
        '''
        self.layer_dims = layer_dims
        self.optimizer = optimizer
        self._initialNN()

    def _initialNN(self):
        '''
        初始化网络参数W和b

        :return:
        '''
        self.W = []
        self.b = []
        L = len(self.layer_dims)-1  # layer的层数
        layer_dims = self.layer_dims

        for l in range(1, L+1):
            self.W.append(np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1]))  # use He initialization
            self.b.append(np.zeros((layer_dims[l],1)))

    def _forward_propagate(self, X, Y):
        '''
        正向传播

        '''
        L = len(self.layer_dims)-1
        cacheA=[]
        cacheZ=[]
        cacheD=[]
        A = X
        W = self.W
        b = self.b
        opt = self.optimizer
        m = X.shape[-1]
        for i in range(0,L-1):   # 前n-1层用relu
            Z = np.dot(W[i], A)+b[i]
            if opt.activation == 'relu':
                A =  relu(Z)
            if opt.activation == 'sigmoid':
                A = sigmoid(Z)

            # Dropout
            if self.optimizer.dropout==True:
                D = np.random.rand(A.shape[0], A.shape[1])
                D = np.where(D<=self.optimizer.keep_prob,1,0)
                A = np.multiply(A,D)
                A = A/self.optimizer.keep_prob
                cacheD.append(D)
            cacheA.append(A)
            cacheZ.append(Z)


        # 最后一层用sigmoid
        Z = np.dot(W[L-1],A)
        A = sigmoid(Z)
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        cacheA.append(A)
        cacheZ.append(Z)
        cache = {
            'cacheA':cacheA,
            'cacheZ':cacheZ,
            'cacheD':cacheD
        }
        return cost,cache

    def _backward_propagate(self, X, Y):
        '''
        反向传播

        '''
        cacheZ = self.cache['cacheZ'] # Z1,Z2,...,Zn
        cacheA = self.cache['cacheA'] # A1,A2,...,An
        cacheD = self.cache['cacheD'] # D1,D2,...,Dn-1
        L = len(self.layer_dims)-1
        m = X.shape[1]

        # 最后一层是sigmoid
        dZ = None
        A = cacheA[L-1]
        if self.optimizer.cost_func=='cross_entropy': # 损失函数是交叉熵
            dZ = A-Y
        dW = 1./m * np.dot(dZ, cacheA[L-2].T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        self.W[L-1]-=self.optimizer.lr*dW
        self.b[L-1]-=self.optimizer.lr*db

        # 往前推每一层都是relu
        for i in range(L-2,0,-1):
            dA = np.dot(self.W[i+1].T,dZ)
            # dropout
            if self.optimizer.dropout:
                dA = dA * cacheD[i]
                dA = dA/self.optimizer.keep_prob
            if self.optimizer.activation=='relu':
                dZ = np.multiply(dA, np.int64(cacheA[i]>0)) # relu的导数
            if self.optimizer.activation=='sigmoid':
                dZ = dA*cacheA[i]*(1-cacheA[i])
            dW = 1./m * np.dot(dZ, cacheA[i-1].T)
            db = 1./m * np.sum(dZ, axis=1, keepdims=True)
            self.W[i]-=self.optimizer.lr*dW
            self.b[i] -= self.optimizer.lr * db

        # 输入层
        dA = np.dot(self.W[1].T, dZ)
        if self.optimizer.dropout:
            dA = dA*cacheD[0]
            dA = dA/self.optimizer.keep_prob
        if self.optimizer.activation == 'relu':
            dZ = np.multiply(dA, np.int64(cacheA[0] > 0))  # relu的导数
        if self.optimizer.activation == 'sigmoid':
            dZ = dA * cacheA[0] * (1 - cacheA[0])
        dW = 1./m * np.dot(dZ, X.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        self.W[0] -= self.optimizer.lr * dW
        self.b[0] -= self.optimizer.lr * db








    def train(self,train_X, train_Y,n_epochs = 1000,  print_cost=False):
        '''
        训练网络

        :param train_X: 输入X
        :param train_Y: 标签
        :param n_epochs: 迭代次数
        :param print_cost: 是否打印每轮的损失
        :return:
        '''
        assert train_X.shape[-1]==train_Y.shape[-1],\
            'dimension of train_X({}) and train_Y({}) is not match'.format(train_X.shape[-1],train_Y.shape[-1])
        assert train_X.shape[0]==self.layer_dims[0],\
            'train_X\'s dimension do not match the nn_input dim: input_dim({}), nn_input_dim({})'.format(train_X.shape[0],self.layer_dims[0])
        costs = []
        m = train_X.shape[-1]
        for i in range(n_epochs):
            self.cost, self.cache = self._forward_propagate(train_X,train_Y)
            if print_cost and i%100==0:
                print ("Cost after iteration {}:{}".format(i, self.cost))
                costs.append(self.cost)
            self._backward_propagate(train_X,train_Y)


    def _forward_propagate_without_dropout(self,X):
        L = len(self.layer_dims)-1
        A = X
        W = self.W
        b = self.b
        opt = self.optimizer
        m = X.shape[-1]
        for i in range(0,L-1):   # 前n-1层用relu
            Z = np.dot(W[i], A)+b[i]
            if opt.activation == 'relu':
                A =  relu(Z)
            if opt.activation == 'sigmoid':
                A = sigmoid(Z)

        # 最后一层用sigmoid
        Z = np.dot(W[L-1],A)
        A = sigmoid(Z)
        return A

    def predcit_dec(self, X):
        a = self._forward_propagate_without_dropout(X)
        predcitions = (a>0.5)
        return predcitions

    def plot_decision_boundary(self,model, X, y):
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min, y_max,h))

        Z = model(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)

        import matplotlib.pyplot as plt
        plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0,:], X[1, :], c=y, cmap=plt.cm.Spectral)
        plt.show()


