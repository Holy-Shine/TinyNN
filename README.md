# TinyNN
小型神经网络库，参考自DeepLearning Specialization

## 使用方法

1. 导入

   `from TinyNN import *`

2. 定义优化器

   `opt = Optimizer(...)`

   **可选参数：**

   - 激活函数：`activation=relu or sigmoid` ，默认sigmoid
   - Dropout：`dropout=True or False`表示是否开启。默认不开启
   - 神经元保存概率：`keep_prob=prob`，dropout不开启情况下为1
   - 学习率：`learning_rate=lr`

3. 定义网络模型

   `NN = TinyNN(layer_dims=[3,2,1],optimizer=opt)`

4. 组织数据

   例如

   ```python
   X=[
       [1,2,3,4,5],
       [6,7,8,9,10]
   ]
   Y=[
       [1,0,1,0,1]
   ]
   ## 注意行维度表示数据值，列维度表示样本个数
   trainX , trainY = np.array(X), np.array(Y) 
   ```

5. 训练

   `NN.train(train_X = trainX, train_Y = trainY, n_epochs=1000, print_cost=True)`

   ​