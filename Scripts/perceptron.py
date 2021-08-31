import numpy as np

def sign(f):
    if f>=0:
        return 1
    else:
        return -1

def perceptron(data,labels,lr=0.1,max_iter=1_000):
    weights = np.ones(len(data[0]))
    bias = 1
    for n in range(max_iter):
        error = 0
        for i in range(len(data)): 
            x_i,y_i = data[i],labels[i]
            y_pred = sign(weights.T.dot(x_i)+bias)
            if y_pred*y_i<=0: # step 1: y_pred,y_i = (-1,1) OR (1,-1)
                # step 2: update weight/bias based on derivatives
                d_weights = -y_i*np.array(x_i)
                d_bias = -y_i
                weights = weights - lr*d_weights
                bias = bias - lr*d_bias
                error += 1
        if error == 0: # step 3, similar concept like tolerance - tol is used in sklearn
            print(f'Optimization finished with {n} iterations')
            break
    print('weights :', *weights)
    print('bias: ',bias)
    res = [weights.T.dot(x)+bias for x in data]
    res = list(map(sign,res))
    print('res: ',res)

# AND
data = [[0,0],[0,1],[1,0],[1,1]]
labels = [1,1,1,-1]
perceptron(data,labels)
# Optimization finished with 15 iterations
# weights : -0.19999999999999987 -0.09999999999999987
# bias:  0.2
# res:  [1, 1, 1, -1]

# XOR
# Try the famous XOR problem with one single perceptron
# Due to the nature of linear separability, it'll never classify dataset correctly.
labels = [1,-1,-1,1]
perceptron(data,labels)
# weights : 0.20000000000000015 0.20000000000000015
# bias:  -0.20000000000000004
# res:  [1, 1, 1, 1]