# Program to Replicate the XOR Gate
# Following report guidlines author names and emails are excluded
import numpy as np
import matplotlib.pyplot as plt
# Function to return output of the network
def xor_net(x1, x2, xor_weight):
    hidden_node_1 = 1.*xor_weight[0] + x1*xor_weight[1] + x2*xor_weight[2]
    hidden_node_1_sigmoid_activation = 1./(1.+np.exp(-hidden_node_1))
    hidden_node_2 = 1.*xor_weight[3] + x1*xor_weight[4] + x2*xor_weight[5]
    hidden_node_2_sigmoid_activation = 1./(1.+np.exp(-hidden_node_2))
    output_node = 1.*xor_weight[6] + hidden_node_1_sigmoid_activation*xor_weight[7] + hidden_node_2_sigmoid_activation*xor_weight[8]
    output_node_sigmoid_activation_function = 1./(1.+np.exp(-output_node))
    return output_node_sigmoid_activation_function

# Function to calculate the mean squared error
def mse(xor_in, xor_out, xor_weight):
    error = 0.
    for i in range(len(xor_in)):
        error += (xor_out[i] - xor_net(xor_in[i,0], xor_in[i,1], xor_weight))**(2.)
    return (error/len(xor_in))

# Function to calculate the gradient of the error
def grdmse(xor_in, xor_out, xor_weight, ep):
    grad = np.zeros(len(xor_weight))
    temp = np.copy(xor_weight)
    for i in range(len(grad)):
        temp[i] += ep
        grad[i] = ( mse(xor_in, xor_out, temp) - mse(xor_in, xor_out, xor_weight) ) / ep
        temp[i] -= ep
    return grad

# Function to find digit from activation function
def classification(x1, x2, xor_weight):
    if xor_net(x1, x2, xor_weight) > 0.5:
        return 1.
    else:
        return 0.

# Function to calculate the number of misclassifications
def misclassifications(xor_in, xor_out, xor_weight):
    count = 0
    for i in range(len(xor_in)):
        if xor_out[i] != classification(xor_in[i,0], xor_in[i,1], xor_weight):
            count += 1
    return count

# Read in data
XOR_in = np.array([[0,0], [0,1], [1,0], [1,1]])
XOR_out = np.array([0,1,1,0])

# Specify training parameters
epsilon = (10.)**(-3.)
iterations = 5000
step_size = 0.9

# Intialize weights
# Note indecies correspond to hidden_node_1_bias, hidden_node_1_weight_1, hidden_node_1_weight_2, hidden_node_2_bias, hidden_node_2_weight_1, hidden_node_2_weight_2, output_node_bias, output_node_weight_1, output_node_weight_2
xor_weights = 1.*np.random.rand(9)
#concerning higher initial weights
hxor_weights = 5.*xor_weights
# Run the network
xiteration=np.arange(1,iterations,100)
ymseit=[]
for i in range(iterations):
    xor_weights = xor_weights - ( step_size * grdmse(XOR_in, XOR_out, xor_weights, epsilon) )
    if i % 100 == 1:
        print(mse(XOR_in, XOR_out, xor_weights), misclassifications(XOR_in, XOR_out, xor_weights))
        ymseit.append(mse(XOR_in, XOR_out, xor_weights))
# Run the network with high initial weight
hxiteration=np.arange(1,iterations,100)
hymseit=[]
for i in range(iterations):
    hxor_weights = hxor_weights - ( step_size * grdmse(XOR_in, XOR_out, hxor_weights, epsilon) )
    if i % 100 == 1:
        print(mse(XOR_in, XOR_out, hxor_weights), misclassifications(XOR_in, XOR_out, hxor_weights))
        hymseit.append(mse(XOR_in, XOR_out, hxor_weights))

#concerning the learning rate
ymsestep=[]
xstep_size=[0.3,0.5,0.7,0.9]
new_xor_weights=np.random.rand(9)
for i in xstep_size:
    for j in range(3000):
        new_xor_weights = new_xor_weights - ( step_size * grdmse(XOR_in, XOR_out, new_xor_weights, epsilon) )
        if j  == 2999:
            #print(mse(XOR_in, XOR_out, xor_weights), misclassifications(XOR_in, XOR_out, xor_weights))
            ymsestep.append(mse(XOR_in, XOR_out, new_xor_weights))


#Plot
#Plot
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax1.plot(xiteration,ymseit,label='small initial weights')
ax1.plot(xiteration,hymseit,label='5 times initial weights')
ax1.legend(loc='best')
ax1.set_xlabel('iterations')
ax1.set_ylabel('mean squared error')
ax1.set_title('training with fixed step size = 0.9')
ax2=fig.add_subplot(1,2,2)
ax2.plot(xstep_size,ymsestep)
ax2.set_xlabel('learning rate')
ax2.set_ylabel('mean squared error')
ax2.set_title('traning with fixed iterations = 5000')





'''
###################################
#with different activation function
####################################

def xor_net_tanh(x1, x2, xor_weight):
    hidden_node_1 = 1.*xor_weight[0] + x1*xor_weight[1] + x2*xor_weight[2]
    hidden_node_1_tanh_activation = np.tanh(hidden_node_1)
    hidden_node_2 = 1.*xor_weight[3] + x1*xor_weight[4] + x2*xor_weight[5]
    hidden_node_2_tanh_activation = np.tanh(hidden_node_2)
    output_node = 1.*xor_weight[6] + hidden_node_1_tanh_activation*xor_weight[7] + hidden_node_2_tanh_activation*xor_weight[8]
    output_node_tanh_activation_function = np.tanh(output_node)
    return output_node_tanh_activation_function
# Function to calculate the mean squared error
def mse_tanh(xor_in, xor_out, xor_weight):
    error = 0.
    for i in range(len(xor_in)):
        error += (xor_out[i] - xor_net_tanh(xor_in[i,0], xor_in[i,1], xor_weight))**(2.)
    return (error/len(xor_in))

# Function to calculate the gradient of the error
def grdmse(xor_in, xor_out, xor_weight, ep):
    grad = np.zeros(len(xor_weight))
    temp = np.copy(xor_weight)
    for i in range(len(grad)):
        temp[i] += ep
        grad[i] = ( mse(xor_in, xor_out, temp) - mse(xor_in, xor_out, xor_weight) ) / ep
        temp[i] -= ep
    return grad

xiteration_tanh=np.arange(1,iterations,100)
ymseit_tanh=[]
xor_weights_tanh = np.random.rand(9)
for i in range(iterations):
    xor_weights_tanh = xor_weights_tanh - ( step_size * grdmse(XOR_in, XOR_out, xor_weights_tanh, epsilon) )
    if i % 100 == 1:
        #print(mse_tanh(XOR_in, XOR_out, xor_weights))
        ymseit_tanh.append(mse_tanh(XOR_in, XOR_out, xor_weights_tanh))

ax3=fig.add_subplot(1,3,3)
ax3.plot(xiteration,ymseit_tanh)
ax3.set_xlabel('iterations')
ax3.set_ylabel('mean squared error')
ax3.set_title('use tanh function traning with fixed iterations = 5000')
'''
plt.show()