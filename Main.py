import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


def sigmoid(x):
    return  1 /( 1 + (math.e)**(-1 * x))

def sigmoid_deriviate(x):
    a = sigmoid(x)
    a = np.reshape(a,(-1,1))
    b = 1 - sigmoid(x)
    b = np.reshape(b,(-1,1))
    b = np.transpose(b)
    return np.diag(np.diag(np.matmul(a,b)))



# Building AutoEncoder Layer by Layer
#AE1

split_ratio = 0.7
eta = 0.3
epochs = 50
data = pd.read_excel('data.xlsx',header=None)
data = np.array(data)
data = data[:3726,:]


minn = np.min(data[:,0])
maxx = np.max(data[:,0])

for i in range(np.shape(data)[0]):
    for j in range(np.shape(data)[1]):
        data[i,j] = (data[i,j] - minn) / (maxx - minn)

split_line_number = int(np.shape(data)[0] * split_ratio)
x_train = data[:split_line_number,:15]
x_test = data[split_line_number:,:15]

input_dimension = np.shape(x_train)[1]
l1_neurons = 12
l2_neurons = 15

w1 = np.random.uniform(low=-1,high=1,size=(l1_neurons,input_dimension))
w2 = np.random.uniform(low=-1,high=1,size=(l2_neurons,l1_neurons))

MSE_train_ae1 = []
MSE_test_ae1 = []
print("_______________________")
print("Auto Encoder 1: ")
for i in range(epochs):
    
        
    sqr_err_epoch_train = []
    sqr_err_epoch_test = []
    
    output_train = []
    output_test = []
    

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward
    
            # Layer 1
    
        net1 = np.matmul(w1,np.transpose(x_train[j]))
        net1 = np.reshape(net1,(-1,1))
        o1 = sigmoid(net1)

    
            # Layer 2
        net2 = np.matmul(w2,o1)
        o2 = net2
    
    
        output_train.append(o2[0])
    
        # Error
        err = x_train[j] - o2[0]
        sqr_err_epoch_train.append(err**2)
    

            # Back propagation
        w2 = np.subtract(w2 , (eta * np.reshape(err,(-1,1)) * -1 * 1 * np.transpose(o1)))
        w1 = np.transpose(w2)
    
    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train_ae1.append(mse_epoch_train)
    
    for j in range(np.shape(x_test)[0]):
        # Feed-Forward
    
            # Layer 1
        net1 = np.matmul(w1,np.transpose(x_test[j]))
        net1 = np.reshape(net1,(-1,1))
        o1 = sigmoid(net1)
    
            # Layer 2
        net2 = np.matmul(w2,o1)
        o2 = net2
    
    
        output_test.append(o2[0])
    
        # Error
        err = x_test[j] - o2[0]
        sqr_err_epoch_test.append(err ** 2)
    
    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test_ae1.append(mse_epoch_test)
    print("Error Epoch " + str(i) + ": " + str(sum(mse_epoch_test)/15))

# AE 2
print("_______________________")
print("Auto Encoder 2: ")

l3_neurons = 8
l4_neurons = 12
input_dimension = l4_neurons


w3 = np.random.uniform(low=-1,high=1,size=(l3_neurons,input_dimension))
w4 = np.random.uniform(low=-1,high=1,size=(l4_neurons,l3_neurons))

MSE_train_ae2 = []
MSE_test_ae2 = []

for i in range(epochs):
    
        
    sqr_err_epoch_train = []
    sqr_err_epoch_test = []
    
    output_train = []
    output_test = []
    

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward
    
            # AE1 Layer 1
    
        net1 = np.matmul(w1,np.transpose(x_train[j]))
        net1 = np.reshape(net1,(-1,1))
        o1 = sigmoid(net1)

            # AE2 Layer 1 (Layer 3)
            
        net3 = np.matmul(w3,o1)
        net3 = np.reshape(net3,(-1,1))
        o3 = sigmoid(net3)

            # AE2 Layer 2 (Layer 4)
        net4 = np.matmul(w4,o3)
        o4 = net4
        
        output_train.append(o4[0])
    
        # Error
        err = o1 - o4[0]
        sqr_err_epoch_train.append(err**2)
    

            # Back propagation
        w4 = np.subtract(w4 , (eta * np.reshape(err,(-1,1)) * -1 * 1 * np.transpose(o3)))
        w3 = np.transpose(w4)
    
    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train_ae2.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward
    
            # AE2 Layer 1 (Layer 3)
        net3 = np.matmul(w3,o1)
        net3 = np.reshape(net3,(-1,1))
        o3 = sigmoid(net3)
    
            # AE2 Layer 2 (Layer 4)
        net4 = np.matmul(w4,o3)
        o4 = net4
    
    
        output_test.append(o4[0])
    
        # Error
        err = o1 - o4[0]
        sqr_err_epoch_test.append(err ** 2)
    
    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test_ae2.append(mse_epoch_test)
    print("Error Epoch " + str(i) + ": " + str(sum(mse_epoch_test)/15))

# AE 3
print("_______________________")
print("Auto Encoder 3: ")

l5_neurons = 6
l6_neurons = 8
input_dimension = l6_neurons


w5 = np.random.uniform(low=-1,high=1,size=(l5_neurons,input_dimension))
w6 = np.random.uniform(low=-1,high=1,size=(l6_neurons,l5_neurons))

MSE_train_ae3 = []
MSE_test_ae3 = []

for i in range(epochs):
    
        
    sqr_err_epoch_train = []
    sqr_err_epoch_test = []
    
    output_train = []
    output_test = []
    

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward
    
            # AE1 Layer 1
    
        net1 = np.matmul(w1,np.transpose(x_train[j]))
        net1 = np.reshape(net1,(-1,1))
        o1 = sigmoid(net1)

            # AE2 Layer 1 (Layer 3)
            
        net3 = np.matmul(w3,o1)
        net3 = np.reshape(net3,(-1,1))
        o3 = sigmoid(net3)

            # AE3 Layer 1 (Layer 5)
            
        net5 = np.matmul(w5,o3)
        net5 = np.reshape(net5,(-1,1))
        o5 = sigmoid(net5)

            # AE3 Layer 2 (Layer 6)
        net6 = np.matmul(w6,o5)
        o6 = net6
        
        output_train.append(o6[0])
    
        # Error
        err = o3 - o6[0]
        sqr_err_epoch_train.append(err**2)
    

            # Back propagation
        w6 = np.subtract(w6 , (eta * np.reshape(err,(-1,1)) * -1 * 1 * np.transpose(o5)))
        w5 = np.transpose(w6)
    
    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train_ae3.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward
    
            # AE3 Layer 1 (Layer 5)
        net5 = np.matmul(w5,o3)
        net5 = np.reshape(net5,(-1,1))
        o5 = sigmoid(net5)
    
            # AE3 Layer 2 (Layer 6)
        net6 = np.matmul(w6,o5)
        o6 = net6
    
    
        output_test.append(o6[0])
    
        # Error
        err = o3 - o6[0]
        sqr_err_epoch_test.append(err ** 2)
    
    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test_ae3.append(mse_epoch_test)
    print("Error Epoch " + str(i) + ": " + str(sum(mse_epoch_test)/15))
##############################################################################
#MLP    

y_train = data[:split_line_number,15]
y_test = data[split_line_number:,15]

input_AE = data[:,:15]
AE_output = []

for j in range(np.shape(input_AE)[0]):
    # Feed-Forward

        # AE1 Layer 1 (Layer 1)
    net1 = np.matmul(w1,np.transpose(input_AE[j]))
    net1 = np.reshape(net1,(-1,1))
    o1 = sigmoid(net1)

        # AE2 Layer 1 (Layer 3)
    net3 = np.matmul(w3,o1)
    o3 = sigmoid(net3)
    
        # AE3 Layer 1 (Layer 5)
    net5 = np.matmul(w5,o3)
    o5 = sigmoid(net5)


    AE_output.append(o5)
AE_output = np.array(AE_output)
AE_output = AE_output[:,:,0]

split_line_number = int(np.shape(data)[0] * split_ratio)
x_train = AE_output[:split_line_number,:15]
x_test = AE_output[split_line_number:,:15]


input_dimension = np.shape(x_train)[1]
mlp_l1_neurons = 6
mlp_l2_neurons = 1


mlp_w1 = np.random.uniform(low=-1,high=1,size=(input_dimension,mlp_l1_neurons))
mlp_w2 = np.random.uniform(low=-1,high=1,size=(mlp_l1_neurons,mlp_l2_neurons))

MSE_train = []
MSE_test = []
for i in range(epochs):


    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    o1 = np.zeros((5,1))

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

            # Layer 1

        net1 = np.matmul(x_train[j],mlp_w1)
        o1 = sigmoid(net1)
        o1 = np.reshape(o1,(-1,1))


            # Layer 2
        net2 = np.matmul(np.transpose(o1),mlp_w2)
        o2 = net2


        output_train.append(o2[0])

        # Error
        err = y_train[j] - o2[0]
        sqr_err_epoch_train.append(err**2)

        # Back propagation
        f_driviate = sigmoid_deriviate(net1)
        mlp_w2_f_deriviate = np.matmul(f_driviate,mlp_w2)
        mlp_w2_f_deriviate_x = np.matmul(mlp_w2_f_deriviate,np.transpose(np.reshape(x_train[j],(-1,1))))
        mlp_w1 = np.subtract(mlp_w1 , np.transpose((eta * err * -1 * 1 * mlp_w2_f_deriviate_x)))
        mlp_w2 = np.subtract(mlp_w2 , (eta * err * -1 * 1 * o1))

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        # Layer 1
        net1 = np.matmul(x_test[j], mlp_w1)
        o1 = sigmoid(net1)
        o1 = np.reshape(o1, (-1, 1))

        # Layer 2
        net2 = np.matmul(np.transpose(o1), mlp_w2)
        o2 = net2

        output_test.append(o2[0])

        # Error
        err = y_test[j] - o2[0]
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)


    # Ploy fits

        # Train
    m_train , b_train = np.polyfit(y_train,output_train,1)

        # Test

    m_test , b_test = np.polyfit(y_test, output_test, 1)

    print(m_train,b_train,m_test,b_test)

    # Plots
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(MSE_train,'b')
    axs[0, 0].set_title('MSE Train')
    axs[0, 1].plot(MSE_test,'r')
    axs[0, 1].set_title('Mse Test')

    axs[1, 0].plot(y_train, 'b')
    axs[1, 0].plot(output_train,'r')
    axs[1, 0].set_title('Output Train')
    axs[1, 1].plot(y_test, 'b')
    axs[1, 1].plot(output_test,'r')
    axs[1, 1].set_title('Output Test')

    axs[2, 0].plot(y_train, output_train, 'b*')
    axs[2, 0].plot(y_train, m_train*y_train+b_train,'r')
    axs[2, 0].set_title('Regression Train')
    axs[2, 1].plot(y_test, output_test, 'b*')
    axs[2, 1].plot(y_test,m_test*y_test+b_test,'r')
    axs[2, 1].set_title('Regression Test')
    if i == (epochs - 1):
        plt.savefig('Results.jpg')
    plt.show()
    time.sleep(1)
    plt.close(fig)

    
    

