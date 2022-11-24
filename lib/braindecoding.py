import numpy as np
import matplotlib.pyplot as plt


from scipy.io import loadmat
from sklearn import preprocessing


from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Lambda, Dense

from keras.regularizers import l2
from keras import backend as K

def getDataset69(datapath):
    handwriten_69=loadmat(datapath)
    Y_train = handwriten_69['fmriTrn']
    Y_test = handwriten_69['fmriTest']

    X_train = handwriten_69['stimTrn'].astype('float32') / 255. #90 gambar dalam baris isi per baris 784
    X_test = handwriten_69['stimTest'].astype('float32') / 255. #10 gambar dalam baris isi 784
    return X_train,X_test,Y_train,Y_test

def reshape2D(X_train,X_test):
    resolution = 28
    #channel di depan
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution])
    #channel di belakang(edit rolly)
    #X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    #X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
    return X_train,X_test

def normalizefMRI(Y_train,Y_test):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
    Y_train = min_max_scaler.fit_transform(Y_train)     
    Y_test = min_max_scaler.transform(Y_test)
    return Y_train,Y_test

def initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # FC Neural Network
    model = Sequential()
    model.add(Dense(2048, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    model.add(Dense(1024, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    model.add(Dense(512, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score, dense modif ke outputan 28x28=784 atau 10*10=100
    prediction = Dense(784,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net