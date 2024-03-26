# baseline MLP for mnist dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Lambda, Layer, LeakyReLU, BatchNormalization,Activation
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import math
import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import Table
from keras.models import Model
from keras import backend as K

def generate_weights(feh,bins=20):
    b_size=[]
    sample_weights=np.ones(len(feh))
    bin_size=(max(feh)-min(feh))/bins
    a1=np.where(feh==min(feh))[0]
    for i in range(bins):
            b=np.where((feh>min(feh)+i*bin_size)&(feh<=min(feh)+(i+1)*bin_size))[0]
            b_size.append(b.shape[0])
    for i in range(bins):
        a0=np.where((feh>min(feh)+i*bin_size)&(feh<=min(feh)+(i+1)*bin_size))[0]
        sample_weights[a0]=np.sqrt(max(b_size)/(a0.shape[0]))
        if i==0:
            sample_weights[a1]=np.sqrt(max(b_size)/(a0.shape[0]))
    return np.array(sample_weights)


train_x=np.random.uniform(0, 1, (189654,13))#The input vector: 13 stellar colors; For example, it is an array with the shape of (189654,13);

train_y=np.random.uniform(-2.5, 0.5, (189654,))#The input vector: [Fe/H] values; For example, it is an array with the shape of (189654,1); 

sample_weights=generate_weights(train_y,bins=20)#The 'sample_weights' and the 'train_y' should have the same shape.

# define baseline model
def baseline_model():
    # create model
    input_x0 = Input(shape=(13,), name='inp0')   
    encoded0 = Dense(300, name='encoded0',kernel_regularizer=regularizers.l2(0.00005),kernel_initializer = "normal")(input_x0)
    a0=LeakyReLU(alpha=0.01)(encoded0)
    encoded1 = Dense(200, name='encoded1',kernel_regularizer=regularizers.l2(0.00005),kernel_initializer = "normal")(a0)
    a1=LeakyReLU(alpha=0.01)(encoded1)
    encoded2 = Dense(100, name='encoded2',kernel_regularizer=regularizers.l2(0.00005),kernel_initializer = "normal")(a1)
    a2=LeakyReLU(alpha=0.01)(encoded2)
    output_y0 = Dense(1,name='oup0',kernel_initializer = "normal")(a2)
    trainable_model=Model(inputs=input_x0, outputs=output_y0)
    trainable_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse')
    return trainable_model
model = baseline_model()

model.fit(train_x, train_y,epochs = 2, batch_size = 2000, sample_weight=sample_weights, verbose = 2,callbacks=[TensorBoard(log_dir='./mytensorboard')], shuffle=True)
model.save('./model/model.h5')#The trained model will be put here.


#when applying the model to your data (eg., jplus dr1), please run the following code.
test_x=np.random.uniform(0, 1, (100,13))
model=load_model('./model/model.h5')
result=model.predict(test_x)


