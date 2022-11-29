# baseline MLP for mnist dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Lambda, Layer, LeakyReLU, BatchNormalization,Activation
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib
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

# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.8,patience=10, mode='min',epsilon=0.0001,min_lr=0.00001)
# load data
bin_size=200
para_l=2
tol1=1e-08
sita0=10
sita1=0.20
sita2=-1.5
sita3=-0.7
gass_sita1=[-2.5]
gass_sita2=[0.50]
bin_num0=1
bin_num1=16
gass_bool=[False,True,True]

def func(x, a,u, sig):
    return a*np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))


def group_spectra(spectra,para):
    #(spectra2)teff<8000k & (spectra3)teff>7000k
    size=spectra.shape[0]
    print("spectra_size",size)
    spectra2=[]
    para2=[]
    spectra3=[]
    para3=[]    
    for i in range(0,size):
        if para[i][para_l]<sita0:
            spectra2.append(spectra[i])
            para2.append(para[i])   
    for i in range(0,size):
        if para[i][para_l]>0.05 and para[i][para_l]<1.0:
            spectra3.append(spectra[i])
            para3.append(para[i])
    return np.array(spectra2),np.array(para2),np.array(spectra3),np.array(para3)


#sample
def sample(spectra):
    spectra_size=spectra.shape[0]
    train_index1=np.linspace(0,spectra_size,num=int(spectra_size/4),endpoint= False,dtype=int)
    test_index1=[]
    for i in range(0,spectra_size):
        if i not in train_index1:
            test_index1.append(i)
    test_index1=np.array(test_index1)
    return test_index1,train_index1


def curve_error(error,bin_size):
    hist, bin_edges = np.histogram(error,bin_size,)
    popthunt, pcovhunt = curve_fit(func, bin_edges[0:bin_size], hist) 
    uhunt = popthunt[1]
    sighunt = popthunt[2]
    return uhunt,uhunt+sighunt,uhunt-sighunt


def media_value(x):
    x=x[np.argsort(x)]
    return x[np.int(x.shape[0]/2)]


def group_error(y_test,error,bin_num):
    spectra_1=np.c_[y_test,error]
    spectra_1=spectra_1[np.argsort(spectra_1[:,0])] 
    bin_size0=(sita2-gass_sita1[0])/bin_num0
    bin_size1=(sita3-sita2)/bin_num
    bin_size2=(gass_sita2[0]-sita3)/bin_num1
    u0=[]
    x0=[]
    u1=[]
    u2=[]
    if gass_bool[0] is True:
        for i in range(0,bin_num0):
            yl=np.where((spectra_1[:,0]>=gass_sita1[0]+i*bin_size0)&((spectra_1[:,0])<gass_sita1[0]+(i+1)*bin_size0))
            tmp=spectra_1[yl[0],:]
            a_x,a,b=curve_error(tmp[:,1],200)
            print((a-b)/2)
            u0.append(a)
            x0.append(media_value(tmp[:,0]))
            u1.append(b)
            u2.append(media_value(tmp[:,1]))
    if gass_bool[1] is True:
        for i in range (0,bin_num):
            if i in [100]:
                continue
            else:
                yl=np.where((spectra_1[:,0]>=sita2+i*bin_size1)&((spectra_1[:,0])<sita2+(i+1)*bin_size1))
                tmp=spectra_1[yl[0],:]
                a_x,a,b=curve_error(tmp[:,1],200)
                print((a-b)/2)
                u0.append(a)
                #x0.append(sita2+(i+0.8)*bin_size1)
                x0.append(media_value(tmp[:,0]))
                u1.append(b)
                #u2.append(a_x) 
                u2.append(media_value(tmp[:,1]))
    if gass_bool[2] is True:
        for i in range(0,bin_num1):
            yl=np.where((spectra_1[:,0]>=sita3+i*bin_size2)&((spectra_1[:,0])<sita3+(i+1)*bin_size2))
            tmp=spectra_1[yl[0],:]
            a_x,a,b=curve_error(tmp[:,1],200)
            print((a-b)/2)
            u0.append(a)
            x0.append(media_value(tmp[:,0]))
            u1.append(b)
            u2.append(media_value(tmp[:,1]))
        
    return x0,u0,u1,u2


data = fits.open('./data0/Jplus.lm.gaia_star_yanglin.fits')

MAG_APER6=data[1].data['MAG_APER6']
ERR_APER6=data[1].data['ERR_APER6']
FLAGS=data[1].data['FLAGS']
PHOT_BP_MEAN_MAG=data[3].data['PHOT_BP_MEAN_MAG']
PHOT_RP_MEAN_MAG=data[3].data['PHOT_RP_MEAN_MAG']
BP_RP=data[3].data['BP_RP']
EBV=data[3].data['EBV']
PHOT_BP_RP_EXCESS_FACTOR=data[3].data['PHOT_BP_RP_EXCESS_FACTOR']
SNRG=data[4].data['SNRG']
TEFF=data[4].data['TEFF']
LOGG=data[4].data['LOGG']
FEH=data[4].data['FEH']
ALPHA_FE=data[5].data['ALPHA_FE']
C_FE=data[5].data['C_FE']
CA_FE=data[5].data['CA_FE']
MG_FE=data[5].data['MG_FE']
N_FE=data[5].data['N_FE']

h0=BP_RP-1.36*EBV
h1=np.array((PHOT_RP_MEAN_MAG-MAG_APER6[:,0])-(-0.678)*EBV)
h2=np.array((PHOT_BP_MEAN_MAG-MAG_APER6[:,1])-(-0.339)*EBV)
h3=np.array((PHOT_RP_MEAN_MAG-MAG_APER6[:,2])-(-0.033)*EBV)
h4=np.array((PHOT_RP_MEAN_MAG-MAG_APER6[:,3])-0.439*EBV)
h5=np.array((PHOT_BP_MEAN_MAG-MAG_APER6[:,4])-(-1.379)*EBV)
h6=np.array((PHOT_BP_MEAN_MAG-MAG_APER6[:,5])-(-1.203)*EBV)
h7=np.array((PHOT_BP_MEAN_MAG-MAG_APER6[:,6])-(-1.158)*EBV)
h8=np.array((PHOT_BP_MEAN_MAG-MAG_APER6[:,7])-(-0.953)*EBV)
h9=np.array((PHOT_BP_MEAN_MAG-MAG_APER6[:,8])-(-0.809)*EBV)
h10=np.array((PHOT_BP_MEAN_MAG-MAG_APER6[:,9])-(-0.069)*EBV)
h11=np.array((PHOT_RP_MEAN_MAG-MAG_APER6[:,10])-(-0.454)*EBV)
h12=np.array((PHOT_RP_MEAN_MAG-MAG_APER6[:,11])-0.339*EBV)
spectra=np.c_[h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12]
index=[]

for i in range(0,189654):
    if FEH[i]>=-2.5 and FEH[i]<=0.5 and PHOT_
    _EXCESS_FACTOR[i]<1.25+0.06*(BP_RP[i]**2) and SNRG[i]>20 and all(ERR_APER6[i][0:12]<[0.01,0.01,0.01,0.01,0.03,0.03,0.03,0.02,0.02,0.02,0.01,0.01]) and (~FLAGS[i].any()):
        index.append(i)
para=np.c_[TEFF,LOGG,FEH]
para=para[index]

spectra=spectra[index]


spectra1,para1,spectra2,para2=group_spectra(spectra,para)    

train_index,test_index=sample(spectra1)

train_x=spectra1[train_index]
scaler = preprocessing.StandardScaler().fit(train_x)
train_x=scaler.transform(train_x)
train_y=para1[train_index]

test_x=spectra1[test_index]
test_x=scaler.transform(test_x)
test_y=para1[test_index]

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
    #kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=None),

    trainable_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse')
    
    return trainable_model
model = baseline_model()

# final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose = 1)
def network(train_y,test_y,model,path1,model_mlp,y_index,bin_num,lx1,ly1,lx2,ly2):
    tmp=test_y
    train_y1=[]
    for i in train_y:
        train_y1.append(i[y_index])      
    test_y1=[]
    for i in test_y:
        test_y1.append(i[y_index])          
    train_y=train_y1
    test_y=test_y1
    
    
    para_x=[]
    for i in tmp:
        para_x.append(i[para_l])          
    para_x=para_x
    
    
    X_train = np.array(train_x)
    y_train = np.array(train_y)
    y_train=y_train.reshape(y_train.shape[0],1)
    '''
    a1=np.where((y_train>-0.35)&(y_train<=0.35))
    a2=np.where((y_train<=0.35)|(y_train>-0.35))
    print(a1[0].shape,a2[0].shape)
    '''
    X_test = np.array(test_x)
    y_test = np.array(test_y)
    y_test=y_test.reshape(y_test.shape[0],1)
    
    v_data=(X_test,y_test)
    
    if model==False:
        sample_weights=np.ones(len(y_train))
        a=max(y_train)-min(y_train)
        a_num=20
        a_bin=a/a_num
        a1=np.where(y_train==min(y_train))
        b_size=[]
        for i in range(a_num):
            b=np.where((y_train>min(y_train)+i*a_bin)&(y_train<=min(y_train)+(i+1)*a_bin))
            b_size.append(b[0].shape[0])
        for i in range(a_num):
            a0=np.where((y_train>min(y_train)+i*a_bin)&(y_train<=min(y_train)+(i+1)*a_bin))
            sample_weights[a0[0]]=np.sqrt(max(b_size)/(a0[0].shape[0]))
            if i==0:
                sample_weights[a1[0]]=np.sqrt(max(b_size)/(a0[0].shape[0]))
        model_mlp.fit(X_train, y_train,epochs = 5000, batch_size = 2000, sample_weight=sample_weights, verbose = 0,callbacks=[TensorBoard(log_dir='./mytensorboard')],validation_data=v_data, shuffle=True)
        model_mlp.save(path1)
    else:
        model_mlp=load_model(path1)      
   

    #mlp_score = model.evaluate(X_test, y_test, verbose = 1)
    
    #print('sklearn\u591a\u5c42\u611f\u77e5\u5668-\u56de\u5f52\u6a21\u578b\u5f97\u5206',mlp_score)#\u9884\u6d4b\u6b63\u786e/\u603b\u6570
    X_train=X_test
    y_train=y_test
    result = model_mlp.predict(X_train)
    y1=np.array(result).reshape(-1,1)
    
    error1=y_train-y1
    j=0
    k=0
    y=[]
    error=[]
    para_x=[]
    #y_train=y_train-0.1
    #y1=y1-0.1
    for i in range(y1.shape[0]):
        if np.abs(error1[i])>sita1 or y_train[i]<gass_sita1[0] or y_train[i]>gass_sita2[0]:
            j+=1
        else:
            k+=1
            y.append(y1[i])
            error.append(error1[i])
            para_x.append(y_train[i])
    '''      
    print("outlier: ",j)

    fig=plt.figure(1,(16,4))

    para_x=np.array(para_x)

    
    ax2=fig.add_subplot(1,2,1)
    ax2.set_xlim(min(para_x), max(para_x))
    ax2.set_ylim(min(error), max(error))
    ax2.scatter(para_x, error, s=1.0, c='K', marker='.')
    plt.xlabel(lx1)
    plt.ylabel(ly1)

    
    ax2=fig.add_subplot(1,2,2)
    ax2.set_xlim(min(y_test), max(y_test))
    ax2.set_ylim(min(y_test), max(y_test))
    ax2.scatter(y_test, y1, s=1.0, c='K', marker='.')
    ax2.plot(y_test,y_test,linewidth=1,c='r')
    plt.xlabel(lx2)
    plt.ylabel(ly2)
    plt.show()
    
    hist, bin_edges = np.histogram(error,bin_size)
    popthunt, pcovhunt = curve_fit(func, bin_edges[0:bin_size], hist, p0=[2,2,2]) 
    ahunt = popthunt[0]
    uhunt = popthunt[1]
    sighunt = popthunt[2]
    yhuntvals = func(bin_edges[0:bin_size],ahunt,uhunt,sighunt)
    print("ahunt:", ahunt, "uhunt:", uhunt, "sighunt", sighunt, end=" ")
    plt.plot(bin_edges[0:bin_size], yhuntvals, 'r',label='TEFF')
    plt.show()
    '''   
   
    
    print("outlier: ",j)
    x0,u0,u1,u2=group_error(para_x,error,bin_num)
    fig=plt.figure(1,(16,4))
    ax1=fig.add_subplot(1,2,1)
    ax1.set_xlim(min(para_x), max(para_x))
    ax1.set_ylim(min(error), max(error))
    ax1.plot(x0,u2,'--',linewidth=1, c='r')
    ax1.scatter(para_x,error, s=1.0, c='k', marker='.')
    ax1.plot(x0,u0,linewidth=1, c='r')
    ax1.plot(x0,u1,linewidth=1, c='r')
    plt.xlabel(lx1)
    plt.ylabel(ly1)
    
    
    
    ax2=fig.add_subplot(1,2,2)
    ax2.set_xlim(min(y_train), max(y_train))
    ax2.set_ylim(min(y_train), max(y_train))
    ax2.scatter(y_train, y1, s=1.0, c='K', marker='.')
    ax2.plot(y_train,y_train,linewidth=1,c='r')
    plt.xlabel(lx2)
    plt.ylabel(ly2)
    plt.show()
    
    hist, bin_edges = np.histogram(error1,bin_size)
    popthunt, pcovhunt = curve_fit(func, bin_edges[0:bin_size], hist) 
    ahunt = popthunt[0]
    uhunt = popthunt[1]
    sighunt = popthunt[2]
    yhuntvals = func(bin_edges[0:bin_size],ahunt,uhunt,sighunt)
    print("ahunt:", ahunt, "uhunt:", uhunt, "sighunt", sighunt, end=" ")
    plt.plot(bin_edges[0:bin_size], yhuntvals, 'r',label='TEFF')
    plt.show()
    
     



#network(train_y,test_y,False,'./model/mlp_teff.pkl', model,para_l,10,"$\mathregular{T_{eff} (K)}$",r'$\Delta{T_{eff} (K)}$',r'${{T}_{eff}}^{LAMOST}$',r'${{T}_{eff}}^{MLP}$')
#network(train_y,test_y,False,'./model/mlp_logg.pkl', model,para_l,10,'log g (dex)', r'$\Delta{log g (dex)}$',r'${log g}_{LAMOST}$', r'${log g}_{MLP}$')
network(train_y,test_y,True,'./model/mlp_feh.h5', model,para_l,3,'[Fe/H] (dex)', r'$\Delta{[Fe/H] (dex)}$',r'${[Fe/H]]}_{LAMOST}$',r'${[Fe/H]]}_{MLP}$')
#network(train_y,test_y,True,'./model/mlp_aheh.h5', model,para_l,20,'['+r'$\alpha$'+'/Fe] (dex)',r'$\Delta$'+'['+r'$\alpha$'+'/Fe] (dex)','['+r'$\alpha$'+r'${/Fe]}_{LAMOST}$','['+r'$\alpha$'+r'${/Fe]}_{MLP}$')
#network(train_y,test_y,False,'./model/mlp_cfe.h5', model,para_l,20,'[C/Fe] (dex)', r'$\Delta{[C/Fe] (dex)}$',r'${[C/Fe]]}_{LAMOST}$',r'${[C/Fe]]}_{MLP}$')
