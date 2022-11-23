# In[]: Load library
from lib import braindecoding as bd


# In[]: Load dataset


handwriten_69=loadmat('./data/digit69_28x28.mat')
Y_train = handwriten_69['fmriTrn']
Y_test = handwriten_69['fmriTest']

X_train = handwriten_69['stimTrn']#90 gambar dalam baris isi per baris 784
X_test = handwriten_69['stimTest']#10 gambar dalam baris isi 784
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# In[]: X adalah gambar stimulus,ukuran pixel 28x28 = 784 di flatten sebelumnya dalam satu baris, 28 row x 28 column dengan channel 1(samaa kaya miyawaki)
resolution = 28
#channel di depan
#X_train = X_train.reshape([X_train.shape[0], 1, resolution, resolution])
#X_test = X_test.reshape([X_test.shape[0], 1, resolution, resolution])
#channel di belakang(edit rolly)
X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
# In[]: Normlization sinyal fMRI
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
Y_train = min_max_scaler.fit_transform(Y_train)     
Y_test = min_max_scaler.transform(Y_test)

print ('X_train.shape : ')
print (X_train.shape)
print ('Y_train.shape')
print (Y_train.shape)
print ('X_test.shape')
print (X_test.shape)
print ('Y_test.shape')
print (Y_test.shape)
numTrn=X_train.shape[0]
numTest=X_test.shape[0]
# %%
