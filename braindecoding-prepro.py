# In[]: Load library
from scipy.io import savemat, loadmat


# In[]: Load dataset
handwriten_69=loadmat('./data/digit69_28x28.mat')
Y_train = handwriten_69['fmriTrn']
Y_test = handwriten_69['fmriTest']

X_train = handwriten_69['stimTrn']#90 gambar dalam baris isi per baris 784
X_test = handwriten_69['stimTest']#10 gambar dalam baris isi 784
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
# %%
