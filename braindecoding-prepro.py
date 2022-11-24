# In[]: Load library
from lib import braindecoding as bd
from matplotlib.pyplot import imshow,imsave
import numpy as np
# In[]: Load dataset

X_train,X_test,Y_train,Y_test = bd.getDataset69('./data/digit69_28x28.mat')


# In[]: normalize fmri
Y_train,Y_test = bd.normalizefMRI(Y_train,Y_test)
# In[]: reshape stimuli
X_train2D,X_test2D=bd.reshape2D(X_train,X_test)

# %%
idx=1
for image in X_test2D:
    img = image.transpose()
    imshow(img)
    fname = './stim/69/test/'+str(idx)+'.png'
    imsave(fname,img)
    idx += 1

idx=1
for image in X_train2D:
    img = image.transpose()
    imshow(img)
    fname = './stim/69/train/'+str(idx)+'.png'
    imsave(fname,img)
    idx += 1

# %%
