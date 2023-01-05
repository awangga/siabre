from lib import braindecoding as bd
from matplotlib.pyplot import imshow,imsave
import numpy as np

X_train,X_test,Y_train,Y_test = bd.getDataset69('./data/digit69_28x28.mat')


Y_train,Y_test = bd.normalizefMRI(Y_train,Y_test)
X_train2D,X_test2D=bd.reshape2D(X_train,X_test)

import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from lib.braindecoding import get_siamese_model
from config import save_path,trainfname,valfname,model_path,lr,evaluate_every,batch_size,n_iter,N_way,n_val,best

model = get_siamese_model((3092))# input pixel
optimizer = Adam(learning_rate = lr)
model.compile(loss="binary_crossentropy",optimizer=optimizer)
model.summary()

#  * get_batch to fill : inputs(2,32,105,105,1)2 for left and right input nets,targets(32)0 for first 16 not same figure and 1 for next 16 with same figure.
#targets
stimtrain6 = X_train[:45,:]
stimtrain61 = stimtrain6[:22,:]
stimtrain62 = stimtrain6[23:,:]

stimtrain9 = X_train[45:,:]
stimtrain91 = stimtrain9[:22,:]
stimtrain92 = stimtrain9[23:,:]

targeta=np.append(stimtrain61,stimtrain91,axis = 0)
targetb=np.append(stimtrain61,stimtrain91,axis = 0)
targets=targetb
#inputs
mritrain6 = Y_train[:45,:]
mritrain61 = mritrain6[:22,:]
mritrain62 = mritrain6[23:,:]

mritrain9 = Y_train[45:,:]
mritrain91 = mritrain9[:22,:]
mritrain92 = mritrain9[23:,:]


inputa= np.append(mritrain61,mritrain91,axis = 0)
inputb= np.append(mritrain62,mritrain92,axis = 0)
inputs=[inputa,inputb]



def test_oneshot(model, inputs, targets, n_val, s = "val", verbose = 0):#tambah 4 parameter sesudah model
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0 #berapa yang bener nya
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(n_val,N_way))
    for i in range(n_val):
        #inputs, targets = make_oneshot_task(N,Xtrain,Xval,train_classes,val_classes,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1#klo bener tambah 1
    percent_correct = (100.0 * n_correct / n_val) # yang bener berapa/jumlah ujicoba
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N_way))
    return percent_correct

t_start = time.time()
for i in range(1, n_iter+1):# No. of training iterations 20000
    #(inputs,targets) = get_batch(batch_size,Xtrain,Xval,train_classes,val_classes,)
    loss = model.train_on_batch(inputs, targets)
    print(loss)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(model, inputs, targets, n_val, verbose=True)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc

