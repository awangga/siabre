import os
from lib.omniglot import get_siamese_model,test_oneshot,test_nn_accuracy
import numpy as np
import pickle

model = get_siamese_model((105, 105, 1))
model.summary()

save_path = './data/'
model_path = 'R:\\awangga\w'
model.load_weights(os.path.join(model_path, "weights.20000.h5"))

ways = np.arange(1,20,2)
resume =  False
trials = 50

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)


val_accs, train_accs,nn_accs = [], [], []
for N in ways:    
    val_accs.append(test_oneshot(model, Xtrain,Xval,train_classes,val_classes,N, trials, s="val", verbose=True))
    train_accs.append(test_oneshot(model, Xtrain,Xval,train_classes,val_classes,N, trials, s="train", verbose=True))
    nn_acc = test_nn_accuracy(N, trials)
    nn_accs.append(nn_acc)
    print ("NN Accuracy = ", nn_acc)
    print("---------------------------------------------------------------------------------------------------------------")