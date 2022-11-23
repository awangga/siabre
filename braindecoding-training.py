import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from lib.braindecoding import loadPickle,get_siamese_model,get_batch,test_oneshot
from config import save_path,trainfname,valfname,model_path,lr,evaluate_every,batch_size,n_iter,N_way,n_val,best

Xtrain, train_classes=loadPickle(save_path,trainfname)
Xval, val_classes=loadPickle(save_path,valfname)


model = get_siamese_model((105, 105, 1))# input pixel
optimizer = Adam(learning_rate = lr)
model.compile(loss="binary_crossentropy",optimizer=optimizer)
model.summary()


for i in range(1, n_iter+1):# No. of training iterations 20000
    (inputs,targets) = get_batch(batch_size,Xtrain,Xval,train_classes,val_classes,)
    loss = model.train_on_batch(inputs, targets)
    print(loss)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(model, Xtrain,Xval,train_classes,val_classes,N_way, n_val, verbose=True)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc
