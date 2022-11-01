import os
import time
from tensorflow.keras.optimizers import Adam
from lib.omniglot import loadPickle,get_siamese_model,get_batch,test_oneshot
from config import save_path,trainfname,valfname,model_path


Xtrain, train_classes=loadPickle(save_path,trainfname)
Xval, val_classes=loadPickle(save_path,valfname)

print("Training alphabets: \n")
print(list(train_classes.keys()))
print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))
############################################
model = get_siamese_model((105, 105, 1))
model.summary()
from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='model.png')
optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)
# Hyper parameters
evaluate_every = 200 # interval for evaluating on one-shot tasks
batch_size = 32
n_iter = 20000 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks, banyak gambar per karakter
n_val = 250 # how many one-shot tasks to validate on
best = -1
print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    (inputs,targets) = get_batch(batch_size,Xtrain,Xval,train_classes,val_classes,)
    print('input : ')
    print(inputs)
    print('output : ')
    print(targets)
    print('input : '+str(len(inputs))+' output : '+str(len(targets)))
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(model, Xtrain,Xval,train_classes,val_classes,N_way, n_val, verbose=True)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc



