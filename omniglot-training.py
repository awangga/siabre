import os
import time
import pickle
from tensorflow.keras.optimizers import Adam
from lib.omniglot import loadimgs,initialize_weights,initialize_bias,get_siamese_model,get_batch,make_oneshot_task,test_oneshot

train_folder = "./images_background/"
val_folder = './images_evaluation/'
save_path = './data/'

X,y,c=loadimgs(train_folder)

with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump((X,c),f)


Xval,yval,cval=loadimgs(val_folder)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((Xval,cval),f)

model = get_siamese_model((105, 105, 1))
model.summary()
from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='model.png')

optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)
##########################
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)
    
print("Training alphabets: \n")
print(list(train_classes.keys()))
######################
with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))


# Hyper parameters
evaluate_every = 200 # interval for evaluating on one-shot tasks
batch_size = 32
n_iter = 20000 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks
n_val = 250 # how many one-shot tasks to validate on
best = -1

model_path = 'R:\\awangga\w'

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    (inputs,targets) = get_batch(batch_size,Xtrain,Xval,train_classes,val_classes,)
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



