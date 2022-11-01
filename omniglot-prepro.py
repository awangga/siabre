import os
import time
import pickle
from tensorflow.keras.optimizers import Adam
from lib.omniglot import loadimgs,initialize_weights,initialize_bias,get_siamese_model,get_batch,make_oneshot_task,test_oneshot

train_folder = "./images_background/"
val_folder = './images_evaluation/'
save_path = './data/'

X,y,c=loadimgs(train_folder) #c=>lang_dict atau class
Xval,yval,cval=loadimgs(val_folder)

with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    print('X : '+X+' c : '+c)
    pickle.dump((X,c),f)

with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((Xval,cval),f)
    print('X : '+Xval+' c : '+cval)

input('press to continue load')
#####################
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)
    
print("Training alphabets: \n")
print(list(train_classes.keys()))
######################
with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))