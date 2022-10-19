from lib.omniglot import make_oneshot_task,plot_oneshot_task
import pickle
import os

save_path = './data/'

with open(os.path.join(save_path, "accuracies.pickle"), "rb") as f:
    (val_accs, train_accs, nn_accs) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)
    
pairs, targets = make_oneshot_task(16,Xtrain,Xval,train_classes,val_classes,"train","Sanskrit")
plot_oneshot_task(pairs)