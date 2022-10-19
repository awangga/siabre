from lib.omniglot import make_oneshot_task,plot_oneshot_task
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

save_path = './data/'

with open(os.path.join(save_path, "accuracies.pickle"), "rb") as f:
    (val_accs, train_accs, nn_accs) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

ways = np.arange(1,20,2)

pairs, targets = make_oneshot_task(16,Xtrain,Xval,train_classes,val_classes,"train","Sanskrit")
plot_oneshot_task(pairs)

fig,ax = plt.subplots(1)
ax.plot(ways, val_accs, "m", label="Siamese(val set)")
ax.plot(ways, train_accs, "y", label="Siamese(train set)")
plt.plot(ways, nn_accs, label="Nearest neighbour")

ax.plot(ways, 100.0/ways, "g", label="Random guessing")
plt.xlabel("Number of possible classes in one-shot tasks")
plt.ylabel("% Accuracy")
plt.title("Omiglot One-Shot Learning Performance of a Siamese Network")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
inputs,targets = make_oneshot_task(20,Xtrain,Xval,train_classes,val_classes, "val", 'Oriya')
plt.show()

plot_oneshot_task(inputs)