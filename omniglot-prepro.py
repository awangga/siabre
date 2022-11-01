from config import train_folder,val_folder,save_path,trainfname,valfname
from lib.omniglot import ImgToPicke,loadPickle

X,y,c=ImgToPicke(train_folder,save_path,trainfname)
Xval,yval,cval=ImgToPicke(val_folder,save_path,valfname)

input('press to continue load')

Xtrain, train_classes=loadPickle(save_path,trainfname)
Xval, val_classes=loadPickle(save_path,valfname)

print("Training alphabets: \n")
print(list(train_classes.keys()))
print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))