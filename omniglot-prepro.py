from config import train_folder,val_folder,save_path,trainfname,valfname
from lib.omniglot import ImgToPicke

X,y,c=ImgToPicke(train_folder,save_path,trainfname)
Xval,yval,cval=ImgToPicke(val_folder,save_path,valfname)
