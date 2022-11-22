# siabre
siameses network fro brain decoding

## For Omniglot Training
image input : 20 image per character. every class of script have many character like alfabet. e.g.
sansscript -> character01 -> 0709_01.png ..... 0709_20.png

input training : 
X:(964,20,105,105)
964 : total alfabetical in all type of script, getting from c
20 : total handwriting pictures in a folder for same character
105x105 : image dimension pixel

c:dict 30 : 0,19 ... 909,963
30 : total folder of type of script
0,19 ... 909,963 : start-end(alfabetical in a type script)

use lib/omniglot
### omniglot-prepro
load image folder and save to pickle 
### omniglot-trainig
* load pickle. 
* get siamese model, set optimizer and loss and then compile model.
* do for loop for n_iter
  * get_batch to fill : inputs(2,32,105,105,1)2 for left and right input nets,targets(32)0 for first 16 not same figure and 1 for next 16 with same figure.
  * get loss from model.train_on_batch(inputs, targets)
  * get validation accuracy with test one shoot using data_val (different from train_on_batch)
    * N_way : how many picture in one char folder
    * n__val : how many one shoot task
  * model.save_weights
  * save best of val accuration
### omniglot-evaluation
ok
### omniglot-result
ok

### one shoot