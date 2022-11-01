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
1. omniglot-trainig
2. omniglot-evaluation
3. omniglot-result

### one shoot