train_folder = "./images_background/"
val_folder = './images_evaluation/'
save_path = './data/'
trainfname = 'train.pickle'
valfname = 'val.pickle'
model_path = 'R:\\awangga\w'

# Hyper parameters
lr = 0.00006
evaluate_every = 200 # interval for evaluating on one-shot tasks
batch_size = 32
n_iter = 20000 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks, banyak gambar per karakter x.
n_val = 250 # how many one-shot tasks to validate on
best = -1