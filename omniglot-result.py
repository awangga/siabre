from lib.omniglot import make_oneshot_task,plot_oneshot_task
import pickle


save_path = './data/'

with open(os.path.join(save_path, "accuracies.pickle"), "rb") as f:
    (val_accs, train_accs, nn_accs) = pickle.load(f)

pairs, targets = make_oneshot_task(16,"train","Sanskrit")
plot_oneshot_task(pairs)