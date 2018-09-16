import random
import argparse
import torch
import numpy as np
from datautil import prepare_mnli_data_loader

def main(arguments):
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed(arguments.seed)
    torch.cuda.set_device(arguments.gpu_id)

    iters = prepare_mnli_data_loader(arguments, reversed=True)  # TODO false/true
    multinli_train_iter, multinli_dev_match_iter, multinli_dev_umatch_iter, multinli_test_match_iter, multinli_test_umatch_iter = iters

    for batch_i, batch in enumerate(multinli_train_iter):
        s1, s1_l = batch.premise
        s2, s2_l = batch.hypothesis
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI training')
    # paths options
    parser.add_argument("--nli_dataset", type=str, default="mnli")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    parser.add_argument("--embedding_file", type=str, default="saved_embd_new.pt")
    # learning options
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--batch_size_eval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--mlp_dim", type=int, default=800)
    # gpu options
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)
