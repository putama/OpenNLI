import random
import argparse
import torch
import numpy as np
import torch.nn as nn
from datautil import build_nli_iterator
from models.residual_encoder import ResEncoder
from models.trainer import build_optimizer, adjust_learning_rate
from tqdm import tqdm

def main(arguments):
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed(arguments.seed)
    torch.cuda.set_device(arguments.gpu_id)

    iters = build_nli_iterator(arguments, reverse=False)
    multinli_train_iter = iters[0]
    multinli_dev_match_iter, multinli_dev_umatch_iter = iters[1], iters[2]

    print("initiate NLI model...")

    if not "vocab_size" in arguments:
        raise Exception("vocab size has not been determined")

    multinli_model = to_cuda(ResEncoder(arguments))
    multinli_model.load_pretrained_emb(arguments)
    optimizer = build_optimizer(multinli_model, arguments)
    multinli_model.train()

    lr = arguments.learning_rate
    for epoch in range(arguments.epoch):
        decay_i = epoch // 2
        lr = lr / (2 ** decay_i)
        adjust_learning_rate(optimizer, lr)

        multinli_train_iter.init_epoch()
        trainbar = tqdm(multinli_train_iter)
        for batch_i, batch in enumerate(trainbar):
            s1, s1_len = batch.premise
            s2, s2_len = batch.hypothesis
            target_y = batch.label
            s1, s1_len, s2, s2_len, target_y = to_cuda(s1, s1_len, s2, s2_len, target_y)

            logit = multinli_model(s1, s1_len, s2, s2_len)
            loss = multinli_model.compute_loss(logit, target_y)
            acc = (torch.max(logit, 1)[1] == target_y).sum().item() / float(len(batch))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_i+1) % arguments.eval_step == 0:
                avg_acc, avg_loss = eval(multinli_model, multinli_dev_match_iter)
                print("training validation. step-%d. average acc: %.3f. average loss: %.3f" %
                      (batch_i+1, avg_acc, avg_loss))
                multinli_model.train()

            trainbar.set_description("Training current acc: %.3f, loss: %.3f" % (acc, loss.item()))


def eval(nli_model: nn.Module, data_iter):
    print("run evaluation...")
    nli_model.eval()

    correct_total = 0
    loss_total = 0.
    data_n = len(data_iter.dataset)
    batch_n = len(data_iter)
    for batch_i, batch in enumerate(data_iter):
        s1, s1_len = batch.premise
        s2, s2_len = batch.hypothesis
        target_y = batch.label
        s1, s1_len, s2, s2_len, target_y = to_cuda(s1, s1_len, s2, s2_len, target_y)

        logit = nli_model(s1, s1_len, s2, s2_len)
        loss = nli_model.compute_loss(logit, target_y)

        correct_n = (torch.max(logit, 1)[1] == target_y).sum().item()
        correct_total += correct_n
        loss_total += loss.item()

    avg_acc = correct_total / float(data_n)
    avg_loss = loss_total / batch_n

    return avg_acc, avg_loss


def to_cuda(*objects):
    if torch.cuda.is_available():
        if len(objects) == 1:
            return objects[0].cuda()
        return [obj.cuda() for obj in objects]
    else:
        if len(objects) == 1:
            return objects[0]
        else:
            return objects


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI training')
    # paths options
    parser.add_argument("--nli_dataset", type=str, default="multinli")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    parser.add_argument("--embedding_pt_file", type=str, default="extracted_glove.pt")
    # learning options
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--batch_size_eval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--mlp_dim", type=int, default=800)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--eval_step", type=int, default=4000)
    parser.add_argument("--use_pretrained_emb", type=int, default=1)
    # data options
    parser.add_argument("--min_freq", type=int, default=10)
    # gpu options
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)