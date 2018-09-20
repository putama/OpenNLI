import argparse
import random
import numpy as np
import torch
from torch import nn as nn
import os

from datautil import build_mnli_split, build_nli_iterator
from utilities import to_cuda
from models.residual_encoder import ResidualEncoder
from tqdm import tqdm


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


def predict(nli_model: nn.Module, data_iter,
            label_field=None, prepare_submission=True):
    print("run evaluation...")
    nli_model.eval()

    if prepare_submission and label_field is None:
        raise Exception("label field should be provided for submission")

    data_iter.init_epoch()
    for batch_i, batch in enumerate(tqdm(data_iter)):
        s1, s1_len = batch.premise
        s2, s2_len = batch.hypothesis
        target_y = batch.label
        s1, s1_len, s2, s2_len, target_y = to_cuda(s1, s1_len, s2, s2_len, target_y)

        logit = nli_model(s1, s1_len, s2, s2_len)
        predictions = list(logit.max(dim=1)[1].cpu().numpy())
        pairIDs = list(batch.pairID.cpu().numpy())
        for id, pred in zip(pairIDs, predictions):
            yield (id, label_field.vocab.itos[pred])



def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)

    splits = build_mnli_split(args, reverse=False)
    if args.dev:
        if args.missmatch:
            split = splits[3]
        else:
            split = splits[1]
    else:
        if args.missmatch:
            split = splits[4]
        else:
            split = splits[2]
    data_iter = build_nli_iterator(split, args, training=False)

    checkpoint = torch.load(arguments.checkpoint_path,
                            map_location=lambda storage, loc: storage)
    state_dict = checkpoint["state_dict"]
    train_args = checkpoint["train_args"]
    nli_model = ResidualEncoder(train_args)
    nli_model.load_state_dict(state_dict)
    nli_model = to_cuda(nli_model)

    data_iter.init_epoch()
    label_field = data_iter.dataset.fields['label']

    if args.dev:
        avg_acc, avg_loss = eval(nli_model, data_iter, label_field=label_field)
    else: # run test and write predictions
        pred_file_path = os.path.join(args.data_root,
                                      args.nli_dataset,
                                      args.submission_file)
        with open(pred_file_path, 'w') as pred_file:
            pred_file.write("pairID,gold_label")
            for id, pred in predict(nli_model, data_iter, label_field=label_field):
                pred_file.write("{},{}\n".format(id, pred))

    print("evaluation finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI training')
    # evaluation options

    parser.add_argument("--nli_dataset", type=str, default="multinli")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--missmatch", action="store_true")
    # paths options
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/best_model.pt.tar")
    parser.add_argument("--submission_file", type=str, default="predictions.txt")
    # data options
    parser.add_argument("--min_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    # gpu options
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)