import argparse
import random
import numpy as np
import torch
from torch import nn as nn
import os
import torch.nn.functional as F

from opennli.data.datautil import build_mnli_split, build_nli_iterator
from opennli.utilities.utilities import to_cuda, ids2words, labels2prob, probs2entropy

from opennli.models.residual_encoder import ResidualEncoder
from opennli.models.infersent import InferSent
from opennli.models.decomposable_attn import DecompAttention

from tqdm import tqdm

def eval(nli_model: nn.Module, data_iter, calibration_error=False):
    print("run evaluation...")
    nli_model.eval()

    if calibration_error:
        m = (1/15)
        bucket_size = int(1.0 / m)
        bucket_correct = np.zeros(bucket_size, dtype=int).tolist()
        bucket_conf = np.zeros(bucket_size, dtype=int).tolist()
        bucket_counter = np.zeros(bucket_size, dtype=int).tolist()

    correct_total = 0
    loss_total = 0.
    data_n = len(data_iter.dataset)
    batch_n = len(data_iter)

    correct_entropy_tot = 0.0
    misclassified_entropy_tot = 0.0
    for batch_i, batch in enumerate(tqdm(data_iter)):
        s1, s1_len = batch.premise
        s2, s2_len = batch.hypothesis
        target_y = batch.label
        s1, s1_len, s2, s2_len, target_y = to_cuda(s1, s1_len, s2, s2_len, target_y)

        class_scores = nli_model(s1, s1_len, s2, s2_len)
        loss = nli_model.compute_loss(class_scores, target_y)

        correct_n = (torch.max(class_scores, 1)[1] == target_y).sum().item()
        correct_total += correct_n
        loss_total += loss.item()

        # evaluate disagreement
        correct_i = (torch.max(class_scores, 1)[1] == target_y).nonzero().squeeze(1)
        incorrect_i = ((torch.max(class_scores, 1)[1] == target_y)==0).nonzero().squeeze(1)

        misclassified_labels = batch.label_list.transpose(1, 0).cuda().index_select(0, incorrect_i)
        correct_labels = batch.label_list.transpose(1, 0).cuda().index_select(0, correct_i)
        misclassified_entropy = probs2entropy(labels2prob(misclassified_labels))
        correct_entropy = probs2entropy(labels2prob(correct_labels))
        misclassified_entropy_tot += sum(misclassified_entropy)/len(misclassified_entropy)
        correct_entropy_tot += sum(correct_entropy)/len(correct_entropy)

        # print(batch.label_list.transpose(1, 0).cuda().index_select(0, incorrect_i))
        # print(F.softmax(class_scores, dim=1).index_select(0, incorrect_i))

        # vocab = data_iter.dataset.fields['premise'].vocab
        # sentpairs = ids2words(vocab, s1.index_select(1, incorrect_i), s2.index_select(1, incorrect_i))
        # for sentpair in sentpairs:
        #     print(sentpair)
        # print("----")

        if calibration_error:
            maxres = torch.max(F.softmax(class_scores, dim=1), 1)
            maxconf = maxres[0].data.cpu().tolist()
            maxcorrect = (maxres[1] == target_y).cpu().tolist()
            for conf, corr in zip(maxconf, maxcorrect):
                bucket_idx = int(conf // m)
                bucket_conf[bucket_idx] += conf
                bucket_correct[bucket_idx] += corr
                bucket_counter[bucket_idx] += 1

    if calibration_error:
        confs = [i/j if j > 0 else 0 for i,j in zip(bucket_conf, bucket_counter)]
        accs = [i/j if j > 0 else 0 for i,j in zip(bucket_correct, bucket_counter)]
        gaps = [abs(i-j) for i, j in zip(confs, accs)]
        ECE = sum(map(lambda gap_count: (gap_count[1] / data_n) * gap_count[0],
                      zip(gaps, bucket_counter)))
        print("counts:\n", "\n".join(("%d" % count) for count in bucket_counter), sep="")
        print("confidence:\n","\n".join(("%.3f" % conf) for conf in confs), sep="")
        print("accuracies:\n", "\n".join(("%.3f" % acc) for acc in accs), sep="")
        print("ECE:", "%.3f" % (ECE * 100))

        print("Avg entropy of correct predictions:", (correct_entropy_tot / batch_n))
        print("Avg entropy of incorrect predictions:", (misclassified_entropy_tot / batch_n))

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

        class_scores = nli_model(s1, s1_len, s2, s2_len)
        predictions = list(class_scores.max(dim=1)[1].cpu().numpy())
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
        if args.mismatch:
            split = splits[3]
        else:
            split = splits[1]
    else:
        if args.mismatch:
            split = splits[4]
        else:
            split = splits[2]
    data_iter = build_nli_iterator(split, args, training=False)

    checkpoint = torch.load(arguments.checkpoint_path,
                            map_location=lambda storage, loc: storage)
    state_dict = checkpoint["state_dict"]
    train_args = checkpoint["train_args"]

    if args.model == "stacked":
        nli_model = ResidualEncoder(train_args)
    elif args.model == "infersent":
        nli_model = InferSent(train_args)
    elif args.model == "decomposable":
        nli_model = DecompAttention(train_args)
    else:
        raise Exception("invalid model")

    nli_model.load_state_dict(state_dict)
    nli_model = to_cuda(nli_model)

    data_iter.init_epoch()
    label_field = data_iter.dataset.fields['label']

    if args.dev:
        avg_acc, avg_loss = eval(nli_model, data_iter, calibration_error=args.calibration)
        print("training validation. "
              "average acc: %.3f. average loss: %.3f" %
              (avg_acc, avg_loss))
    else: # run test and write predictions
        pred_file_path = os.path.join(args.data_root,
                                      args.nli_dataset,
                                      args.submission_file)
        with open(pred_file_path, 'w') as pred_file:
            pred_file.write("pairID,gold_label\n")
            for id, pred in predict(nli_model, data_iter, label_field=label_field):
                pred_file.write("{},{}\n".format(id, pred))

    print("evaluation finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI training')
    # evaluation options
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--nli_dataset", type=str, default="multinli")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    # paths options
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/best_model.pt.tar")
    parser.add_argument("--submission_file", type=str, default="predictions.txt")
    # data options
    parser.add_argument("--min_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    # gpu options
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)