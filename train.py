import random
import argparse
import torch
import numpy as np
from datautil import build_mnli_split, build_nli_iterator_all
from evaluate import eval
from models.residual_encoder import ResidualEncoder
from models.infersent import InferSent
from models.trainer import build_optimizer, adjust_learning_rate
from tqdm import tqdm
from datetime import datetime
import os
from utilities import to_cuda


def main(arguments):
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed(arguments.seed)
    torch.cuda.set_device(arguments.gpu_id)

    splits = build_mnli_split(arguments, reverse=False)
    iters = build_nli_iterator_all(splits, arguments)
    multinli_train_iter = iters[0]
    multinli_dev_match_iter, multinli_dev_umatch_iter = iters[1], iters[2]

    print("initiate NLI model...")
    if not "vocab_size" in arguments:
        raise Exception("vocab size has not been determined")

    if arguments.model == "stacked":
        multinli_model = ResidualEncoder(arguments)
    elif arguments.model == "infersent":
        multinli_model = InferSent(arguments)
    else:
        raise Exception("invalid model")

    multinli_model.load_pretrained_emb(arguments)
    multinli_model = to_cuda(multinli_model)
    multinli_model.display()

    optimizer = build_optimizer(multinli_model, arguments)
    multinli_model.train()

    # prepare to save model
    save_dir = datetime.now().strftime("experiment_D%d-%m_H%H-%M")
    os.mkdir(os.path.join(arguments.checkpoint_dir, save_dir))

    step_i = 0
    for epoch in range(arguments.epoch):
        decay_i = epoch // arguments.decay_every
        lr = arguments.learning_rate * (arguments.decay_rate ** decay_i)
        adjust_learning_rate(optimizer, lr)
        print("learning rate is decayed to:", lr)

        multinli_train_iter.init_epoch()
        trainbar = tqdm(multinli_train_iter)
        for batch_i, batch in enumerate(trainbar):
            step_i += 1
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

            if step_i % arguments.eval_step == 0:
                avg_acc, avg_loss = eval(multinli_model, multinli_dev_match_iter)
                print("training validation. step-%d. "
                      "average acc: %.3f. average loss: %.3f" %
                      (batch_i + 1, avg_acc, avg_loss))
                multinli_model.train()
                # save current model to ckpt file
                save_file = "%s_%s_model_epoch_%d_step_%d_acc_%.3f.pt.tar" % \
                            (arguments.model, arguments.nli_dataset,
                             (epoch + 1), (batch_i + 1), avg_acc)
                save_path = os.path.join(arguments.checkpoint_dir,
                                         save_dir,
                                         save_file)
                print("saving the model to checkpoint file", save_path)
                torch.save({"state_dict": multinli_model.state_dict(),
                            "train_args": arguments},
                           save_path)

            trainbar.set_description("Epoch-%d, current acc: %.3f, loss: %.3f" %
                                     ((epoch+1), acc, loss.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI training')
    # paths options
    parser.add_argument("--nli_dataset", type=str, default="multinli")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--embedding_pt_file", type=str, default="extracted_glove.pt.tar")
    # learning options
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument("--fc_dim", type=int, default=800)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--eval_step", type=int, default=6000)
    parser.add_argument("--use_pretrained_emb", type=int, default=1)
    parser.add_argument("--decay_every", type=int, default=5)
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    # residual encoder options
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument('--lstm_dims', nargs='+', type=int,
                        default=[512, 1024, 2048])
    # infersent options
    parser.add_argument("--nonlinear_fc", type=int, default=0)
    parser.add_argument("--lstm_dim", type=int, default=2048)
    parser.add_argument("--lstm_dropout_rate", type=float, default=0.)
    parser.add_argument("--pool_type", type=str, default="max")

    # data options
    parser.add_argument("--max_seq_length", type=int, default=60)
    parser.add_argument("--min_freq", type=int, default=10)
    # gpu options
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)
