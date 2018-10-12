import torch
import tqdm
import argparse
import numpy as np
from opennli.data.datautil import build_mnli_split
from tqdm import tqdm
import os


def read_and_match_embeddings(vocab, arguments):
    word2idx = vocab.stoi
    embs_matrix = np.random.uniform(-.1, .1, (len(word2idx), arguments.embedding_dim))
    match_counter = 0
    line_counter = 0

    print("iterate through {}".format(arguments.embedding_txt_file))
    with tqdm(total=len(word2idx)) as tbar:
        for line in open(arguments.embedding_txt_file, "rb"):
            if not line: # very last line
                print("not line")
                break
            if match_counter == len(word2idx):
                break
                print("all words")

            line_split = line.decode('utf8').strip().split(' ')
            word = line_split[0]
            if word in word2idx:
                match_counter += 1
                embs_matrix[word2idx[word]] = [float(em) for em in line_split[1:]]
                tbar.update()

            line_counter += 1

    print("{} from {} words found matched.".format(match_counter, line_counter))
    return torch.Tensor(embs_matrix)


def main(arguments):
    print("split and build vocab from entire sets")
    splits = build_mnli_split(arguments, reverse=False)

    # all splits share the same text field
    if len(splits) > 0:
        split = splits[0]
        vocab = split.fields['premise'].vocab
    else:
        raise Exception()

    all_emb = read_and_match_embeddings(vocab, arguments)
    save_path =  os.path.join(arguments.data_root, arguments.nli_dataset, 'extracted_emb.pt.tar')
    torch.save(all_emb, save_path)
    print("extracted embedding is saved at", save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLI training')
    # paths options
    parser.add_argument("--nli_dataset", type=str, default="multinli")
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--save_dir", type=str, default="../../data/glove")
    parser.add_argument("--embedding_txt_file", type=str,
                        default="../../data/glove/glove.840B.300d.txt")
    parser.add_argument("--embedding_dim", type=int, default=300)
    # gpu options
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # data options
    parser.add_argument("--min_freq", type=int, default=10)
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)