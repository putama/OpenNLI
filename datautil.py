from torchtext import data, datasets
import torch

def build_mnli_split(arguments, reverse=True):
    text_field = datasets.nli.ParsedTextField(reverse=reverse)
    label_field = data.LabelField()
    genre_field = data.LabelField()
    parse_field = datasets.nli.ShiftReduceField()
    pairID_field = data.Field(use_vocab=False, sequential=False, unk_token=None)

    print("splitting MultiNLI datasets...")
    splits = datasets.MultiNLI.splits(text_field, label_field,
                                      parse_field, genre_field,
                                      root=arguments.data_root,
                                      train="multinli_1.0_train.jsonl",
                                      validation="multinli_1.0_dev_matched.jsonl",
                                      test="multinli_1.0_dev_mismatched.jsonl")
    train_split, dev_m_split, dev_um_split = splits
    splits = datasets.MultiNLI.splits(text_field, label_field, parse_field, 
                                      genre_field, pairID_field, 
                                      root=arguments.data_root,
                                      train=None,
                                      validation="multinli_0.9_test_"
                                                 "matched_unlabeled.jsonl",
                                      test="multinli_0.9_test_"
                                           "mismatched_unlabeled.jsonl")
    test_m_split, test_um_split = splits

    print("building vocabularies from each sets...")
    text_field.build_vocab(train_split, dev_m_split, dev_um_split,
                           test_m_split, test_um_split,
                           min_freq=arguments.min_freq)
    label_field.build_vocab(train_split)
    genre_field.build_vocab(train_split, dev_m_split, dev_um_split,
                            test_m_split, test_um_split)
    print("Target labels:", label_field.vocab.itos)
    print("Available genres:", genre_field.vocab.itos)

    arguments.vocab_size = len(text_field.vocab)

    return train_split, dev_m_split, test_m_split, dev_um_split, test_um_split


def build_nli_iterator(split, args, training=True):
    """
    :param split: Dataset
    :param args: arguments
    :param training: is training or not
    :return: iterator of dataset
    """

    if training:
        data_iters = data.Iterator.splits((split,), batch_size=(args.batch_size),
                                          repeat=False)
    else:
        data_iters = data.Iterator.splits((split,), shuffle=False,
                                          batch_size=args.batch_size,
                                          repeat=False)
    return data_iters[0]


def build_nli_iterator_all(splits, args):
    if args.nli_dataset == "multinli":
        train, dev_match, test_match, dev_umatch, test_umatch = splits
    else:
        raise Exception("invalid NLI dataset name")

    (multinli_train_iter,) = data.Iterator.splits((train,),
                                                  batch_size=(args.batch_size),
                                                  repeat=False)

    match_iters = data.Iterator.splits((dev_match, test_match), shuffle=False,
                                       batch_size=args.batch_size_eval,
                                       repeat=False)
    multinli_dev_match_iter, multinli_test_match_iter = match_iters

    umatch_iters = data.Iterator.splits((dev_umatch, test_umatch), shuffle=False,
                                        batch_size=args.batch_size_eval,
                                        repeat=False)
    multinli_dev_umatch_iter, multinli_test_umatch_iter = umatch_iters

    iters = (multinli_train_iter, multinli_dev_match_iter, multinli_dev_umatch_iter,
             multinli_test_match_iter, multinli_test_umatch_iter)
    return iters