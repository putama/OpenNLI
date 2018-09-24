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


def custom_sort_key(ex):
    return max(len(ex.premise), len(ex.hypothesis))


def build_nli_iterator(split, args, training=True, bucket=True):
    """
    :param split: Dataset
    :param args: arguments
    :param training: is training or not
    :return: iterator of dataset
    """
    print("build nli iterator for the specified sets...")
    iteratorCls = data.BucketIterator if bucket else data.Iterator

    if training:
        data_iters = iteratorCls.splits((split,), batch_size=(args.batch_size),
                                        repeat=False, sort_within_batch=True,
                                        sort_key=custom_sort_key)
    else:
        data_iters = iteratorCls.splits((split,), shuffle=False,
                                        batch_size=args.batch_size, repeat=False,
                                        sort_within_batch=True, sort_key=custom_sort_key
                                        )
    return data_iters[0]


def build_nli_iterator_all(splits, args, bucket=True):
    print("build nli iterator for all sets...")
    if args.nli_dataset == "multinli":
        train, dev_match, test_match, dev_umatch, test_umatch = splits
    else:
        raise Exception("invalid NLI dataset name")

    iteratorCls = data.BucketIterator if bucket else data.Iterator

    (multinli_train_iter,) = iteratorCls.splits((train,),
                                                batch_size=(args.batch_size),
                                                sort_within_batch=True,
                                                repeat=False, sort_key=custom_sort_key)

    match_iters = iteratorCls.splits((dev_match, test_match), shuffle=False,
                                     batch_size=args.batch_size_eval, repeat=False,
                                     sort_within_batch=True, sort_key=custom_sort_key)
    multinli_dev_match_iter, multinli_test_match_iter = match_iters

    umatch_iters = iteratorCls.splits((dev_umatch, test_umatch), shuffle=False,
                                      batch_size=args.batch_size_eval, repeat=False,
                                      sort_within_batch=True, sort_key=custom_sort_key)
    multinli_dev_umatch_iter, multinli_test_umatch_iter = umatch_iters

    iters = (multinli_train_iter, multinli_dev_match_iter, multinli_dev_umatch_iter,
             multinli_test_match_iter, multinli_test_umatch_iter)
    return iters
