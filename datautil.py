from torchtext import data, datasets


def build_mnli_split(arguments, reverse=True):
    text_field = datasets.nli.ParsedTextField(reverse=reverse)
    label_field = data.LabelField()
    genre_field = data.LabelField()
    parse_field = datasets.nli.ShiftReduceField()

    print("splitting MultiNLI datasets...")
    splits_match = datasets.MultiNLI.splits(text_field, label_field, parse_field, genre_field,
                                            root=arguments.data_root,
                                            train="multinli_1.0_train.jsonl",
                                            validation="multinli_1.0_dev_matched.jsonl",
                                            test="multinli_0.9_test_"
                                                 "matched_unlabeled.jsonl")
    multinli_train, multinli_dev_match, multinli_test_match = splits_match
    splits_umatch = datasets.MultiNLI.splits(text_field, label_field, parse_field,
                                             genre_field, root=arguments.data_root,
                                             train=None,
                                             validation="multinli_1.0_"
                                                        "dev_mismatched.jsonl",
                                             test="multinli_0.9_test_"
                                                  "mismatched_unlabeled.jsonl")
    multinli_dev_umatch, multinli_test_umatch = splits_umatch

    print("building vocabularies from each sets...")
    text_field.build_vocab(multinli_train, multinli_dev_match, multinli_dev_umatch,
                           multinli_test_match, multinli_test_umatch,
                           min_freq=arguments.min_freq)
    label_field.build_vocab(multinli_train)
    genre_field.build_vocab(multinli_train, multinli_dev_match, multinli_dev_umatch,
                           multinli_test_match, multinli_test_umatch)
    print("Target labels:", label_field.vocab.itos)
    print("Available genres:", genre_field.vocab.itos)

    arguments.vocab_size = len(text_field.vocab)

    return (multinli_train, multinli_dev_match, multinli_test_match,
            multinli_dev_umatch, multinli_test_umatch)

def build_nli_iterator(args, reverse=True):
    if args.nli_dataset == "multinli":
        train, dev_match, test_match, dev_umatch, test_umatch = build_mnli_split(args,
                                                                                 reverse)
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