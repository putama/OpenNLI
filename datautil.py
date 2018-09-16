import torchtext
import custom_nli_datasets as datasets
from torchtext import data

def prepare_mnli_data_loader(arguments, reverse=True):
    text_field = datasets.ParsedTextField(reverse=reverse)
    label_field = data.Field(unk_token=None, sequential=False)
    genre_field = data.Field(unk_token=None, sequential=False)
    parse_field = datasets.ShiftReduceField()

    print("splitting MultiNLI datasets...")
    splits_match = datasets.MultiNLI.splits(text_field, label_field, parse_field, genre_field,
                                            root=arguments.data_root,
                                            train="multinli_1.0_dev_matched.jsonl",
                                            validation="multinli_1.0_dev_matched.jsonl",
                                            test="multinli_0.9_test_matched_unlabeled.jsonl")
    multinli_train, multinli_dev_match, multinli_test_match = splits_match
    splits_umatch = datasets.MultiNLI.splits(text_field, label_field, parse_field, genre_field,
                                             root=arguments.data_root, train=None,
                                             validation="multinli_1.0_dev_mismatched.jsonl",
                                             test="multinli_0.9_test_mismatched_unlabeled.jsonl")
    multinli_dev_umatch, multinli_test_umatch = splits_umatch

    print("building vocabularies from each sets...")
    text_field.build_vocab(multinli_train, multinli_dev_match, multinli_dev_umatch,
                           multinli_test_match, multinli_test_umatch)
    label_field.build_vocab(multinli_train)
    genre_field.build_vocab(multinli_train, multinli_dev_match, multinli_dev_umatch,
                           multinli_test_match, multinli_test_umatch)
    print("Target labels:", label_field.vocab.itos)
    print("Available genres:", genre_field.vocab.itos)
    # TODO load vectors
    arguments.vocab_size = len(text_field.vocab)

    (multinli_train_iter,) = data.Iterator.splits((multinli_train,), batch_size=(arguments.batch_size))
    multinli_dev_match_iter, multinli_test_match_iter = data.Iterator.splits((multinli_dev_match, multinli_test_match),
                                                                             shuffle=False, batch_size=arguments.batch_size_eval)
    multinli_dev_umatch_iter, multinli_test_umatch_iter = data.Iterator.splits((multinli_dev_umatch, multinli_test_umatch),
                                                                               shuffle=False, batch_size=arguments.batch_size_eval)
    iters = (multinli_train_iter, multinli_dev_match_iter, multinli_dev_umatch_iter,
             multinli_test_match_iter, multinli_test_umatch_iter)
    return iters