import torchtext
from torchtext import data, vocab, datasets

def prepare_mnli_data_loader(arguments, reversed=False):
    # text_field = ParsedTextLField(reversed=reversed)
    text_field = data.Field(include_lengths=True)
    label_field = data.Field(unk_token=None, sequential=False)
    genre_field = data.Field(unk_token=None, sequential=False)
    parse_field = datasets.nli.ShiftReduceField()

    print("splitting MultiNLI datasets...")
    splits_match = datasets.MultiNLI.splits(text_field, label_field, parse_field,
                                            root=arguments.data_root,
                                            train="multinli_1.0_dev_matched.jsonl",
                                            validation="multinli_1.0_dev_matched.jsonl",
                                            test="multinli_0.9_test_matched_unlabeled.jsonl")
    multinli_train, multinli_dev_match, multinli_test_match = splits_match
    splits_umatch = datasets.MultiNLI.splits(text_field, label_field, parse_field,
                                             root=arguments.data_root, train=None,
                                             validation="multinli_1.0_dev_mismatched.jsonl",
                                             test="multinli_0.9_test_mismatched_unlabeled.jsonl")
    multinli_dev_umatch, multinli_test_umatch = splits_umatch

    print("building vocabularies from each sets...")
    text_field.build_vocab(multinli_train, multinli_dev_match, multinli_dev_umatch,
                           multinli_test_match, multinli_test_umatch)
    label_field.build_vocab(multinli_train)
    # TODO build vocab for genre here
    print("Target labels:", label_field.vocab.itos)
    # TODO load vectors

    (multinli_train_iter,) = data.Iterator.splits((multinli_train,), batch_size=(arguments.batch_size))
    multinli_dev_match_iter, multinli_test_match_iter = data.Iterator.splits((multinli_dev_match, multinli_test_match),
                                                                             shuffle=False, batch_size=arguments.batch_size_eval)
    multinli_dev_umatch_iter, multinli_test_umatch_iter = data.Iterator.splits((multinli_dev_umatch, multinli_test_umatch),
                                                                               shuffle=False, batch_size=arguments.batch_size_eval)
    iters = (multinli_train_iter, multinli_dev_match_iter, multinli_dev_umatch_iter,
             multinli_test_match_iter, multinli_test_umatch_iter)
    return iters



class ParsedTextLField(data.Field):
    def __init__(self, eos_token='<pad>', lower=False, include_lengths=True, reversed=False):
        if reversed:
            super(ParsedTextLField, self).__init__(
                eos_token=eos_token, lower=lower, include_lengths=include_lengths,
                preprocessing=lambda parse: [t for t in parse if t not in ('(', ')')],
                postprocessing=lambda parse, _, __: [list(reversed(p)) for p in parse])
        else:
            super(ParsedTextLField, self).__init__(
                eos_token=eos_token, lower=lower, include_lengths=include_lengths,
                preprocessing=lambda parse: [t for t in parse if t not in ('(', ')')])
        self.reversed = reversed

    def plugin_new_words(self, new_vocab):
        for word, i in new_vocab.stoi.items():
            if word in self.vocab.stoi:
                continue
            else:
                self.vocab.itos.append(word)
                self.vocab.stoi[word] = len(self.vocab.itos)-1
