import torch
import numpy as np

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

def ids2words(vocab, ids1, ids2=None):
    if ids2 is None:
        ids_list = ids1.transpose(1, 0).cpu().tolist()
        senttext_list = []
        for sent in ids_list:
            senttext = " ".join([vocab.itos[word_id] for word_id in sent])
            senttext_list.append(senttext)
        return senttext_list
    else:
        ids_list1 = ids1.transpose(1, 0).cpu().tolist()
        ids_list2 = ids2.transpose(1, 0).cpu().tolist()
        sentpairtext_list = []
        for sent1, sent2 in zip(ids_list1, ids_list2):
            senttext1 = " ".join([vocab.itos[word_id] for word_id in sent1])
            senttext2 = " ".join([vocab.itos[word_id] for word_id in sent2])
            sentpair = "{} ==> {}".format(senttext1, senttext2)
            sentpairtext_list.append(sentpair)
        return sentpairtext_list

def labels2prob(labels):
    labelslist = labels.cpu().tolist()
    probs = []
    for ex in labelslist:
        counter = np.zeros(3)
        for label in ex:
            counter[label] += 1
        probs.append(counter / counter.sum())
    return np.array(probs)

def probs2entropy(probs):
    entropies = []
    for ex in probs:
        accum = 0
        for prob in ex:
            if prob > 0:
                accum += np.log(prob) * prob
        entropies.append(-accum)
    return entropies