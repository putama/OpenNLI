import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utilities import to_cuda


class NLI_Model(nn.Module):

    label_num = 3

    def display(self):
        print(self)
        print("is GPU?: {}".format(str(next(self.parameters()).is_cuda)))


    def load_pretrained_emb(self, arguments):
        if arguments.use_pretrained_emb:
            emb_path = os.path.join(arguments.data_root,
                                    arguments.nli_dataset,
                                    "extracted_emb.pt.tar")

            if not os.path.exists(emb_path):
                raise Exception("embedding file not found. "
                                "run extract_embedding.py first!")

            emb_tensor = torch.load(emb_path, map_location=lambda storage, loc: storage)

            if not self.embeddings.weight.data.size() == emb_tensor.size():
                raise Exception("size mismatch between model emb and the loaded one")

            self.embeddings.weight.data = emb_tensor
            return True
        else:
            return False

    def rnn_autosort_forward(self, rnn: nn.RNNBase, input, input_len):
        input_len_sorted, idx_sort = input_len.sort(descending=True)
        _, idx_unsort = idx_sort.sort()
        idx_sort = to_cuda(idx_sort)
        input_reordered = input.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        input_reordered_packed = pack_padded_sequence(input_reordered, input_len_sorted)
        output, _ = rnn(
            input_reordered_packed)  # batch_max_len x batch_size x hidden_size
        output, _ = pad_packed_sequence(output)

        # Un-sort by length
        idx_unsort = to_cuda(idx_unsort)
        output_ori_order = output.index_select(1, Variable(idx_unsort))

        return output_ori_order

    def max_along_time(self, inputs, lengths, batch_first=False):
        return self.pool_along_time("max", inputs, lengths, batch_first=batch_first)

    def mean_along_time(self, inputs, lengths, batch_first=False):
        return self.pool_along_time("mean", inputs, lengths, batch_first=batch_first)

    def sum_along_time(self, inputs, lengths, batch_first=False):
        return self.pool_along_time("sum", inputs, lengths, batch_first=batch_first)

    def pool_along_time(self, pooling_type, inputs, lengths, batch_first=False):
        """
        :param inputs: [T * B * D]
        :param lengths:  [B]
        :return: [B * D] max_along_time
        """
        ls = list(lengths)

        b_seq_pooled_list = []
        for i, l in enumerate(ls):
            if not batch_first:
                seq_i = inputs[:l, i, :]
            else:
                seq_i = inputs[i, :l, :]

            if pooling_type == "max":
                seq_i_pooled, _ = seq_i.max(dim=0)
            elif pooling_type == "mean":
                seq_i_pooled = seq_i.mean(dim=0)
            elif pooling_type == "sum":
                seq_i_pooled = seq_i.sum(dim=0)
            b_seq_pooled_list.append(seq_i_pooled)
        return torch.stack(b_seq_pooled_list)

    def compute_loss(self, logit, target):
        """
        :param logit: matrix of logit vectors of size [batchsize x n_class]
        :param target: tensor of true labels [batchsize]
        :return: loss computed based on defined criterion
        """
        return self.criterion(logit, target)