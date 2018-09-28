import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os

def pad(t, length):
    if length == t.size(0):
        return t
    else:
        return torch.cat([t, Variable(t.data.new(length - t.size(0), *t.size()[1:]).zero_())])

def pack_list_sequence(inputs, l):
    batch_list = []
    max_l = max(list(l))
    batch_size = len(inputs)

    for b_i in range(batch_size):
        batch_list.append(pad(inputs[b_i], max_l))
    pack_batch_list = torch.stack(batch_list, dim=1)
    return pack_batch_list


def pack_for_rnn_seq(inputs, lengths):
    """
    :param inputs: [T * B * D]
    :param lengths:  [B]
    :return:
    """
    _, sorted_indices = lengths.sort()
    '''
        Reverse to decreasing order
    '''
    r_index = reversed(list(sorted_indices))

    s_inputs_list = []
    lengths_list = []
    reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

    for j, i in enumerate(r_index):
        s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
        lengths_list.append(lengths[i])
        reverse_indices[i] = j

    reverse_indices = list(reverse_indices)

    s_inputs = torch.cat(s_inputs_list, 1)
    packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

    return packed_seq, reverse_indices


def unpack_from_rnn_seq(packed_seq, reverse_indices):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq)
    s_inputs_list = []

    for i in reverse_indices:
        s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
    return torch.cat(s_inputs_list, 1)


def auto_rnn_bilstm(lstm: nn.LSTM, seqs, lengths):

    batch_size = seqs.size(1)

    state_shape = lstm.num_layers * 2, batch_size, lstm.hidden_size

    h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths)

    output, (hn, cn) = lstm(packed_pinputs, (h0, c0))

    output = unpack_from_rnn_seq(output, r_index)

    return output

def max_along_time(inputs, lengths):
    """
    :param inputs: [T * B * D]
    :param lengths:  [B]
    :return: [B * D] max_along_time
    """
    ls = list(lengths)

    b_seq_max_list = []
    for i, l in enumerate(ls):
        seq_i = inputs[:l, i, :]
        seq_i_max, _ = seq_i.max(dim=0)
        seq_i_max = seq_i_max.squeeze()
        b_seq_max_list.append(seq_i_max)

    return torch.stack(b_seq_max_list)

class StackedEncoder(nn.Module):
    def __init__(self, args, h_size=[512, 1024, 2048], d=300, mlp_d=1600, dropout_r=0.1, max_l=60):
        super(StackedEncoder, self).__init__()
        self.Embd = nn.Embedding(args.vocab_size, d)

        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(d + h_size[0] * 2), hidden_size=h_size[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(d + (h_size[0] + h_size[1]) * 2), hidden_size=h_size[2],
                              num_layers=1, bidirectional=True)

        self.max_l = max_l
        self.h_size = h_size

        self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                          self.sm])

        # set loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def display(self):
        for param in self.parameters():
            print(param.data.size())

    def load_pretrained_emb(self, arguments):
        if arguments.use_pretrained_emb:
            emb_path = os.path.join(arguments.data_root,
                                    arguments.nli_dataset,
                                    "extracted_emb.pt.tar")

            if not os.path.exists(emb_path):
                raise Exception("embedding file not found. "
                                "run extract_embedding.py first!")

            emb_tensor = torch.load(emb_path, map_location=lambda storage, loc: storage)

            if not self.Embd.weight.data.size() == emb_tensor.size():
                raise Exception("size mismatch between model emb and the loaded one")

            # normalized embbedings into unit vector
            # freeze the parameters
            if arguments.normalized_fixed:
                print("normalize embeddings")
                emb_tensor = self.normalize_emb(emb_tensor)
                self.embeddings.weight.data = emb_tensor
                self.embeddings.weight.require_grad = False
            else:
                self.Embd.weight.data = emb_tensor
            return True
        else:
            return False

    def forward(self, s1, l1, s2, l2):
        if self.max_l:
            l1 = l1.clamp(max=self.max_l)
            l2 = l2.clamp(max=self.max_l)
            if s1.size(0) > self.max_l:
                s1 = s1[:self.max_l, :]
            if s2.size(0) > self.max_l:
                s2 = s2[:self.max_l, :]

        p_s1 = self.Embd(s1)
        p_s2 = self.Embd(s2)

        s1_layer1_out = auto_rnn_bilstm(self.lstm, p_s1, l1)
        s2_layer1_out = auto_rnn_bilstm(self.lstm, p_s2, l2)

        # Length truncate
        len1 = s1_layer1_out.size(0)
        len2 = s2_layer1_out.size(0)
        p_s1 = p_s1[:len1, :, :] # [T, B, D]
        p_s2 = p_s2[:len2, :, :] # [T, B, D]

        # Using residual connection
        s1_layer2_in = torch.cat([p_s1, s1_layer1_out], dim=2)
        s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)

        s1_layer2_out = auto_rnn_bilstm(self.lstm_1, s1_layer2_in, l1)
        s2_layer2_out = auto_rnn_bilstm(self.lstm_1, s2_layer2_in, l2)

        s1_layer3_in = torch.cat([p_s1, s1_layer1_out, s1_layer2_out], dim=2)
        s2_layer3_in = torch.cat([p_s2, s2_layer1_out, s2_layer2_out], dim=2)

        s1_layer3_out = auto_rnn_bilstm(self.lstm_2, s1_layer3_in, l1)
        s2_layer3_out = auto_rnn_bilstm(self.lstm_2, s2_layer3_in, l2)

        s1_layer3_maxout = max_along_time(s1_layer3_out, l1)
        s2_layer3_maxout = max_along_time(s2_layer3_out, l2)

        # Only use the last layer
        features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
                              torch.abs(s1_layer3_maxout - s2_layer3_maxout),
                              s1_layer3_maxout * s2_layer3_maxout],
                             dim=1)

        out = self.classifier(features)
        return out

    def compute_loss(self, logit, target):
        """
        :param logit: matrix of logit vectors of size [batchsize x n_class]
        :param target: tensor of true labels [batchsize]
        :return: loss computed based on defined criterion
        """
        return self.criterion(logit, target)