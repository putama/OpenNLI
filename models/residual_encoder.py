import torch
import torch.nn as nn
from models.nli_model import NLI_Model


class ResidualEncoder(NLI_Model):
    def __init__(self, arguments, h_size=[600, 600, 600], max_l=60):
        """
        :param arguments: main python arguments
        :param h_size: hidden size of each layer of LSTM
        :param dropout_r: dropout rate
        :param max_l: maximum sequence length
        :param k:
        :param n_layers:
        """
        super(ResidualEncoder, self).__init__()
        self.embeddings = nn.Embedding(arguments.vocab_size, arguments.embedding_dim)

        self.lstm = nn.LSTM(input_size=arguments.embedding_dim, hidden_size=h_size[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(arguments.embedding_dim + h_size[0] * 2),
                              hidden_size=h_size[1],num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(arguments.embedding_dim + h_size[0] * 2),
                              hidden_size=h_size[2], num_layers=1, bidirectional=True)

        self.max_l = max_l
        self.h_size = h_size

        if arguments.n_layers == 1:
            self.classifier = nn.Sequential(*[nn.Linear(h_size[2] * 2 * 4,
                                                        arguments.fc_dim),
                                              nn.ReLU(),
                                              nn.Dropout(arguments.dropout_rate),
                                              nn.Linear(arguments.fc_dim, 3)])
        elif arguments.n_layers == 2:
            self.classifier = nn.Sequential(*[nn.Linear(h_size[2] * 2 * 4,
                                                        arguments.fc_dim),
                                              nn.ReLU(),
                                              nn.Dropout(arguments.dropout_rate),
                                              nn.Linear(arguments.fc_dim,
                                                        arguments.fc_dim),
                                              nn.ReLU(),
                                              nn.Dropout(arguments.dropout_rate),
                                              nn.Linear(arguments.fc_dim, 3)])
        else:
            raise Exception("invalid number of MLP layers")

        # set loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def count_params(self):
        total_c = 0
        for param in self.parameters():
            if len(param.size()) == 2:
                d1, d2 = param.size()[0], param.size()[1]
                total_c += d1 * d2
        print("Total count:", total_c)


    def forward(self, s1, l1, s2, l2):
        if self.max_l:
            l1 = l1.clamp(max=self.max_l)
            l2 = l2.clamp(max=self.max_l)
            if s1.size(0) > self.max_l:
                s1 = s1[:self.max_l, :]
            if s2.size(0) > self.max_l:
                s2 = s2[:self.max_l, :]

        p_s1 = self.embeddings(s1)
        p_s2 = self.embeddings(s2)

        s1_layer1_out = self.rnn_autosort_forward(self.lstm, p_s1, l1)
        s2_layer1_out = self.rnn_autosort_forward(self.lstm, p_s2, l2)

        # Length truncate
        len1 = s1_layer1_out.size(0)
        len2 = s2_layer1_out.size(0)
        p_s1 = p_s1[:len1, :, :]
        p_s2 = p_s2[:len2, :, :]

        # Using high way
        s1_layer2_in = torch.cat([p_s1, s1_layer1_out], dim=2)
        s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)

        s1_layer2_out = self.rnn_autosort_forward(self.lstm_1, s1_layer2_in, l1)
        s2_layer2_out = self.rnn_autosort_forward(self.lstm_1, s2_layer2_in, l2)

        s1_layer3_in = torch.cat([p_s1, s1_layer1_out + s1_layer2_out], dim=2)
        s2_layer3_in = torch.cat([p_s2, s2_layer1_out + s2_layer2_out], dim=2)

        s1_layer3_out = self.rnn_autosort_forward(self.lstm_2, s1_layer3_in, l1)
        s2_layer3_out = self.rnn_autosort_forward(self.lstm_2, s2_layer3_in, l2)

        s1_layer3_maxout = self.max_along_time(s1_layer3_out, l1)
        s2_layer3_maxout = self.max_along_time(s2_layer3_out, l2)

        # Only use the last layer
        features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
                              torch.abs(s1_layer3_maxout - s2_layer3_maxout),
                              s1_layer3_maxout * s2_layer3_maxout],
                             dim=1)

        out = self.classifier(features)
        return out

    def max_along_time(self, inputs, lengths, batch_first=False):
        """
        :param inputs: [T * B * D]
        :param lengths:  [B]
        :return: [B * D] max_along_time
        """
        ls = list(lengths)

        if not batch_first:
            b_seq_max_list = []
            for i, l in enumerate(ls):
                seq_i = inputs[:l, i, :]
                seq_i_max, _ = seq_i.max(dim=0)
                seq_i_max = seq_i_max.squeeze()
                b_seq_max_list.append(seq_i_max)

            return torch.stack(b_seq_max_list)
        else:
            b_seq_max_list = []
            for i, l in enumerate(ls):
                seq_i = inputs[i, :l, :]
                seq_i_max, _ = seq_i.max(dim=0)
                seq_i_max = seq_i_max.squeeze()
                b_seq_max_list.append(seq_i_max)

            return torch.stack(b_seq_max_list)

    def compute_loss(self, logit, target):
        """
        :param logit: matrix of logit vectors of size [batchsize x n_class]
        :param target: tensor of true labels [batchsize]
        :return: loss computed based on defined criterion
        """
        return self.criterion(logit, target)