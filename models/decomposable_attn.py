import torch
from torch import nn
import torch.nn.functional as F
from models.nli_model import NLI_Model

class DecompAttention(NLI_Model):

    def __init__(self, args, para_init=0.01):
        super(DecompAttention, self).__init__()

        self.embedding_size = args.embedding_dim
        self.label_size = super().label_num
        self.para_init = para_init
        self.max_length = args.max_seq_length

        self.embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.mlp_f = self._mlp_layers(self.embedding_size, self.embedding_size)
        self.mlp_g = self._mlp_layers(2 * self.embedding_size, self.embedding_size)
        self.mlp_h = self._mlp_layers(2 * self.embedding_size, self.embedding_size)

        self.final_linear = nn.Linear(self.embedding_size, self.label_size, bias=True)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                m.bias.data.normal_(0, self.para_init)

        # set loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)

    def forward(self, sent1, len1, sent2, len2):
        """
        :param sent1: premise sentence
        :param len1: lengths of each premise
        :param sent2: hypothesis sentence
        :param len2: lengths of each hypothesis
        :return: class scores vector
        """
        if self.max_length:
            len1 = len1.clamp(max=self.max_length)
            if sent1.size(0) > self.max_length:
                sent1 = sent1[:self.max_length, :]
            len2 = len2.clamp(max=self.max_length)
            if sent2.size(0) > self.max_length:
                sent2 = sent2[:self.max_length, :]

        # retrieve the lexical embeddings
        sent1 = self.embeddings(sent1)
        sent2 = self.embeddings(sent2)
        # swap the dimension of batch and length
        sent1 = torch.transpose(sent1, 1, 0).contiguous()
        sent2 = torch.transpose(sent2, 1, 0).contiguous()

        maxlen1 = len1.max().item()  # max len of sent1
        maxlen2 = len2.max().item()  # max len of sent2

        '''attend'''
        f1 = self.mlp_f(sent1.view(-1, self.embedding_size))
        f2 = self.mlp_f(sent2.view(-1, self.embedding_size))

        f1 = f1.view(-1, maxlen1, self.embedding_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, maxlen2, self.embedding_size)
        # batch_size x len2 x hidden_size

        # compute e_{ij}, out dim: batch_size x len1 x len2
        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # compute softmax dist for each a_i over b_j
        # softmax out: batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, maxlen2), dim=1).view(-1, maxlen1, maxlen2)

        # compute e_{ji}, out dim: batch_size x len2 x len1
        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # compute softmax dist for each b_j over a_i
        # softmax out: batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, maxlen1), dim=1).view(-1, maxlen2, maxlen1)

        # cat each word emb with its (soft) aligned counterpart
        sent1_combine = torch.cat((sent1, torch.bmm(prob1, sent2)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat((sent2, torch.bmm(prob2, sent1)), 2)
        # batch_size x len2 x (hidden_size x 2)

        # project the concatenation back to its original size
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.embedding_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.embedding_size))
        g1 = g1.view(-1, maxlen1, self.embedding_size)
        g2 = g2.view(-1, maxlen2, self.embedding_size)

        sent1_output = self.sum_along_time(g1, len1, batch_first=True)
        sent2_output = self.sum_along_time(g2, len2, batch_first=True)

        # concat sum-pooled representation and
        # project it back to original size
        input_combine = torch.cat((sent1_output, sent2_output), 1)
        h = self.mlp_h(input_combine)
        # linear classification
        out = self.final_linear(h)

        return out

