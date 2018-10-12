import torch
from torch import nn
from opennli.models.nli_model import NLI_Model


class InferSent(NLI_Model):
    def __init__(self, args):
        super(InferSent, self).__init__()

        # sentence encoder
        self.word_emb_dim = args.embedding_dim
        self.enc_lstm_dim = args.lstm_dim
        self.pool_type = args.pool_type
        self.dpout_model = args.lstm_dropout_rate
        self.max_length = args.max_seq_length

        self.embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                            bidirectional=True, dropout=self.dpout_model)

        # classifier
        self.nonlinear_fc = args.nonlinear_fc
        self.fc_dim = args.fc_dim
        self.lstm_dim = args.lstm_dim
        self.dpout_fc = args.dropout_rate
        self.inputdim = 2 * 4 * self.lstm_dim

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, super().label_num)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, super().label_num)
            )

        # set loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def encode_sentence(self, sent, len):
        if self.max_length:
            len = len.clamp(max=self.max_length)
            if sent.size(0) > self.max_length:
                sent = sent[:self.max_length, :]

        embeds = self.embeddings(sent)
        sent_enc_out = self.rnn_autosort_forward(self.lstm, embeds, len)
        # pooling of contextualized word representation over steps
        if self.pool_type == "mean":
            sent_enc_pooled = self.mean_along_time(sent_enc_out, len)
        elif self.pool_type == "max":
            sent_enc_pooled = self.max_along_time(sent_enc_out, len)

        return sent_enc_pooled

    def forward(self, sent1, len1, sent2, len2):
        s1_enc_pooled = self.encode_sentence(sent1, len1)
        s2_enc_pooled = self.encode_sentence(sent2, len2)

        classifier_input = torch.cat((s1_enc_pooled,
                                      s2_enc_pooled,
                                      s1_enc_pooled - s2_enc_pooled,
                                      s1_enc_pooled * s2_enc_pooled), 1)

        out = self.classifier(classifier_input)
        return out

