import torch
import torch.nn as nn
from opennli.models.nli_model import NLI_Model


class ResidualEncoder(NLI_Model):
    def __init__(self, arguments):
        """
        :param arguments: main python arguments
        """
        super(ResidualEncoder, self).__init__()

        self.max_length = arguments.max_seq_length
        self.hidden_size = arguments.lstm_dims
        self.pool_type = arguments.pool_type

        self.embeddings = nn.Embedding(arguments.vocab_size, arguments.embedding_dim)

        self.lstm = nn.LSTM(input_size=arguments.embedding_dim,
                            hidden_size=self.hidden_size[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(arguments.embedding_dim +
                                          self.hidden_size[0] * 2),
                              hidden_size=self.hidden_size[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(arguments.embedding_dim +
                                          self.hidden_size[0] * 2 +
                                          self.hidden_size[1] * 2),
                              hidden_size=self.hidden_size[2], num_layers=1,
                              bidirectional=True)

        if arguments.n_layers == 1:
            self.classifier = nn.Sequential(*[nn.Linear(self.hidden_size[2] * 2 * 4,
                                                        arguments.fc_dim),
                                              nn.ReLU(),
                                              nn.Dropout(arguments.dropout_rate),
                                              nn.Linear(arguments.fc_dim,
                                                        super().label_num)])
        elif arguments.n_layers == 2:
            self.classifier = nn.Sequential(*[nn.Linear(self.hidden_size[2] * 2 * 4,
                                                        arguments.fc_dim),
                                              nn.ReLU(),
                                              nn.Dropout(arguments.dropout_rate),
                                              nn.Linear(arguments.fc_dim,
                                                        arguments.fc_dim),
                                              nn.ReLU(),
                                              nn.Dropout(arguments.dropout_rate),
                                              nn.Linear(arguments.fc_dim,
                                                        super().label_num)])
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

    def encode_sentence(self, sent, len):
        if self.max_length:
            len = len.clamp(max=self.max_length)
            if sent.size(0) > self.max_length:
                sent = sent[:self.max_length, :]

        # layer-1 encoding
        embeds = self.embeddings(sent)
        sent_layer1_out = self.rnn_autosort_forward(self.lstm, embeds, len)
        # shortcut connection #1
        sent_layer2_in = torch.cat([embeds, sent_layer1_out], dim=2)
        # layer-2 encoding
        sent_layer2_out = self.rnn_autosort_forward(self.lstm_1, sent_layer2_in, len)
        # shortcut connection #2
        sent_layer3_in = torch.cat([embeds, sent_layer1_out, sent_layer2_out], dim=2)
        # layer-3 encoding
        sent_layer3_out = self.rnn_autosort_forward(self.lstm_2, sent_layer3_in, len)
        # pooling of contextualized word representation over steps
        if self.pool_type == "mean":
            sent_enc_pooled = self.mean_along_time(sent_layer3_out, len)
        elif self.pool_type == "max":
            sent_enc_pooled = self.max_along_time(sent_layer3_out, len)

        return sent_enc_pooled

    def forward(self, sent1, len1, sent2, len2):
        s1_enc_pooled = self.encode_sentence(sent1, len1)
        s2_enc_pooled = self.encode_sentence(sent2, len2)

        classifier_input = torch.cat((s1_enc_pooled,
                                      s2_enc_pooled,
                                      torch.abs(s1_enc_pooled - s2_enc_pooled),
                                      s1_enc_pooled * s2_enc_pooled),
                                     dim=1)

        logit = self.classifier(classifier_input)
        return logit

