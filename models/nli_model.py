import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utilities import to_cuda

class NLI_Model(nn.Module):
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