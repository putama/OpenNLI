import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NLI_Model(nn.Module):
    def load_pretrained_emb(self, arguments):
        if arguments.use_pretrained_emb:
            emb_path = os.path.join(arguments.data_root,
                                    arguments.nli_dataset,
                                    "extracted_emb.pt")

            if not os.path.exists(emb_path):
                raise Exception("embedding file not found. "
                                "run extract_embedding.py first!")

            emb_tensor = torch.load(emb_path)
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


def to_cuda(object: torch.Tensor):
    if torch.cuda.is_available():
        return object.cuda()
    else:
        return object