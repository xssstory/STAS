
import numpy as np
import torch

from . import data_utils, FairseqDataset

SENT_SEP = '</s>'

def split_list(lst, key):
    istart = 0
    res = []
    sublist = []
    for i, v in enumerate(lst):
        sublist.append(v.item())
        if v == key:
            if len(sublist) > 0:
                res.append( sublist )
            sublist = []
    if len(sublist) > 0:
        res.append(sublist)

    return res

# right padding (easy to compute during training)
def docs2tensor(docs, max_nsent, max_sent_len, pad_idx, cls_idx):
    bsz = len(docs)
    src_tokens = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(pad_idx)
    src_tokens[:, :, 0] = cls_idx
    doc_pad_mask = torch.ByteTensor(bsz, max_nsent).fill_(1)
    for i, doc in enumerate(docs):
        for j, sent in enumerate(doc):
            doc_pad_mask[i, j] = 0
            sent_len = len(sent)
            src_tokens[i, j, :sent_len] = torch.LongTensor(sent)

    return src_tokens, doc_pad_mask

def create_src_tok_batch(samples, sep_id, cls_idx, pad_idx, key, max_sent_length=None):
    docs = []
    max_nsent = 0
    max_sent_len = 0
    for sample in samples:
        sents = sample[key]

        if max_sent_length is not None:
            sents = [ sent if len(sent) <= max_sent_length-2 else sent[0:max_sent_length-2] + [sep_id] for sent in sents]
        sents = [[cls_idx] + sent for sent in sents]
        assert sents
        if sents[-1][-1] != sep_id:
            sents[-1].append(sep_id)
        max_nsent = max(max_nsent, len(sents))
        cur_max_sent_len = max( map(len, sents) )
        max_sent_len = max( max_sent_len, cur_max_sent_len )
        docs.append(sents)

    return docs2tensor(docs, max_nsent, max_sent_len, pad_idx, cls_idx)

# def create_target_batch(samples, pad_idx):
#     maxlen = max( [len(s['target']) for s in samples] )
#     bsz = len(samples)
#     target = torch.LongTensor(bsz, maxlen).fill_(pad_idx)
#     for i, s in enumerate(samples):
#         tgt = s['target']
#         tgt_len = len(tgt)
#         target[i, 0:tgt_len] = tgt
#     return target

def collate(samples, src_dict, tgt_dict, left_pad_source=False, left_pad_target=False, max_sent_len=None):
    if len(samples) == 0:
        return {}
    global SENT_SEP
    if src_dict.index(SENT_SEP) == src_dict.unk_index:
        SENT_SEP = '[SEP]'
    sep_id = src_dict.index(SENT_SEP)
    assert sep_id != src_dict.unk_index

    id = torch.LongTensor([s['id'] for s in samples])
    selected_sent_tokens, select_sent_mask = create_src_tok_batch(samples, src_dict.index(SENT_SEP), src_dict.cls_index, src_dict.pad(), 'select_sent', max_sent_length=max_sent_len)

    positive_sent_tokens, positive_sent_mask = create_src_tok_batch(samples, src_dict.index(SENT_SEP), src_dict.cls_index, src_dict.pad(), 'positive_sents', max_sent_length=max_sent_len)
    negative_sent_tokens, negative_sent_mask = create_src_tok_batch(samples, src_dict.index(SENT_SEP), src_dict.cls_index, src_dict.pad(), 'negative_sents', max_sent_length=max_sent_len)    

    assert sum([select_sent_mask.sum(), positive_sent_mask.sum(), negative_sent_mask.sum()]) == 0

    ntokens = sum([ sum([sum([len(i) for i in v]) for k, v in sample.items() if k != 'id']) for sample in samples ])

    return {
        'id': id,
        'ntokens': ntokens,
        'nsentences': id.shape[0],
        'net_input': {
            'src_tokens':{
            'selected_sent_tokens': selected_sent_tokens.squeeze(dim=1), # bsz x len_sent
            'positive_sent_tokens': positive_sent_tokens,   # bsz x 2 x len_sent
            'negative_sent_tokens': negative_sent_tokens,    # bsz x negative_number x len_sent
            }
        },
    }


class FinetunePacsumDatasetV2(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        max_sent_len=None,
        max_doc_len=None,
        negative_num=5,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.max_sent_len = 60
        self.max_doc_len = None
        self.negative_num = negative_num

        global SENT_SEP
        if self.src_dict.index(SENT_SEP) == self.src_dict.unk_index:
            SENT_SEP = '[SEP]'
        self.sent_sep_idx = self.src_dict.index(SENT_SEP)
        print(SENT_SEP, self.sent_sep_idx)

    def __getitem__(self, index):

        global SENT_SEP
        if self.src_dict.index(SENT_SEP) == self.src_dict.unk_index:
            SENT_SEP = '[SEP]'
        sep_id = self.src_dict.index(SENT_SEP)
        assert sep_id != self.src_dict.unk_index, SENT_SEP

        doc = self.src[index]
        doc = split_list(doc, sep_id)
        # doc = doc[0: self.max_doc_len]

        # select_idx = torch.randint(len(doc)-1, [1]).item()
        select_sent = doc[0]
        select_idx = torch.randint(2, [1]).item()
        positive_sents = doc[select_idx + 1]

        # if select_sent == 0:
        #     positive_sents = [doc[1], doc[1]]
        # elif select_sent == len(doc) - 1:
        #     positive_sents = [doc[-2], doc[-2]]
        # else:
        #     positive_sents = [doc[select_idx-1], doc[select_idx+1]]
        negative_sents = self.get_random_sents(index, self.negative_num, sep_id)

        return {
            'id': index,
            'select_sent': [select_sent],
            'positive_sents': [positive_sents],
            'negative_sents': negative_sents
        }

    def get_random_sents(self, index, simple_size, sep_id):
        sents = []
        for i in range(simple_size):
            idx = index
            count = 0
            while(abs(idx - index) < 5):
                idx = torch.randint(len(self.src), [1]).item()
                sampled_doc = self.src[idx]
                if count > 20:
                    raise(RuntimeError, '{}, {}, {}'.format(idx, index, sampled_doc))
                count += 1
            sampled_doc = split_list(sampled_doc, sep_id)
            sents.append(sampled_doc[0])
        return sents

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, self.src_dict, self.tgt_dict,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            max_sent_len=self.max_sent_len,
        )

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        '''we need random order'''
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        '''
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        '''
        return indices

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)