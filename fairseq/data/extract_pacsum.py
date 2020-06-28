
import numpy as np
import torch

from . import data_utils, FairseqDataset

SENT_SEP = '</s>'
MAX_DOC_LEN = 512 # 512 words for each doc due to bert embedding
MAX_NUM_SENT = 100000

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
    doc_pad_mask = torch.BoolTensor(bsz, max_nsent).fill_(1)
    for i, doc in enumerate(docs):
        for j, sent in enumerate(doc):
            doc_pad_mask[i, j] = 0
            sent_len = len(sent)
            src_tokens[i, j, :sent_len] = torch.LongTensor(sent)

    return src_tokens, doc_pad_mask

def create_src_tok_batch(samples, sep_id, cls_idx, pad_idx, max_sent_length=None):
    docs = []
    max_nsent = 0
    max_sent_len = 0
    for sample in samples:
        src = sample['source']
        sents = split_list(src, sep_id)
        sents = sents[:MAX_NUM_SENT] if len(sents) > MAX_NUM_SENT else sents

        if max_sent_length is not None:
            sents = [ sent if len(sent) <= max_sent_length-2 else sent[0:max_sent_length-2] + [sep_id] for sent in sents]
        sents = [[cls_idx] + sent for sent in sents]
        if not sents:
            sents = [[sep_id]]
            print('this sents has no content')
        if sents[-1][-1] != sep_id:
            sents[-1].append(sep_id)
        max_nsent = max(max_nsent, len(sents))
        cur_max_sent_len = max( map(len, sents) )
        max_sent_len = max( max_sent_len, cur_max_sent_len )
        docs.append(sents)

    return docs2tensor(docs, max_nsent, max_sent_len, pad_idx, cls_idx)

def create_target_batch(samples, pad_idx):
    maxlen = max( [len(s['target']) for s in samples] )
    bsz = len(samples)
    target = torch.LongTensor(bsz, maxlen).fill_(pad_idx)
    for i, s in enumerate(samples):
        tgt = s['target']
        tgt_len = len(tgt)
        target[i, 0:tgt_len] = tgt
    return target

def collate(samples, src_dict, tgt_dict, left_pad_source=False, left_pad_target=False, max_sent_len=None):
    if len(samples) == 0:
        return {}

    global SENT_SEP
    if src_dict.index(SENT_SEP) == src_dict.unk_index:
        SENT_SEP = '[SEP]'

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens, doc_pad_mask = create_src_tok_batch(samples, src_dict.index(SENT_SEP), src_dict.cls_index, src_dict.pad(), max_sent_length=max_sent_len)
    segment_ids = torch.zeros_like(src_tokens)
    '''
    print('***********************************')
    for i in range(src_tokens.size(0)):
        print( src_dict.string_complete(src_tokens[i]) )
    print('***********************************')
    '''

    # simply add a sepecial token
    doc_pos_tok = torch.LongTensor( doc_pad_mask.size() ).copy_(src_tokens[:, :, 0])
    doc_pos_tok[ doc_pad_mask ] = src_dict.pad()

    # ntokens = sum(len(s['target']) for s in samples)
    # ntokens = sum(len(s['target']) if s['target'] else (s['source']==src_dict.index(SENT_SEP)).sum() + 1 for s in samples )
    ntokens = int((doc_pad_mask==0).sum())
    target = create_target_batch(samples, tgt_dict.pad()) if samples[0]['target'] is not None else None


    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'doc_pad_mask': doc_pad_mask,
            'segment_ids': segment_ids,
            'doc_pos_tok': doc_pos_tok,
        },
        'target': target,
    }


class ExtractPacsumDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        max_sent_len=None,
        max_doc_len=None,
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
        self.max_doc_len = max_doc_len

        self.sent_sep_idx = self.src_dict.index(SENT_SEP)
        print(SENT_SEP, self.sent_sep_idx)

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, self.src_dict, self.tgt_dict,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            max_sent_len=self.max_sent_len,
        )

    def get_dummy_batch(self, num_docs, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        # src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_docs

        sent_len = MAX_DOC_LEN // self.max_doc_len
        last_sent_len = MAX_DOC_LEN - (self.max_doc_len-1)*sent_len

        def create_tgt():
            return torch.LongTensor([self.tgt_dict.index('F')] * self.max_doc_len)

        def create_src():
            doc = []
            for i in range(self.max_doc_len):
                cur_sent_len = sent_len if i != self.max_doc_len-1 else last_sent_len
                for j in range(cur_sent_len-1):
                    doc.append(self.src_dict.unk())
                if i != self.max_doc_len-1:
                    doc.append(self.sent_sep_idx)
            return torch.LongTensor(doc)

        batch = self.collater([
            {
                'id': i,
                'source': create_src(),
                'target': create_tgt(),
            }
            for i in range(bsz)
        ])

        return batch

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