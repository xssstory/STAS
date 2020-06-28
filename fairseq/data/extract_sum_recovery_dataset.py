
import numpy as np
import torch

from . import data_utils, FairseqDataset

sent_sep_dic = {
    'roberta-base': '</s>',
    'roberta-large': '</s>',
    'bert-base-uncased': '[SEP]',
    'bert-base-chinese': '[SEP]'
}

# SENT_SEP = '</s>'
SENT_MASK = '<sent_mask>'
MAX_DOC_LEN = 512 # 512 words for each doc due to bert embedding

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

def mask_others(src_tokens, cpos, mask_other_sents, pad_idx):

    token_mask = src_tokens.ne(pad_idx)
    if mask_other_sents:
        n_tokens = token_mask.sum(dim=-1)
        token_mask[:] = False
        token_mask.unsqueeze_(1).unsqueeze_(2)
        token_mask = token_mask.repeat([1, 1, token_mask.shape[-1], 1])
        for i in range(cpos.shape[0]):
            for j, k in zip(cpos[i, :-1], cpos[i, 1:]):
                if k == 0:
                    break
                token_mask[i, :, j:k, j:k] = True
            if k == 0:
                token_mask[i, :, j:n_tokens[i], j:n_tokens[i]] = True
            else:
                token_mask[i, :, k:n_tokens[i], k:n_tokens[i]] = True

    return token_mask

# right padding (easy to compute during training)
def docs2tensor(docs, max_nsent, max_sent_len, pad_idx, sep_idx):
    bsz = len(docs)
    src_tokens = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(pad_idx)
    src_tokens[:, :, -1] = sep_idx
    doc_pad_mask = torch.ByteTensor(bsz, max_nsent).fill_(1)
    for i, doc in enumerate(docs):
        for j, sent in enumerate(doc):
            doc_pad_mask[i, j] = 0
            sent_len = len(sent)
            src_tokens[i, j, -sent_len:] = torch.LongTensor(sent)

    return src_tokens, doc_pad_mask


def create_target_batch(samples, pad_idx, nsents):
    maxlen = max(nsents)
    bsz = len(samples)
    target = torch.LongTensor(bsz, maxlen).fill_(pad_idx)
    for i, s in enumerate(samples):
        tgt = s['target']
        tgt_len = nsents[i]
        target[i, 0:tgt_len] = tgt[0:tgt_len]
    return target


# split each doc into sentences
# sent sep included!
# [cls] w1 w2 ... [sep]
def get_docs(samples, vocab, maxlen=None, max_tokens_len=512, bert_model=None):
    sep_id = vocab.index(sent_sep_dic[bert_model])
    assert sep_id != vocab.unk_index, sent_sep_dic[bert_model]
    cls_idx = vocab.cls_index
    max_sent_length = maxlen
    docs = []
    cls_poses = []
    for sample in samples:
        source = sample['source']
        sents = split_list(source, sep_id)

        if max_sent_length is not None:
            sents = [sent if len(sent) <= max_sent_length else sent[0:max_sent_length-1] + [sep_id] for sent in sents]
        sents = [[cls_idx] + sent for sent in sents]

        if sents[-1][-1] != sep_id:
            sents[-1].append(sep_id)

        truncate_doc = []
        cls_pos = []
        truncate_doc_sents = []
        for sent in sents:
            if len(truncate_doc) + len(sent) <= max_tokens_len:
                cls_pos.append(len(truncate_doc))
                truncate_doc.extend(sent)
                # truncate_doc.append(sent)
                truncate_doc_sents.append(sent)
            else:
                break

        # print('doc len', len(truncate_doc))

        docs.append( truncate_doc_sents )
        cls_poses.append( cls_pos )

    return docs, cls_poses

class ExtractSumRecoveryDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        max_sent_len=None,
        max_doc_len=None,
        masked_sent_prob=None,
        max_predictions_per_doc=None,
        rng=None,
        doc_sizes=None,
        mask_other_sents=False,
        max_tokens_len=512,
        bert_model=None,
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
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.masked_sent_prob = masked_sent_prob
        self.max_predictions_per_doc = max_predictions_per_doc
        self.min_predictions_per_doc = 1
        self.rng = rng
        self.mask_other_sents = mask_other_sents
        self.max_tokens_len = max_tokens_len
        self.bert_model = bert_model

        self.sent_sep_idx = self.src_dict.index(sent_sep_dic[self.bert_model])
        assert self.sent_sep_idx != self.src_dict.unk_index
        print(sent_sep_dic[self.bert_model], self.sent_sep_idx)

        self.sent_mask_idx = self.src_dict.index(SENT_MASK)
        print(SENT_MASK, self.sent_mask_idx)

        # number of tokens in a doc: max_nsent x max_sent_len
        self.src_doc_sizes = doc_sizes

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
        return self.collate(
            samples, self.src_dict, self.tgt_dict,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            max_sent_len=self.max_sent_len,
            mask_other_sents=self.mask_other_sents,
        )

    def collate(self, samples, src_dict, tgt_dict, left_pad_source=True,
                left_pad_target=False, max_sent_len=None, mask_other_sents=False):

        def masked_doc_gengerator(docs):
            max_nsents = max(map(len, docs))
            for idx in range(max_nsents):
                id = torch.LongTensor([s['id'] for s in samples])        
                src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, segment_ids, cpos, nsents = self.create_batch(docs, idx, src_dict, max_sent_length=max_sent_len)
                doc_pos_tok = torch.LongTensor( doc_pad_mask.size() ).fill_(src_tokens[0, 0])
                doc_pos_tok[ doc_pad_mask ] = src_dict.pad()

                ntokens_label = sum(nsents)
                # target_label = create_target_batch(samples, tgt_dict.pad(), nsents)
                target_label = None
                # ntokens = tgt_masked_sents.ne( self.tgt_dict.pad() ).sum().item()
                ntokens = tgt_masked_sents.ne( self.src_dict.pad() ).sum().item()
                token_mask = mask_others(src_tokens, cpos, mask_other_sents, src_dict.pad())

                yield {
                    'id': id,
                    'ntokens_label': ntokens_label,
                    'ntokens': ntokens,
                    'net_input': {
                        'src_tokens': src_tokens,
                        'doc_pad_mask': doc_pad_mask,
                        'segment_ids': segment_ids,
                        'doc_pos_tok': doc_pos_tok,
                        'cls_pos': cpos,
                        'masked_sent_positions': tgt_selected_indexes,
                        'prev_output_tokens': tgt_input_masked_sents,
                        'token_mask': token_mask,
                    },
                    'target_label': target_label,
                    'target': tgt_masked_sents,
                }

        if len(samples) == 0:
            return {}    
        docs, cls_pos = get_docs(samples, src_dict, max_tokens_len=self.max_tokens_len, bert_model=self.bert_model)
        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens, doc_pad_mask, segment_ids, cpos, nsents = self.doc2tensor(docs, src_dict)
        doc_pos_tok = torch.LongTensor( doc_pad_mask.size() ).fill_(src_tokens[0, 0])
        doc_pos_tok[ doc_pad_mask ] = src_dict.pad()
        # random shuffle the sents
        # from.sents_recovery_after_shuffle_dataset import create_src_tok_batch
        # import numpy as np
        # rng = np.random.RandomState(1)
        # src_tokens, doc_pad_mask, segment_ids, cpos, nsents, tgt_orders, truncate_sents = create_src_tok_batch(samples, src_dict, max_doc_length=512, max_sent_length=55, rng=rng, shuffle_prob=1)
        token_mask = mask_others(src_tokens, cpos, mask_other_sents, src_dict.pad())
        doc_without_mask = {
            'id': id,
            'net_input':{
                'src_tokens': src_tokens,
                'doc_pad_mask': doc_pad_mask,
                'segment_ids': segment_ids,
                'doc_pos_tok': doc_pos_tok,
                'cls_pos': cpos,
                'token_mask': token_mask,
            },
            'nsents': nsents,
            'target': None
        }

        # src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, segment_ids, cpos, nsents = self.create_batch(docs, idx, src_dict, max_sent_length=max_sent_len)
        doc_pos_tok = torch.LongTensor( doc_pad_mask.size() ).fill_(src_tokens[0, 0])
        doc_pos_tok[ doc_pad_mask ] = src_dict.pad()

        return {
            "id": id,
            "net_input": {
                "bsz": len(samples),
                "max_nsents": max(map(len, cls_pos)),
                "masked_docs_generator": masked_doc_gengerator(docs),
                "doc_without_mask": doc_without_mask
            }
        }
    # need to be fixed
    # right padding sentence [cls] w1 w2 ... [sep]
    def doc2tensor(self, docs_, vocab):
        import itertools
        # docs_ to a flatten docs
        docs = []
        for doc_ in docs_:
            doc = []
            cls_pos = []
            for sent in doc_:
                cls_pos.append(len(doc))
                doc.extend(sent)
            docs.append( (doc, cls_pos) )

        pad_idx = vocab.pad()
        sep_idx = self.sent_sep_idx

        bsz = len(docs)
        max_doc_len = 0
        max_nsent = 0
        for doc, cls_pos in docs:
            max_doc_len = max(len(doc), max_doc_len)
            max_nsent = max(max_nsent, len(cls_pos))
        # print('max_doc_len', max_doc_len, ' max_nsent', max_nsent)

        src_tokens = torch.LongTensor(bsz, max_doc_len).fill_(pad_idx)
        segment_ids = torch.LongTensor(bsz, max_doc_len).fill_(0)
        cpos = torch.LongTensor(bsz, max_nsent).fill_(0)
        try:
            doc_pad_mask = torch.BoolTensor(bsz, max_nsent).fill_(1)
        except:
            doc_pad_mask = torch.ByteTensor(bsz, max_nsent).fill_(1)

        nsents = []
        for i, item in enumerate(docs):
            doc, cls_pos = item
            cls_len = len(cls_pos)
            doc_pad_mask[i, 0:cls_len] = 0
            cpos[i, 0:cls_len] = torch.LongTensor(cls_pos)
            doc_len = len(doc)
            src_tokens[i, 0:doc_len] = torch.LongTensor(doc)
            nsents.append(cls_len)

        return src_tokens, doc_pad_mask, segment_ids, cpos, nsents

    def mask_sentences(self, index, docs, masked_sent_prob, max_predictions_per_doc, vocab, mask_idx):

        doc = docs[index]
        assert isinstance(mask_idx, int) or isinstance(mask_idx, list)
        selected_indexes = [mask_idx] if isinstance(mask_idx, int) else mask_idx
        output_doc = list(doc)
        masked_sents = []

        for i in selected_indexes:
            try:
                masked_sent = [ self.sent_mask_idx ] * len(output_doc[i])
                masked_sent[0] = self.src_dict.cls_index
                masked_sent[-1] = self.sent_sep_idx
                output_doc[i] = masked_sent

                masked_sents.append( doc[i] )
            except IndexError:
                masked_sents.append([self.src_dict.cls_index, self.sent_sep_idx])

        return output_doc, selected_indexes, masked_sents


    def masked_sents2tensor(self, docs_selected_indexes, docs_masked_sents):
        bsz = len(docs_selected_indexes)
        max_nsent = max( [len(sel_idxs) for sel_idxs in docs_selected_indexes] )
        tgt_selected_indexes = torch.LongTensor(bsz, max_nsent).fill_(0)
        for i, sel_idxs in enumerate(docs_selected_indexes):
            si_len = len(sel_idxs)
            tgt_selected_indexes[i, 0:si_len] = torch.LongTensor(sel_idxs)

        max_nsent2, max_sent_len = 0, 0
        for masked_sents in docs_masked_sents:
            max_nsent2 = max( max_nsent2, len(masked_sents) )
            local_max_sent_len = max( map(len, masked_sents) )
            max_sent_len = max(max_sent_len, local_max_sent_len)

        assert max_nsent == max_nsent2

        tgt_input_masked_sents = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(self.src_dict.pad())
        tgt_input_masked_sents[:, :, 0] = self.src_dict.cls_index
        tgt_masked_sents = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(self.src_dict.pad())
        for i, masked_sents in enumerate(docs_masked_sents):
            for j, sent in enumerate(masked_sents):
                assert sent[0] == self.src_dict.cls_index
                # print( 'after truncate? : ', self.src_dict.string_complete(sent), len(sent) )
                sent = sent[1:]
                sent_len = len(sent)
                assert sent[-1] == self.sent_sep_idx
                # sent[-1] = self.src_dict.eos()
                tgt_input_masked_sents[i, j, 1:sent_len] = torch.LongTensor(sent[0:-1])
                tgt_masked_sents[i, j, 0:sent_len] = torch.LongTensor(sent)

        return tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents


    def create_batch(self, docs, idx, vocab, max_sent_length=None):
  
        # create masked sentence
        new_docs = []
        docs_selected_indexes = []
        docs_masked_sents = []
        for i in range(len(docs)):
            new_doc, selected_indexes, masked_sents = self.mask_sentences(i, docs,
                                                        self.masked_sent_prob,
                                                        self.max_predictions_per_doc,
                                                        vocab, idx)
            new_docs.append(new_doc)
            docs_selected_indexes.append(selected_indexes)
            docs_masked_sents.append(masked_sents)

        # print('mask sentences done!')
        # doc to tensor
        # src_tokens, doc_pad_mask = self.doc2tensor(new_docs, cls_poses, vocab)
        # after masking, cls_poses may change!
        src_tokens, doc_pad_mask, segment_ids, cpos, nsents = self.doc2tensor(new_docs, vocab)
        '''
        for i in range(src_tokens.size(0)):
            print( vocab.string_complete(src_tokens[i]) )
            print(src_tokens[i].size())
        '''

        # print('cpos', cpos.size())

        # get masked sentences
        tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents = self.masked_sents2tensor(docs_selected_indexes, docs_masked_sents)

        return src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, segment_ids, cpos, nsents


    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        # nsent, max_sent_len = self.src_doc_sizes[index]
        # return nsent * min(self.max_sent_len, max_sent_len)

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
        # if self.align_dataset is not None:
        #     self.align_dataset.prefetch(indices)