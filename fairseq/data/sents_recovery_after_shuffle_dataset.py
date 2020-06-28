import copy
import numpy as np
import torch
import math

from . import data_utils, FairseqDataset

SENT_SEP = '</s>'
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

# right padding [cls] w1 w2 ... [sep] [cls] w1 w2 ... [sep] ...
def docs2tensor(docs, pad_idx, sep_id):
    bsz = len(docs)
    max_doc_len = 0
    max_nsent = 0
    for doc, cls_pos in docs:
        max_doc_len = max(len(doc), max_doc_len)
        max_nsent = max(max_nsent, len(cls_pos))

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

def create_src_tok_batch(samples, src_dict, max_doc_length=None, max_sent_length=None, rng=None, shuffle_prob=1):

    def shuffle_sents(sents, shuffle_prob):
        truncate_sents = []
        for sent in sents:
            if sum([len(s) for s in truncate_sents]) + len(sent) <= MAX_DOC_LEN:
                truncate_sents.append(sent)
            else:
                break
        
        # print('\n', 'perm')
        # print([len(doc) for doc in truncate_sents])
        shuffled_sents = [None] * len(truncate_sents)
        # shuffle the sents
        target_order = np.arange(len(truncate_sents))
        if rng.uniform() < shuffle_prob:
 
            rng.shuffle(target_order)
            for idx, order in enumerate(target_order):
                shuffled_sents[order] = sents[idx]
        else:
            shuffled_sents = copy.deepcopy(truncate_sents)
        
        return truncate_sents, shuffled_sents, target_order

    sep_id, cls_idx, pad_idx = src_dict.index(SENT_SEP), src_dict.cls_index, src_dict.pad()
    docs = []
    target_orders = []
    truncate_docs = []
    max_nsent = 0
    max_sent_len = 0
    for sample in samples:
        src = sample['source']
        sents = split_list(src, sep_id)

        if max_doc_length:
            sents = sents[:max_doc_length]

        if max_sent_length is not None:
            sents = [sent if len(sent) <= max_sent_length else sent[0:max_sent_length-1] + [sep_id] for sent in sents]

        sents = [[cls_idx] + sent for sent in sents]

        if sents[-1][-1] != sep_id:
            sents[-1].append(sep_id)

        truncate_sents, shuffled_sents, target_order = shuffle_sents(sents, shuffle_prob)

        shuffle_doc = []
        cls_pos = []
        for sent in shuffled_sents:
            if len(shuffle_doc) + len(sent) <= MAX_DOC_LEN:
                cls_pos.append(len(shuffle_doc))
                shuffle_doc.extend(sent)
            else:
                break

        docs.append( (shuffle_doc, cls_pos) )
        target_orders.append(target_order)
        truncate_docs.append(truncate_sents)

    return docs2tensor(docs, pad_idx, sep_id) + (target_orders, truncate_docs)


class SentsRecoveryAfterShuffle(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        max_sent_len=None,
        max_doc_len=None,
        shuffle_prob=None,
        rng=None,
        doc_sizes=None,
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
        self.shuffle_prob = shuffle_prob
        self.masked_sent_prob = 0.15
        self.min_predictions_per_doc = 1
        self.rng =  rng

        self.sent_sep_idx = self.src_dict.index(SENT_SEP)
        print(SENT_SEP, self.sent_sep_idx)
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
            samples, self.src_dict,
            max_doc_len=self.max_doc_len
        )

    def collate(self, samples, src_dict, max_doc_len=None):
        def merge_target(tgt1, tgt2, fill_idx):
            assert tgt1.shape[1] >= tgt2.shape[1]
            tgt = torch.cat([tgt1, tgt1.clone().fill_(fill_idx)], dim=0)
            tgt[-tgt2.shape[0]:, :tgt2.shape[1]] = tgt2
            return tgt
        
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens, doc_pad_mask, segment_ids, cpos, nsents, tgt_orders, truncate_sents = create_src_tok_batch(samples, src_dict, max_doc_length=max_doc_len, max_sent_length=self.max_sent_len+1, rng=self.rng, shuffle_prob=self.shuffle_prob)
        tgt_selected_indexes, tgt_input_sents, tgt_sents = self.create_target(nsents, truncate_sents)
        src_tokens_with_mask, doc_pad_mask_2, tgt_selected_indexes_mask, tgt_input_masked_sents, tgt_masked_sents, segment_ids_2, cpos_mask, nsents_2 = self.create_batch(samples, truncate_sents, src_dict, max_sent_length=self.max_sent_len)

        assert doc_pad_mask.equal(doc_pad_mask_2)
        assert segment_ids.equal(segment_ids_2)
        assert nsents == nsents_2

        # src_tokens = torch.cat([src_tokens, src_tokens_with_mask], dim=0)
        # doc_pad_mask = torch.cat([doc_pad_mask, doc_pad_mask_2], dim=0)
        # segment_ids = torch.cat([segment_ids, segment_ids_2], dim=0)
        # cpos = torch.cat([cpos, cpos_mask], dim=0)
        # tgt_selected_indexes = merge_target(tgt_selected_indexes, tgt_selected_indexes_mask, 0)
        # tgt_input_sents = merge_target(tgt_input_sents, tgt_input_masked_sents, self.src_dict.pad())
        # tgt_sents = merge_target(tgt_sents, tgt_masked_sents, self.src_dict.pad())

        # src_tokens, doc_pad_mask, segment_ids = create_src_tok_batch(samples, src_dict.index(SENT_SEP), src_dict.cls_index, src_dict.pad(), max_sent_length=max_sent_len)
        '''
        print('***********************************')
        for i in range(src_tokens.size(0)):
            print( src_dict.string_complete(src_tokens[i]) )
        print('***********************************')
        '''

        # simply add a sepecial token
        doc_pos_tok = torch.LongTensor( doc_pad_mask.size() ).fill_(src_tokens[0, 0])
        doc_pos_tok[ doc_pad_mask ] = src_dict.pad()

        # ntokens = sum(len(s['target']) for s in samples)
        # ntokens = sum(nsents)
        # ntokens = tgt_sents.ne( self.src_dict.pad() ).sum().item() + tgt_masked_sents.ne( self.src_dict.pad() ).sum().item()
        ntokens =  tgt_masked_sents.ne( self.src_dict.pad() ).sum().item()
        shuffle_ntokens = tgt_sents.ne( self.src_dict.pad() ).sum().item()

        return {
            'id': id,
            'ntokens': ntokens,
            'shuffle_ntokens': shuffle_ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_tokens_with_mask': src_tokens_with_mask,
                'doc_pad_mask': doc_pad_mask,
                'segment_ids': segment_ids,
                'doc_pos_tok': doc_pos_tok,
                'cls_pos': cpos,
                'cls_mask_pos': cpos_mask,
                'sent_positions': tgt_selected_indexes,
                'masked_sent_positions': tgt_selected_indexes_mask,
                'prev_output_tokens': tgt_input_sents,
                'prev_mask_output_tokens': tgt_input_masked_sents,
            },
            'target': tgt_sents,
            'mask_target': tgt_masked_sents
        }

    def create_target(self, nsents, docs_masked_sents):
        docs_selected_indexes = [list(range(n)) for n in nsents]
        bsz = len(docs_selected_indexes)
        # max_nsent = max( [len(sel_idxs) for sel_idxs in docs_selected_indexes] )
        max_nsent = max(nsents)
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

        tgt_input_sents = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(self.src_dict.pad())
        tgt_input_sents[:, :, 0] = self.src_dict.cls_index
        tgt_sents = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(self.src_dict.pad())
        for i, masked_sents in enumerate(docs_masked_sents):
            for j, sent in enumerate(masked_sents):
                assert sent[0] == self.src_dict.cls_index
                # print( 'after truncate? : ', self.src_dict.string_complete(sent), len(sent) )
                sent = sent[1:]
                sent_len = len(sent)
                assert sent[-1] == self.sent_sep_idx
                # sent[-1] = self.src_dict.eos()
                tgt_input_sents[i, j, 1:sent_len] = torch.LongTensor(sent[0:-1])
                tgt_sents[i, j, 0:sent_len] = torch.LongTensor(sent)

        return tgt_selected_indexes, tgt_input_sents, tgt_sents


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

    def mask_sentences(self, index, docs, masked_sent_prob, max_predictions_per_doc=math.inf, vocab=None):
        def get_rnd_sent(index, docs):
            rnd_idx = -1
            for i in range(10):
                rnd_idx = self.rng.randint(0, len(docs))
                if rnd_idx != index:
                    break
            sampled_doc = docs[rnd_idx]

            return sampled_doc[ self.rng.randint(0, len(sampled_doc)) ]

        doc = docs[index]
        candi_indexes = list(range(len(doc)))
        self.rng.shuffle(candi_indexes)
        num_pred = min( max(self.min_predictions_per_doc, int(len(candi_indexes) * masked_sent_prob)),
                        max_predictions_per_doc )

        assert len(candi_indexes[0:num_pred]) == len(set(candi_indexes[0:num_pred]))

        output_doc = list(doc)
        selected_indexes = candi_indexes[0:num_pred]
        selected_indexes.sort()
        masked_sents = []

        for i in selected_indexes:
            if self.rng.uniform() < 0.8:
                masked_sent = [ self.sent_mask_idx ] * len(output_doc[i])
                masked_sent[0] = self.src_dict.cls_index
                masked_sent[-1] = self.sent_sep_idx
                output_doc[i] = masked_sent
            else:
                if self.rng.uniform() < 0.5:
                    output_doc[i] = doc[i]
                else:
                    rnd_sent = get_rnd_sent(index, docs)
                    ori_len = len(doc[i])
                    if len(rnd_sent) > ori_len:
                        rnd_sent = rnd_sent[0:ori_len-1] + [rnd_sent[-1]]
                    elif len(rnd_sent) < ori_len:
                        rnd_sent = rnd_sent[:-1] + [self.sent_mask_idx] * (ori_len - len(rnd_sent)) + [self.sent_sep_idx]
                    '''
                    print('rnd_sent', rnd_sent)
                    print(len(rnd_sent))
                    '''

                    output_doc[i] = rnd_sent
            masked_sents.append( doc[i] )

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


    def create_batch(self, samples, docs, vocab, max_sent_length=None):
        # split sample into documents
        # docs, cls_poses = get_docs(samples, vocab, max_sent_length+1)
        # create masked sentence
        new_docs = []
        docs_selected_indexes = []
        docs_masked_sents = []
        for i in range(len(docs)):
            new_doc, selected_indexes, masked_sents = self.mask_sentences(i, docs,
                                                        self.masked_sent_prob,
                                                        math.inf,
                                                        vocab)
            new_docs.append(new_doc)
            docs_selected_indexes.append(selected_indexes)
            docs_masked_sents.append(masked_sents)

        # print('mask sentences done!')
        # doc to tensor
        # src_tokens, doc_pad_mask = self.doc2tensor(new_docs, cls_poses, vocab)
        # after masking, cls_poses may change!
        src_tokens, doc_pad_mask, segment_ids, cpos, nsents = self.doc2tensor(new_docs, vocab)
        '''
        print('\n', 'mask')
        print([len(doc) for doc in sum(docs, [])])
        print([len(doc) for doc in sum(new_docs, [])])
        '''
        '''
        for i in range(src_tokens.size(0)):
            print( vocab.string_complete(src_tokens[i]) )
            print(src_tokens[i].size())
        '''

        # print('cpos', cpos.size())

        # get masked sentences
        tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents = self.masked_sents2tensor(docs_selected_indexes, docs_masked_sents)

        return src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, segment_ids, cpos, nsents

    def get_dummy_batch(self, num_docs, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        # src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        '''
        bsz = num_docs

        def create_tgt():
            return torch.LongTensor([self.tgt_dict.index('F')] * self.max_doc_len)

        def create_src():
            doc = []
            for i in range(self.max_doc_len):
                for j in range(self.max_sent_len):
                    doc.append(self.src_dict.unk())
                if i != self.max_doc_len-1:
                    doc.append(self.sent_sep_idx)
            return torch.LongTensor(doc)
        '''
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

        orig_min_predictions_per_doc = self.min_predictions_per_doc
        # self.min_predictions_per_doc = self.max_predictions_per_doc
        batch = self.collater([
            {
                'id': i,
                'source': create_src(),
                'target': create_tgt(),
            }
            for i in range(bsz)
        ])
        # self.min_predictions_per_doc = orig_min_predictions_per_doc
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
        # if self.align_dataset is not None:
        #     self.align_dataset.prefetch(indices)
