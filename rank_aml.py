#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import collections
import itertools
import os, sys
import math
import torch
import numpy

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def main(args):
    # we should not do this!
    '''
    if args.max_tokens is None:
        args.max_tokens = 6000
    '''
    utils.xpprint(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    utils.xprintln('setup task done!')

    # Load dataset splits
    load_dataset_splits(args, task, ['train'])
    valid_dataset = args.valid_subset.split(',')
    load_dataset_splits(args, task, valid_dataset, shuffle=False)
    utils.xprintln('load dataset done!')

    if args.task.startswith('extractive_summarization'):
        if distributed_utils.is_master(args):
            from sum_eval import MultiProcSumEval
            sum_eval_pool = MultiProcSumEval(args.ncpu_eval)
            sum_valid_pool_params = dict(article_file=args.raw_valid + '.article',
                                          summary_file=args.raw_valid + '.summary',
                                          entity_map_file=None,
                                          length=-1, eval_type='predict',
                                          topk=args.topk_sent_eval, rerank=False, with_m=False,
                                          cmd='-a -c 95 -m -n 4 -w 1.2',
                                          trigram_block=args.trigram_block,)

            sum_test_pool_params = dict(article_file=args.raw_test + '.article',
                                          summary_file=args.raw_test + '.summary',
                                          entity_map_file=None,
                                          length=-1, eval_type='predict',
                                          topk=args.topk_sent_eval, rerank=False, with_m=False,
                                          cmd='-a -c 95 -m -n 4 -w 1.2',
                                          trigram_block=args.trigram_block,)
            sum_pool_params = dict(valid=sum_valid_pool_params, test=sum_test_pool_params)

            def make_params(default_dict, result_file, out_rouge_file, rerank=False, with_m=False):
                para_dict = dict(default_dict)
                para_dict['result_file'] = result_file
                para_dict['out_rouge_file'] = out_rouge_file
                para_dict['rerank'] = rerank
                para_dict['with_m'] = with_m
                return para_dict

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))
    # print(model)
    import sys
    sys.stdout.flush()

    # if summarization try to load pretrained model
    # if args.task.startswith('extractive_summarization') or args.task == 'pretrain_document_modeling':
    #     # assume this is a single GPU program
    if args.init_from_pretrained_doc_model:
        task.load_pretrained_model(model, args.pretrained_doc_model_path)
    sys.stdout.flush()

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    max_positions = trainer.get_model().max_positions()
    epoch_itr = trainer.get_train_iterator(epoch=0, load_dataset=False)

    # Load the latest checkpoint if one is available
    # load_checkpoint(args, trainer, epoch_itr)
    # make sure training from a different checkpoint will use different random seed
    cur_dataset = task.dataset('train')
    if hasattr(cur_dataset, 'rng'):
        print('epoch ', epoch_itr.epoch)
        cur_dataset.rng = numpy.random.RandomState(args.seed+epoch_itr.epoch)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    for alpha in range(10, 9, -1):
        # train for one epoch
        # train(args, trainer, task, epoch_itr)

        epoch_itr.next_epoch_itr()

        if epoch_itr.epoch % args.validate_interval == 0:
            if args.task.startswith('extractive_summarization'):
                if distributed_utils.is_master(args):
                    validate_metric(args, trainer, task, epoch_itr, valid_subsets)
                    # not compute rouge on the aml
                    # for subset in valid_subsets:
                    #     print(subset)
                    #     valid_result_file = os.path.join(args.save_dir, '{}.{}.txt'.format(epoch_itr.epoch, subset))
                    #     print(valid_result_file)
                    #     valid_out_file = os.path.join(args.save_dir, '{}.{}'.format(epoch_itr.epoch, subset))
                    #     sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, False, False))
                    #     # sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, True, False))

def init_output_file(out_put_file, dictionary):

    fout = open(out_put_file, 'w', encoding='utf8')
    # firstly, output dictionary information
    fout.write('%d\n'%len(dictionary))
    for i in range(len(dictionary)):
        fout.write('{}\t{}\n'.format(dictionary[i], i))
        fout.flush()
    return fout


def validate_metric(args, trainer, task, epoch_itr, subsets):
    # when training with distributed trainer, only one of them (the one args.distributed_rank == 0) is working ...
    print('args.distributed_rank', args.distributed_rank)
    print('args.distributed_world_size', args.distributed_world_size)
    if not distributed_utils.is_master(args):
        return

    """Evaluate the model on the validation set(s) and return the losses."""
    for subset in subsets:

        model_output_placeholder = os.path.join(args.save_dir, '{}.{}.txt'.format('placeholder', subset))
        model_output_file_list = []

        # fout = open(model_output_file, 'w', encoding='utf8')
        # # firstly, output dictionary information
        # fout.write('%d\n'%len(task.target_dictionary))
        # for i in range(len(task.target_dictionary)):
        #     fout.write('{}\t{}\n'.format(task.target_dictionary[i], i))
        #     fout.flush()

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            # max_positions=trainer.get_model().max_positions(),
            max_positions=None,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=1,
            shard_id=0,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )
        cnt = 0
        for sample in progress:
            preds = []
            scores = []
            trainer.model.eval()
            sample = utils.move_to_cuda(sample)
            # net_output = trainer.model(args.lam1, args.lam2, args.transpose_method, **sample['net_input'])
            with torch.no_grad():
                net_output = trainer.model(**sample['net_input'])
            # probs = trainer.model.get_normalized_probs(net_output, log_probs=False)
            # _, pred = probs.max(2)
            if isinstance(net_output[0], list):
                if len(model_output_file_list) < len(net_output[0]):
                    for idx, sub_net_output in enumerate(net_output[0]):
                        model_output_file_list.append(init_output_file(model_output_placeholder.replace('placeholder', str(idx)), task.target_dictionary))
                for sub_net_output, sub_score in zip(net_output[0], net_output[1]):
                    preds.append(sub_net_output)
                    scores.append(sub_score)
            else:
                if len(model_output_file_list) == 0:
                    model_output_file_list.append(init_output_file(model_output_placeholder.replace('placeholder', '1'), task.target_dictionary))
                preds.append(net_output[0])
                scores.append(net_output[1])

            if sample.get('target', None) is not None:
                target = trainer.model.get_targets(sample, net_output)
                if target.size(1) > preds[0].size(1):
                    target = target[:, : preds[0].size(1)]
            else:
                target = torch.ones_like(preds[0])
            target = torch.where(preds[0]==0, torch.zeros_like(preds[0]), target.int())
            assert len(preds) == len(scores) == len(model_output_file_list)
            for pred, score, fout in zip(preds, scores, model_output_file_list):
                for i in range(pred.size(0)):
                    labels = []
                    pred_labels = []
                    pred_dists = []
                    pred_scores = []
                    for j in range(pred.size(1)):
                        if target[i, j] != task.target_dictionary.pad():
                            labels.append( task.target_dictionary[target[i, j]] )
                            pred_labels.append( task.target_dictionary[pred[i, j]] )
                            pred_scores.append( str(round(score[i, j].item(), 5)) )
                            # pred_dists.append( ' '.join( map(lambda x: str(x.item()), probs[i, j]) ) )d
                        else:
                            break
                    fout.write('True      Labels:\t%s\n'%' '.join(labels))
                    fout.write('Predicted Labels:\t%s\n'%' '.join(pred_labels))
                    fout.write('Score:\t%s\n'%' '.join(pred_scores))
                    fout.write('Predicted Distri:\t%s\n'%' | '.join(pred_dists))
                    fout.flush()
            assert cnt == sample['id'][0]
            cnt += sample['id'].shape[0]

        for fout in model_output_file_list:
            fout.close()
            utils.xprintln('valid metric %s done!'%fout.name)


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
        end_of_epoch and not args.no_epoch_checkpoints and
        epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
        not end_of_epoch and args.save_interval_updates > 0 and
        updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
        val_loss is not None and
        (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


def load_dataset_splits(args, task, splits, shuffle=True):
    for split in splits:
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            try:
                task.load_dataset(split_k, shuffle)
                print('| {} {} {} examples'.format(args.data, split_k, len(task.dataset(split_k))))
            except FileNotFoundError as e:
                if k > 0:
                    break
                raise e


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    # if args.distributed_init_method is None:
    #     distributed_utils.infer_init_method(args)

    # if args.distributed_init_method is not None:
    #     # distributed training
    #     if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
    #         start_rank = args.distributed_rank
    #         args.distributed_rank = None  # assign automatically
    #         torch.multiprocessing.spawn(
    #             fn=distributed_main,
    #             args=(args, start_rank),
    #             nprocs=torch.cuda.device_count(),
    #         )
    #     else:
    #         distributed_main(args.device_id, args)
    # elif args.distributed_world_size > 1:
    #     # fallback for single node with multiple GPUs
    #     assert args.distributed_world_size <= torch.cuda.device_count()
    #     port = random.randint(10000, 20000)
    #     args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
    #     args.distributed_rank = None  # set based on device id
    #     if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
    #         print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
    #     torch.multiprocessing.spawn(
    #         fn=distributed_main,
    #         args=(args, ),
    #         nprocs=args.distributed_world_size,
    #     )
    # else:
    #     # single GPU training
    main(args)


if __name__ == '__main__':
    cli_main()
