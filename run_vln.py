# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import copy, time, json

import sys
sys.path.insert(0, '.')
import csv
import json
import math
import base64
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import _pickle as cPickle

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

from oscar.utils.misc import set_seed
from oscar.utils.task_utils import (_truncate_seq_pair, convert_examples_to_features_vqa,
                        output_modes, processors)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification, BertTokenizer),
}


log_json = []


def _load_dataset(args, name):
    processor = processors[args.task_name]()
    if name == 'train':
        examples = processor.get_train_examples(args.data_dir, 'R2R_train.json')
    elif name == 'val':
        examples = processor.get_dev_examples(args.data_dir, 'R2R_val_seen.json')
    elif name == 'test':  # test-submission
        examples = processor.get_test_examples(args.data_dir, 'R2R_test.json')
    return examples

class VLNDataset(Dataset):
    """ VLN Dataset """

    def __init__(self, args, name, tokenizer):
        super(VLNDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        self.output_mode = output_modes[args.task_name]
        self.tokenizer = tokenizer
        self.args = args
        self.name = name

        csv.field_size_limit(sys.maxsize)
        img_features = {}
        TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
        VIEWPOINT_SIZE = 36
        FEATURE_SIZE = 2048
        infile = '/home/xuefei/Oscar/datasets/R2R/ResNet-152-imagenet.tsv'
        with open(infile, "rt") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=TSV_FIELDNAMES)
            for item in reader:
                viewpoint_id = item['viewpointId']
                item['scanId'] = item['scanId']
                item['viewpointId'] = item['viewpointId']
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['vfov'] = int(item['vfov'])
                item['features'] = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(
                    (VIEWPOINT_SIZE, FEATURE_SIZE))
                position_array = np.zeros((6, 36))
                item['features'] = np.concatenate((item['features'], position_array.T), axis=1)
                img_features[viewpoint_id] = item

        self.img_features = img_features
        self.examples = _load_dataset(args, name)
        logger.info((len(self.img_features)))
        logger.info('%s Data Examples: %d' % (name, len(self.examples)))

    def tensorize_example(self, example, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):

        tokens_a = self.tokenizer.tokenize(example.text_a)

        tokens_b = None

        # original
        #if example.text_b:
        #    txt_b = example.text_b.replace(';', ' ').strip()
        #    tokens_b = self.tokenizer.tokenize(txt_b)
        #    _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.args.max_seq_length - 2:
            tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # image features
        img_feat = self.img_features[example.img_key]  # torch
        img_feat = torch.from_numpy(img_feat)
        # img_feat = self.img_features.item().get(example.img_key)  # numpy
        if img_feat.shape[0] > self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length, ]
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        if self.args.output_mode == "classification":
            if (example.label is None):
                label_id = [0]
                score = [0]
            elif len(example.label) == 0:
                label_id = [0]
                score = [0]
            else:
                label_id = [example.label]
                score = [example.score]
        elif self.args.output_mode == "regression":
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            #img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)

        return (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(input_mask, dtype=torch.long),
                    torch.tensor(segment_ids, dtype=torch.long),
                    torch.tensor([label_id[0]], dtype=torch.long),
                    img_feat,)

    def __getitem__(self, index):
        if self.args.load_fast:
            example = self.features[index]
        else:
            entry = self.examples[index]
            example = self.tensorize_example(entry,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example

    def __len__(self):
        return len(self.examples)

def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, sampler=train_sampler, batch_size=args.train_batch_size) #, collate_fn=trim_batch)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    for epoch in range(int(args.num_train_epochs)):
    #for epoch in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        total_loss = 0
        total_norm = 0
        count_norm = 0

        t_start = time.time()
        for step, batch in enumerate(train_dataloader):
        #for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
              'labels': batch[3],
              'img_feats': None if args.img_feature_dim == -1 else batch[4]}
            outputs = model(**inputs)

            #loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss, logits = outputs[:2]

            if args.n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                count_norm += 1

            #batch_score = compute_score_with_logits(logits, batch[4]).sum()
            #train_score += batch_score.item()

            tr_loss += loss.item()
            total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:# Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
                        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
                        if eval_score > best_score:
                            best_score = eval_score
                            best_model['epoch'] = epoch
                            best_model['model'] = copy.deepcopy(model)

                        logger.info("EVALERR: {}%".format(100 * best_score))
                    logging_loss = tr_loss

            #if args.max_steps > 0 and global_step > args.max_steps:
            #    epoch_iterator.close()
            #    break

        t_end = time.time()
        logger.info('Train Time Cost: %.3f' % (t_end-t_start))

        # evaluation
        logger.info("Epoch: %d" % (epoch))
        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)
            #best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())

        # save checkpoints
        if args.local_rank in [-1, 0] and args.save_epoch > 0 and epoch % args.save_epoch == 0: # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

            save_num = 0
            while (save_num < 10):
                try:
                    logger.info("Saving model attempt: {}".format(save_num))
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 'eval_score': eval_score, 'best_score':best_score}
        log_json.append(epoch_log)

        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))
        logger.info("EVALERR: {}%".format(100*best_score))
        logger.info("LOSS: {}%".format(total_loss / len(train_dataset)))

    with open(args.output_dir + '/eval_logs.json', 'w') as fp:
        json.dump(log_json, fp)

    if args.local_rank in [-1, 0]: # Save the final model checkpoint
        output_dir = os.path.join(args.output_dir, 'best-{}'.format(best_model['epoch']))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

        save_num = 0
        while (save_num < 10):
            try:
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                tokenizer.save_pretrained(output_dir)
                break
            except:
                save_num += 1
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    #if args.n_gpu > 1: model = torch.nn.DataParallel(model) # debug: single-gpu or multi-gpus

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        num_data = 0
        correct_num = 0

        for batch in eval_dataloader:
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3],
                          'img_feats':      None if args.img_feature_dim == -1 else batch[4]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

                #logger.info('logits: %s, batch[3]: %s' % (str(logits.shape), str(batch[3].shape)))
                #logger.info('correct: %s' % (str(logits.argmax(1) == batch[3].view(-1))))

                correct = logits.argmax(1) == batch[3].view(-1)
                correct_num += correct.sum().item()
                num_data += logits.size(0)

                # debug
                #val, idx = logits.max(1)
                #logger.info('idx: %s, batch[4]: %s' % (str(idx.shape), str(batch[3].shape)))
                #for i in range(idx.size(0)):
                #    logger.info('idx: %d, pred: %d, real: %d' % (idx[i].item(), eval_dataset.labels[idx[i].item()], batch[3][i].item()))

            nb_eval_steps += 1

        acc = float(correct_num) / len(eval_dataloader.dataset)

        logger.info("Eval Results:")
        logger.info("Eval Accuracy: %.3f" % (100*acc))
        logger.info("Eval Loss: %.3f" % (eval_loss))

    t_end = time.time()
    logger.info('Eva Time Cost: %.3f' % (t_end - t_start))

    return results, acc

def test(args, model, eval_dataset=None, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    label2ans = cPickle.load(open(args.label2ans_file, 'rb'))
    logger.info('label2ans: %d' % (len(label2ans)))

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval
        logger.info("***** Running Test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        for batch in eval_dataloader:
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         None,
                          'img_feats':      None if args.img_feature_dim == -1 else batch[4]}
                outputs = model(**inputs)
                logits = outputs[0]

                val, idx = logits.max(1)
                #logger.info('idx: %s, batch[6]: %s' % (str(idx.shape), str(batch[6].shape)))

                for i in range(idx.size(0)):
                    result = {}
                    result['questionId'] = str(batch[6][i].item())
                    result['prediction'] = label2ans[eval_dataset.labels[idx[i].item()]]
                    results.append(result)

                    #logger.info('q_id: {0}, answer: {1}'.format(result['question_id'], result['answer']))

    with open(args.output_dir + ('/{}_results.json'.format(eval_dataset.name)), 'w') as fp:
        json.dump(results, fp)

    t_end = time.time()
    logger.info('# questions: %d' % (len(results)))
    logger.info('Test Time Cost: %.3f' % (t_end - t_start))

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")

    parser.add_argument("--data_label_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--train_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--eval_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--loss_type", default='kl', type=str, help="kl or xe")

    parser.add_argument("--spatial_dim", default=6, type=int, help="spatial_dim")

    parser.add_argument("--max_label_pos_length", default=45, type=int, help="The maximum total input label position sequence length.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")

    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save checkpoint every X epochs.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")
    parser.add_argument("--load_fast", action='store_true', help="Load Tensor Fast")
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 1 --img_feature_dim 565 --img_feature_type dis_code '

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 10 --img_feature_dim 565 --img_feature_type other '

    #args = parser.parse_args(args.split())

    args = parser.parse_args()

    if args.philly:  # use philly
        logger.info('Info: Use Philly, all the output folders are reset.')
        args.output_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.output_dir)
        logger.info('OUTPUT_DIR:', args.output_dir)

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train: logger.info("Output Directory Exists.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=2, finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # discrete code
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.code_voc = args.code_voc
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.classifier = args.classifier
    config.cls_hidden_scale = args.cls_hidden_scale
    config.spatial_dim = args.spatial_dim

    # load discrete code
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Load discrete code from: {}'.format(args.data_dir))
        t_start = time.time()
        train_code = torch.load(os.path.join(args.data_dir, 'vqvae', 'train.pt'))
        t_end = time.time()
        logger.info('Load time: %.3f' % (t_end - t_start))

        if args.code_level == 'top':
            config.code_dim = train_code['embeddings_t'].shape[0]
            config.code_size = train_code['feats_top'][list(train_code['feats_top'].keys())[0]].shape[0]
        elif args.code_level == 'bottom':
            config.code_dim = train_code['embeddings_b'].shape[0]
            config.code_size = train_code['feats_bottom'][list(train_code['feats_bottom'].keys())[0]].shape[0]
        elif args.code_level == 'both':
            config.code_dim = train_code['embeddings_t'].shape[0] + train_code['embeddings_b'].shape[0]

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    if args.img_feature_type in ['dis_code', 'dis_code_t']:
        logger.info('Initializing the code embedding with {}'.format(args.code_level))
        if args.code_level == 'top':
            model.init_code_embedding(train_code['embeddings_t'].t())
        elif args.code_level == 'bottom':
            model.init_code_embedding(train_code['embeddings_b'].t())

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    #if args.do_eval:
    eval_dataset = VLNDataset(args, 'val', tokenizer, )
    #eval_dataset = GQADataset(args, 'test-dev', img_features, tokenizer) # test-dev as val

    if args.do_test:
        test_dataset = VLNDataset(args, 'test', tokenizer)

    # Training
    if args.do_train:
        train_dataset = VLNDataset(args, 'train',tokenizer)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]: os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`. They can then be reloaded using `from_pretrained()`
        #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        #model_to_save.save_pretrained(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        #model = model_class.from_pretrained(args.output_dir)
        #tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        #model.to(args.device)


    # Evaluation
    #results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result, score = evaluate(args, model, eval_dataset, prefix=global_step)
            #result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            #results.update(result)

    # Test-Submission
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            test(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()