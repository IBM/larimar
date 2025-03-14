import numpy as np
import os, sys
import torch
from torch import nn, optim
import subprocess
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, RandomSampler
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from modules import MemVAE
#from torchmetrics.text import Perplexity
import nltk

import json
import pdb

import torch.nn.init as init

import glob
import logging
import pickle
import random
from torch.utils.data.distributed import DistributedSampler
from nltk.tokenize import wordpunct_tokenize

logger = logging.getLogger(__name__)



class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class BucketSamplerr(Sampler):
    def __init__(self, lens, bucket_size, batch_size, drop_last=False, shuffle=True):
        # super().__init__()
        self._lens = lens
        self.batch_size = batch_size
        self._bucket_size = bucket_size
        self.drop_last = drop_last
        self._shuf = shuffle

    def __iter__(self):
        breakpoint()
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)

        buckets = [sorted(ids[i:i+self._bucket_size], key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]

        # A = []
        # for bucket in buckets:
        #     A.append([])
        #     for b in bucket:
        #         A[-1].append(self._lens[b])

        # buckets = [ids[i:i+self._bucket_size] for i in range(0, len(ids), self._bucket_size)]          
        # batches = [bucket[i:i+self.batch_size]
        #            for bucket in buckets
        #            for i in range(0, len(bucket), self.batch_size)]

        batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                batches.append(batch)

        # batchesA = []
        # for bucket in A:
        #     for i in range(0, len(bucket), self.batch_size):
        #         batch = bucket[i:i + self.batch_size]
        #         batchesA.append(batch)

        if self.drop_last:
            batches = [batch for batch in batches
                       if len(batch) == self.batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self.drop_last:
            return sum(s//self.batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self.batch_size) for s in bucket_sizes)


class FeatureDataset(Dataset):
    def __init__(self, features, max_len=None):
        self.features = features
        self.max_len = max_len  # this max_len do truncate

    def __getitem__(self, i):
        feat_dict = self.features[i]
        feat = InputFeatures(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids_bert = pad_sequence([torch.tensor(f.input_ids_bert, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        input_ids_gpt = pad_sequence([torch.tensor(f.input_ids_gpt, dtype=torch.long) for f in features], batch_first=True, padding_value=0)
        lm_labels = pad_sequence([torch.tensor(f.input_ids_gpt, dtype=torch.long) for f in features], batch_first=True, padding_value=-1)
        return (input_ids_bert, input_ids_gpt, lm_labels)


class BucketingDataLoaderPL(object):
    def __init__(self,
                 file_path,
                 batch_size,
                 max_seq_length,
                 tokenizer,
                 block_size,
                 use_labels,
                 dataset,
                 use_philly,
                 num_chunk,
                 num_data_workers,
                 batches_per_bucket=100,
                 shuffle=True,
                 perturb=""):
        ext = ""
        if perturb:
            ext = perturb + "_"
        self.dataset = TokenDataset(tokenizer,
                                    use_labels,
                                    dataset,
                                    use_philly,
                                    int(num_chunk),
                                    file_path,
                                    text_split_mode='natural',
                                    block_size=block_size,
                                    ext=ext)
        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = batches_per_bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]
        self.num_data_workers = num_data_workers

    def get(self):

        if self.bucket_size > 0:
            # if shuffle = True, data reshuffling relies on dataloader reloading on new epoch
            self.dataset.create_buckets_and_shuffle(self.bucket_size, self.shuffle)

        loader = DataLoader(self.dataset,
                            num_workers=self.num_data_workers,
                            pin_memory=True,
                            collate_fn=TokenDataset.collate,
                            batch_size=self.batch_size,
                            shuffle=False,  # here shuffle is always false
                            drop_last=True)
        return loader



class BucketingDataLoader(object):
    def __init__(self,
                 file_path,
                 batch_size,
                 max_seq_length,
                 tokenizer,
                 block_size,
                 use_labels,
                 dataset,
                 use_philly,
                 bucket=100,
                 shuffle=True,
                 perturb=""):
        ext = ""
        if perturb:
            ext = perturb + "_"
        self.dataset = TokenDataset(tokenizer,
                                    use_labels,
                                    dataset,
                                    use_philly,
                                    file_path,
                                    text_split_mode='natural',
                                    block_size=block_size,
                                    ext=ext)
        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]

    def __iter__(self):
        sampler = BucketSampler(self.example_lengths, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=TokenDataset.collate)
        yield from loader

    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass


class Dialog_BucketingDataLoader(object):
    def __init__(self, file_path, batch_size, max_seq_length, tokenizer, args, bucket=100, shuffle=True):

        self.dataset = Dialog_TokenDataset(tokenizer, args, file_path, block_size=args.block_size)
        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]

    def __iter__(self):
        sampler = BucketSampler(self.example_lengths, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=Dialog_TokenDataset.collate)
        yield from loader

    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass



class MultipleFiles_DataLoader(object):
    def __init__(self, file_path, batch_size, max_seq_length, tokenizer, args, bucket=100, shuffle=True, use_tensor=True):


        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.args = args
        self.use_tensor=use_tensor

        # prepare for the first file
        self.file_idx = 0
        self.cached_features_file = os.path.join(self.file_path, args.dataset.lower()+f'.segmented.nltk.split.seq64.{self.file_idx}.json' )
        self.dataset = PreparedTokenDataset(tokenizer, self.args, self.cached_features_file, block_size=self.args.block_size)
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]


    def __iter__(self):
        
        sampler = BucketSampler(self.example_lengths, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=PreparedTokenDataset.collate if self.use_tensor else PreparedTokenDataset.get_examples )
        yield from loader

        # update file name for next file
        self.file_idx += 1
        self.cached_features_file = os.path.join(self.file_path, self.args.dataset.lower()+f'.segmented.nltk.split.seq64.{self.file_idx}.json' )
        self.dataset = PreparedTokenDataset(self.tokenizer, self.args, self.cached_features_file, block_size=self.args.block_size)
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//self.batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]


    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass

    def reset(self):
        self.file_idx = 0


# When the dataset is too big, we can divide it into multiple small files.
# This class is used load multiple files.
class BucketingMultipleFiles_DataLoader(object):
    def __init__(self, file_path, batch_size, max_seq_length, tokenizer, args, bucket=100, shuffle=True):

        self.batch_size = batch_size
        self.max_len = max_seq_length
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.args = args

        # prepare for the first file
        self.file_idx = 0
        self.cached_features_file = os.path.join(self.file_path, args.dataset.lower()+f'.segmented.nltk.split.seq64.{self.file_idx}.json' )
        self.dataset = PreparedTokenDataset(tokenizer, self.args, self.cached_features_file, block_size=self.args.block_size)
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]


    def __iter__(self):
        
        # sampler = BucketSampler(self.example_lengths, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        # loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=PreparedTokenDataset.collate)

        # distributed
        if self.args.distributed:
            sampler = DistributedSampler(self.dataset)
        else:
            sampler = RandomSampler(self.dataset)
        loader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size, pin_memory=True, num_workers=0, collate_fn=PreparedTokenDataset.collate)
        yield from loader

        # update file name for next file
        self.file_idx += 1
        self.cached_features_file = os.path.join(self.file_path, self.args.dataset.lower()+f'.segmented.nltk.split.seq64.{self.file_idx}.json' )
        self.dataset = PreparedTokenDataset(self.tokenizer, self.args, self.cached_features_file, block_size=self.args.block_size)
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//self.batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]


    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass

    def reset(self):
        self.file_idx = 0

    def set(self, idx):
        self.file_idx = idx
        self.cached_features_file = os.path.join(self.file_path, self.args.dataset.lower()+f'.segmented.nltk.split.seq64.{self.file_idx}.json' )
        self.dataset = PreparedTokenDataset(self.tokenizer, self.args, self.cached_features_file, block_size=self.args.block_size)
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples//self.batch_size
        self.example_lengths = [example['bert_token_length'] for example in self.dataset.examples]

class PreparedTokenDataset(Dataset):
    def __init__(self, tokenizers, args, cached_features_file='train', text_split_mode='natural', block_size=512):
        logger.info(cached_features_file)
        assert os.path.isfile(cached_features_file)

        self.examples = []
        self.tokenizers = tokenizers

        # Bert tokenizer special tokens
        self.bert_pad_token=tokenizers[0].convert_tokens_to_ids([tokenizers[0].pad_token])[0]

        # GPT-2 tokenizer special tokens
        self.gpt2_pad_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].pad_token])[0]
        self.gpt2_bos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].bos_token])[0]
        self.gpt2_eos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].eos_token])[0]

        global bert_pad_token
        global gpt2_pad_token
        bert_pad_token = self.bert_pad_token
        gpt2_pad_token = self.gpt2_pad_token

        if args.dataset == 'Yahoo' or args.dataset == 'Penn' or args.dataset == 'Snli' or args.dataset == 'Debug' or args.dataset == 'wikipedia':
            label_on = False
        elif args.dataset == 'Yelp':
            label_on = True

        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, 'r') as handle:
            self.examples = json.load(handle)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


    @staticmethod
    def get_examples(examples):
        token_lengths = torch.tensor( [[f['bert_token_length'], f['gpt2_token_length']] for f in examples] , dtype=torch.long)
        return examples, token_lengths


    @staticmethod
    def collate(examples):
        # Convert to Tensors and build dataset
        input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=bert_pad_token)
        input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=gpt2_pad_token)
        token_lengths = torch.tensor( [[f['bert_token_length'], f['gpt2_token_length']] for f in examples] , dtype=torch.long)

        return (input_ids_bert, input_ids_gpt, token_lengths)


class TokenDataset(Dataset):
    def __init__(self,
                 tokenizers,
                 use_labels,
                 dataset,
                 use_philly,
                 num_chunk,
                 file_path='train',
                 text_split_mode='natural',
                 block_size=512,
                 ext=""):

        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)

        if 'test' in filename:
            num_chunk=False

        if not num_chunk:
            if use_labels == 0:
                cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{ext}{filename[:-4]}.json')
            elif use_labels == 1:
                cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{ext}labels_int_{filename[:-4]}.json')
            elif use_labels == 2:
                cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{ext}labels_str_{filename[:-4]}.json')
            else:
                raise ValueError("invalid label setting")
        else:
            print("Loading multi-chunk data...")
            cached_features_files = [os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{ext}{filename[:-4]}_chunk_{k}.json') 
                                    for k in range(num_chunk)]
        self.examples = []
        self.tokenizers = tokenizers

        # Bert tokenizer special tokens
        self.bert_pad_token=tokenizers[0].convert_tokens_to_ids([tokenizers[0].pad_token])[0]

        # GPT-2 tokenizer special tokens
        self.gpt2_pad_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].pad_token])[0]
        self.gpt2_bos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].bos_token])[0]
        self.gpt2_eos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].eos_token])[0]

        global bert_pad_token
        global gpt2_pad_token
        bert_pad_token = self.bert_pad_token
        gpt2_pad_token = self.gpt2_pad_token
 
        if dataset == 'Yelp':
            label_on = True
        else: 
            label_on = False
        
        if not num_chunk:
            if os.path.exists(cached_features_file):
                logger.info("Loading features from cached file %s", cached_features_file)
            
                with open(cached_features_file, 'r') as handle:
                    self.examples = json.load(handle)
                print(f"Loaded {len(self.examples)} examples...")
            else:
                logger.info("Creating features from dataset file at %s", directory)
                dropped, count = self._read_corpus_natural_split(fname=file_path, label=label_on, max_length=block_size, block_size=block_size)
                logger.info("The number of dropped sentences is %d", dropped)
                logger.info("The number of processed sentences is %d", count)

                # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.
                logger.info("Saving features into cached file %s", cached_features_file)
                if use_philly:
                    save_solid = False
                    while not save_solid:
                        try:           
                            with open(cached_features_file, 'w') as handle:
                                json.dump(self.examples, handle)
                        except:
                            pass
                else:
                    with open(cached_features_file, 'w') as handle:
                        json.dump(self.examples, handle)
        else:
            self.examples=[]
            for cached_features_file in tqdm(cached_features_files):
                with open(cached_features_file, 'r') as handle:
                    self.examples += json.load(handle)
            print(f"Loaded {len(self.examples)} examples...")
       

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def create_buckets_and_shuffle(self, bucket_size, shuffle):

        lens = [example['bert_token_length'] for example in self.examples]

        combined = list(zip(self.examples, lens))

        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

        sorted_examples = [item[0] for item in sorted_combined]

        buckets = []
        for i in range(0, len(sorted_examples), bucket_size):
            bucket = sorted_examples[i:i + bucket_size]

            # Shuffle within the bucket
            if shuffle:
                bucket = [bucket[j] for j in torch.randperm(len(bucket)).tolist()]

            buckets.append(bucket)

        # Shuffle the order of the buckets
        if shuffle:
            buckets = [buckets[j] for j in torch.randperm(len(buckets)).tolist()]

        self.examples = [sentence for bucket in buckets for sentence in bucket]


    @staticmethod
    def collate(examples):
        # Convert to Tensors and build dataset
        input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=bert_pad_token)
        input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=gpt2_pad_token)
        token_lengths = torch.tensor( [[f['bert_token_length'], f['gpt2_token_length']] for f in examples] , dtype=torch.long)
        if 'bert_token_noisy' in examples[0]:
            input_ids_bert_noisy = pad_sequence([torch.tensor(f['bert_token_noisy'], dtype=torch.long) for f in examples], batch_first=True, padding_value=bert_pad_token)
            input_ids_gpt_noisy = pad_sequence([torch.tensor(f['gpt2_token_noisy'], dtype=torch.long) for f in examples], batch_first=True, padding_value=gpt2_pad_token)
            token_lengths_noisy = torch.tensor( [[f['bert_token_length_noisy'], f['gpt2_token_length_noisy']] for f in examples] , dtype=torch.long)
            return (input_ids_bert, input_ids_gpt, token_lengths, input_ids_bert_noisy, input_ids_gpt_noisy, token_lengths_noisy)
        else:
            return (input_ids_bert, input_ids_gpt, token_lengths)

    def _read_corpus_natural_split(self, fname, label, max_length, block_size):
        data = []
        labels = [] if label else None
        dropped = 0
        count = 0

        with open(fname) as fin:
            for line in fin:
                if label:
                    split_line = line.split('\t')
                    lb = split_line[0]
                    split_line_text = split_line[1]
                else:
                    split_line_text = line
                    split_line_text = split_line_text.strip()

                if len(split_line_text.split()) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(lb)

                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(split_line_text))
                #tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0) 

                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(split_line_text))
                #tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = [self.gpt2_bos_token] + tokenized_text1 + [self.gpt2_eos_token]
                tokenized_text1_length = len(tokenized_text1)

                example = {
                    'bert_token': tokenized_text0,
                    'bert_token_length':tokenized_text0_length,
                    'gpt2_token':tokenized_text1,
                    'gpt2_token_length': tokenized_text1_length
                }
                self.examples.append(example)
                count +=1

        return dropped, count



class Dialog_TokenDataset(Dataset):
    def __init__(self, tokenizers, args, file_path='train', text_split_mode='natural', block_size=512):

        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{filename[:-4]}.json')

        self.examples = []
        self.tokenizers = tokenizers

        # Bert tokenizer special tokens
        self.bert_pad_token=tokenizers[0].convert_tokens_to_ids([tokenizers[0].pad_token])[0]

        # GPT-2 tokenizer special tokens
        self.gpt2_pad_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].pad_token])[0]
        self.gpt2_bos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].bos_token])[0]
        self.gpt2_eos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].eos_token])[0]

        global bert_pad_token
        global gpt2_pad_token
        bert_pad_token = self.bert_pad_token
        gpt2_pad_token = self.gpt2_pad_token

        if args.dataset == 'Yelp':
            label_on = True
        else:
            label_on = False

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'r') as handle:
                self.examples = json.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            dropped, count = self._read_dialog_corpus_natural_split(fname=file_path, label=label_on, max_length=block_size, block_size=block_size, args=args)

            logger.info("The number of dropped sentences is %d", dropped)
            logger.info("The number of processed sentences is %d", count)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            if args.use_philly:
                save_solid = False
                while not save_solid:
                    try:           
                        with open(cached_features_file, 'w') as handle:
                            json.dump(self.examples, handle)
                    except:
                        pass
            else:
                with open(cached_features_file, 'w') as handle:
                    json.dump(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples):
        # Convert to Tensors and build dataset
        input_ids_bert_ctx = pad_sequence([torch.tensor(f['bert_token_ctx'], dtype=torch.long) for f in examples], batch_first=True, padding_value=bert_pad_token)
        input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=bert_pad_token)
        input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples], batch_first=True, padding_value=gpt2_pad_token)
        token_lengths = torch.tensor( [[f['bert_token_ctx_length'], f['bert_token_length'], f['gpt2_token_length']] for f in examples] , dtype=torch.long)

        return (input_ids_bert_ctx, input_ids_bert, input_ids_gpt, token_lengths)

    def _read_dialog_corpus_natural_split(self, fname, label, max_length, block_size, args):
        data = []
        labels = [] if label else None
        dropped = 0
        count = 0

        with open(fname) as fin:
            for line in fin:

                split_line_text = line
                split_line_text = split_line_text.strip()

                if len(split_line_text.split()) < 1:
                    dropped += 1
                    continue

                # if max_length:
                #     if len(split_line_text.split()) > max_length:
                #         dropped += 1
                #         continue

                context_text, response_text = split_line_text.split('\t')

                tokenized_text_ctx = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(context_text))
                tokenized_text_ctx = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text_ctx)
                
                if len(tokenized_text_ctx)>512:
                    tokenized_text_ctx = tokenized_text_ctx[-512:]
                    # pdb.set_trace()
                tokenized_text_ctx_length = len(tokenized_text_ctx) 

                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(response_text))
                tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
                if len(tokenized_text0)>512:
                    tokenized_text0 = tokenized_text0[-512:]
                    
                tokenized_text0_length = len(tokenized_text0) 

                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(response_text))
                tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = [self.gpt2_bos_token] + tokenized_text1 + [self.gpt2_eos_token]
                tokenized_text1_length = len(tokenized_text1)

                # pdb.set_trace()
                example = {
                    'bert_token_ctx': tokenized_text_ctx,
                    'bert_token_ctx_length':tokenized_text_ctx_length,
                    'bert_token': tokenized_text0,
                    'bert_token_length':tokenized_text0_length,
                    'gpt2_token':tokenized_text1,
                    'gpt2_token_length': tokenized_text1_length
                }
                self.examples.append(example)
                count +=1

        return dropped, count






class TextDataset_Split(Dataset):
    def __init__(self, tokenizer, args, file_path='train', text_split_mode='natural', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_gpt_{block_size}_{filename}')

        self.examples = []
        self.tokenizer = tokenizer

        # GPT tokenizers
        self.pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        self.bos_token_id=tokenizer.convert_tokens_to_ids([tokenizer.bos_token])[0]
        self.eos_token_id=tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]

        if args.dataset == 'Yelp':
            label_on = True
        else:
            label_on = False 
        
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            if text_split_mode == 'block':
                self._read_corpus_block_split(fname=file_path, block_size = block_size)
            elif text_split_mode == 'natural': 
                self._read_corpus_natural_split(fname=file_path, label=label_on, max_length=block_size, block_size=block_size)
            else:
                print('Please specify the mode to split the raw text')

            # pdb.set_trace()

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # pdb.set_trace()
        # Convert to Tensors and build dataset
        tokenized_text1= torch.tensor(self.examples[item][0], dtype=torch.long)
        tokenized_text_lengths = torch.tensor([self.examples[item][1]], dtype=torch.long)
        # pdb.set_trace()
        return (tokenized_text1, tokenized_text_lengths)

    def _read_corpus_natural_split(self, fname, label, max_length, block_size):
        data = []
        labels = [] if label else None
        dropped = 0
        


        with open(fname) as fin:
            for line in fin:

                if label:
                    split_line = line.split('\t')
                    lb = split_line[0]
                    split_line_text = split_line[1]
                else:
                    split_line_text = line

                if len(split_line_text) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(lb)

                tokenized_text1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(split_line_text))
                tokenized_text1 = self.tokenizer.add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1_length = len(tokenized_text1)
                
                tokenized_text1 = [self.bos_token_id] + tokenized_text1 + [self.eos_token_id]
                tokenized_text1 = tokenized_text1 + ([self.pad_token_id] *  (block_size - tokenized_text1_length - 2) ) # Pad up to the sequence length.
                assert len(tokenized_text1) == block_size

                self.examples.append([tokenized_text1, tokenized_text1_length])
                

                    

    def _read_corpus_block_split(self, fname, block_size):

        with open(fname, encoding="utf-8") as f:
            text = f.read()

        # Chunyuan: divide the linguistic text into the same length, then different tokenization schemes are applied
        while len(text) >= block_size:  # Truncate in block of block_size

            tokenized_text1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[:block_size]))
            tokenized_text1 = self.tokenizer.add_special_tokens_single_sentence(tokenized_text1)
            tokenized_text1_length = len(tokenized_text1)

            tokenized_text1 = [bos_token_id] + tokenized_text1 + [eos_token_id]
            tokenized_text1 = tokenized_text1 + ([pad_token_id] *  (block_size - tokenized_text1_length - 2) ) # Pad up to the sequence length.
            assert len(tokenized_text1) == block_size

            self.examples.append([tokenized_text1, tokenized_text1_length])
            text = text[block_size:]





class TextDataset_2Tokenizers_LCtrlG(Dataset):
    def __init__(self, tokenizers, args, file_path='train', text_split_mode='natural', block_size=512, create_new=0):
        print(file_path)
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{filename}')

        self.examples = []
        self.tokenizers = tokenizers

        # GPT tokenizers
        self.pad_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].pad_token])[0]
        self.bos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].bos_token])[0]
        self.eos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].eos_token])[0]

        if not create_new and os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            if text_split_mode == 'natural':
                if args.dataset == 'Yelp':
                    dropped = self._read_corpus_natural_split_yelp(fname=file_path, label=True, max_length=block_size, block_size=block_size)
                    logger.info("The number of dropped sentences is %d", dropped)
                elif args.dataset == 'yahoo':
                    pass
                else:
                    raise NotImplementedError
            else:
                raise ValueError('Please specify the mode to split the raw text')

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # pdb.set_trace()
        # Convert to Tensors and build dataset
        tokenized_text0= torch.tensor(self.examples[item][0], dtype=torch.long)
        tokenized_text1= torch.tensor(self.examples[item][2], dtype=torch.long)
        tokenized_text_lengths = torch.tensor([self.examples[item][1], self.examples[item][3]], dtype=torch.long)
        label = torch.tensor(self.examples[item][4], dtype=torch.long)

        # pdb.set_trace()
        return (tokenized_text0, tokenized_text1, tokenized_text_lengths, label)

    def get_labels(self):
        return ['0', '1']

    def _read_corpus_natural_split_yelp(self, fname, label, max_length, block_size):
        # label: the file contains labels.
        dropped = 0
        label_fname = fname.replace('.text', '.labels')

        with open(fname) as fin, open(label_fname) as lfin:
            for line, label_line in zip(fin, lfin):
                # pdb.set_trace()
                split_line_text = line
                lb = int(label_line)
                assert lb in [0, 1]   # binary sentiment in yelp dataset.

                if len(split_line_text) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
                        dropped += 1
                        continue

                # tokenize by tokenizers[0]
                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(split_line_text))
                tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0)
                pad_token=self.tokenizers[0].convert_tokens_to_ids([self.tokenizers[0].pad_token])[0]
                # pad to max_seq_length (block_size)
                if block_size > tokenized_text0_length:
                    tokenized_text0 = tokenized_text0 + ([pad_token] * (block_size - tokenized_text0_length)  ) # Pad up to the sequence length.
                else:
                    dropped += 1
                    continue
                assert len(tokenized_text0) == block_size

                # tokenize by tokenizers[1]
                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(split_line_text))
                tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = [self.bos_token] + tokenized_text1 + [self.eos_token]
                tokenized_text1_length = len(tokenized_text1)
                # pad to max_seq_length (block_size)
                if block_size > tokenized_text1_length:
                    tokenized_text1 = tokenized_text1 + ([self.pad_token] *  (block_size - tokenized_text1_length) ) # Pad up to the sequence length.
                else:
                    dropped += 1
                    continue
                assert len(tokenized_text1) == block_size

                self.examples.append([tokenized_text0, tokenized_text0_length, tokenized_text1, tokenized_text1_length, lb])

        return dropped


class TextDataset_2Tokenizers(Dataset):
    def __init__(self, tokenizers, args, file_path='train', text_split_mode='natural', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_gpt_bert_{block_size}_{filename}')

        self.examples = []
        self.tokenizers = tokenizers

        # GPT tokenizers
        self.pad_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].pad_token])[0]
        self.bos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].bos_token])[0]
        self.eos_token=tokenizers[1].convert_tokens_to_ids([tokenizers[1].eos_token])[0]

        if args.dataset == 'Yelp':
            label_on = True
        else:
            label_on = False 

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            if text_split_mode == 'block':
                self._read_corpus_block_split(fname=file_path, block_size = block_size)
            elif text_split_mode == 'natural': 
                dropped, count = self._read_corpus_natural_split(fname=file_path, label=label_on, max_length=block_size, block_size=block_size, args=args)
                logger.info("The number of dropped sentences is %d", dropped)
                logger.info("The number of used sentences is %d", count)
            else:
                print('Please specify the mode to split the raw text')

            # pdb.set_trace()

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            if args.use_philly:
                save_solid = False
                while not save_solid:
                    try:           
                        with open(cached_features_file, 'wb') as handle:
                            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
            else:
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # pdb.set_trace()
        # Convert to Tensors and build dataset
        tokenized_text0= torch.tensor(self.examples[item][0], dtype=torch.long)
        tokenized_text1= torch.tensor(self.examples[item][2], dtype=torch.long)
        tokenized_text_lengths = torch.tensor([self.examples[item][1], self.examples[item][3]], dtype=torch.long)
        
        # pdb.set_trace()
        return (tokenized_text0, tokenized_text1, tokenized_text_lengths)

    def _read_corpus_natural_split(self, fname, label, max_length, block_size, args):
        data = []
        labels = [] if label else None
        dropped = 0
        count = 0

        with open(fname) as fin:
            for line in fin:
                # pdb.set_trace()

                if label:
                    split_line = line.split('\t')
                    lb = split_line[0]
                    split_line_text = split_line[1]
                else:
                    split_line_text = line

                if len(split_line_text.split()) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(lb)

                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(split_line_text))
                tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
                tokenized_text0_length = len(tokenized_text0) 
                pad_token=self.tokenizers[0].convert_tokens_to_ids([self.tokenizers[0].pad_token])[0]
                if block_size>tokenized_text0_length:
                    tokenized_text0 = tokenized_text0 + ([pad_token] * (block_size - tokenized_text0_length)  ) # Pad up to the sequence length.
                else:
                    dropped += 1
                    continue   

                assert len(tokenized_text0) == block_size
                
                tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(split_line_text))
                tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
                tokenized_text1 = [self.bos_token] + tokenized_text1 + [self.eos_token]
                tokenized_text1_length = len(tokenized_text1)
                
                if block_size>tokenized_text1_length:
                    tokenized_text1 = tokenized_text1 + ([self.pad_token] *  (block_size - tokenized_text1_length) ) # Pad up to the sequence length.
                else:
                    dropped += 1
                    continue                 
                
                assert len(tokenized_text1) == block_size

                self.examples.append([tokenized_text0, tokenized_text0_length, tokenized_text1, tokenized_text1_length])

                count +=1
                # if args.dataset == 'wikipedia' and count==10: 
                #     break

        return dropped, count

    def _read_corpus_block_split(self, fname, block_size):

        with open(fname, encoding="utf-8") as f:
            text = f.read()

        # Chunyuan: divide the linguistic text into the same length, then different tokenization schemes are applied
        while len(text) >= block_size:  # Truncate in block of block_size

            tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(self.tokenizers[0].tokenize(text[:block_size]))
            tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(tokenized_text0)
            tokenized_text0_length = len(tokenized_text0) 
            pad_token=self.tokenizers[0].convert_tokens_to_ids([self.tokenizers[0].pad_token])[0]
            tokenized_text0 = tokenized_text0 + ([pad_token] * (block_size - tokenized_text0_length)  ) # Pad up to the sequence length.
            assert len(tokenized_text0) == block_size
            
            tokenized_text1 = self.tokenizers[1].convert_tokens_to_ids(self.tokenizers[1].tokenize(text[:block_size]))
            tokenized_text1 = self.tokenizers[1].add_special_tokens_single_sentence(tokenized_text1)
            tokenized_text1_length = len(tokenized_text1)

            
            tokenized_text1 = [bos_token] + tokenized_text1 + [eos_token]
            tokenized_text1 = tokenized_text1 + ([pad_token] *  (block_size - tokenized_text1_length - 2) ) # Pad up to the sequence length.
            assert len(tokenized_text1) == block_size

            self.examples.append([tokenized_text0, tokenized_text0_length, tokenized_text1, tokenized_text1_length])
            text = text[block_size:]


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.3):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L 

def frange_cycle_both_ramp(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_zero=0.2, ratio_one=0.2, ratio_increase=0.2):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
                # v -= step
            elif period*ratio_zero <= i <period*(ratio_zero+ratio_increase):
                L[int(i+c*period)] = v
                v += step
            elif period*(ratio_zero+ratio_increase) <= i <period*(ratio_zero+ratio_increase+ratio_one):
                L[int(i+c*period)] = stop
            elif period*(ratio_zero+ratio_increase+ratio_one) <= i < period*(ratio_zero+ratio_increase+ratio_one+ratio_increase):
                L[int(i+c*period)] = v
                v -= step
            else: 
                L[int(i+c*period)] = start
            i += 1
    return L 


class uniform_initializer(object):
        def __init__(self, stdv):
            self.stdv = stdv
        def __call__(self, tensor):
            nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)

def reconstruct(model, test_data_batch, vocab, strategy, fname):
    hyps = []
    refs = []
    with open(fname, "w") as fout:
        #for i in range(10):
            # batch_data = test_data_batch[i]

        for batch_data in test_data_batch:
            decoded_batch = model.reconstruct(batch_data, strategy)

            source = [[vocab.id2word(id_.item()) for id_ in sent] for sent in batch_data]
            for j in range(len(batch_data)):
                ref = " ".join(source[j])
                hyp = " ".join(decoded_batch[j])
                fout.write("SOURCE: {}\n".format(ref))
                fout.write("RECON: {}\n\n".format(hyp))

                refs += [ref[len("<s>"): -len("</s>")]]
                if strategy == "beam":
                    hyps += [hyp[len("<s>"): -len("</s>")]]
                else:
                    hyps += [hyp[: -len("</s>")]]

    fname_ref = fname + ".ref"
    fname_hyp = fname + ".hyp"
    with open(fname_ref, "w") as f:
        f.write("\n".join(refs))
    with open(fname_hyp, "w") as f:
        f.write("\n".join(hyps))
    call_multi_bleu_perl("scripts/multi-bleu.perl", fname_hyp, fname_ref, verbose=True)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # logits.masked_fill_(logits < threshold, filter_value)  # (B, vocab_size)
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (B, vocab_size)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # (B, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # indices_to_remove = sorted_indices[sorted_indices_to_remove]

        # logits.masked_fill_(indices_to_remove, filter_value)
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def sample_sequence_conditional(model, length, context, past=None, temperature=1, top_k=0, top_p=0.0, decoder_tokenizer=None):
    generated = context
    i = 0
    finished = []
    for j in range(generated.size(0)):
        finished.append(False)
    finished = torch.tensor(finished).to(context.device)
    with torch.no_grad():
        while True:
        # for _ in trange(length):
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            #print(outputs[0].shape)
            next_token_logits = outputs[0][:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            # pdb.set_trace()
            finished = torch.logical_or(finished, next_token[:,0] == decoder_tokenizer.encode('<EOS>')[0])
            #print(finished.sum())
            if finished.sum() == next_token.size(0) or i == length:
                break
            i += 1

    return generated

def text_from_latent_code(latent_z,
                          model_vae,
                          temperature,
                          top_k,
                          top_p,
                          tokenizer_decoder):
    past = latent_z
    context_tokens = tokenizer_decoder.encode('<BOS>')
    context = torch.tensor(context_tokens, dtype=torch.long, device=latent_z.device)
    context = context.unsqueeze(0).repeat(latent_z.size(0), 1)

    length = 256 # maximum length, but not used 
    out = sample_sequence_conditional(
        model=model_vae.decoder,
        context=context,
        past=past,
        length= length, # Chunyuan: Fix length; or use <EOS> to complete a sentence
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        decoder_tokenizer = tokenizer_decoder
    )
    
    texts = []
    eos_token_val = tokenizer_decoder.encode('<EOS>')[0]
    for i in range(latent_z.size(0)):
        tokens = out[i,1:].tolist() #filter out BOS
        if eos_token_val in tokens:
            loc = tokens.index(eos_token_val)
            tokens = tokens[:loc] #end before EOS
        text_x1 = tokenizer_decoder.decode(tokens, clean_up_tokenization_spaces=True)
        #print(text_x1)
        text_x1 = text_x1.split()#[1:-1]
        text_x1 = ' '.join(text_x1)
        texts.append(text_x1)
    if len(texts) == 1:
        return texts[0]
    return texts

def calc_bleu_reconstruction(model_vae, eval_dataloader, tokenizer_decoder, args, nsamples=10, sample_latents=True):
    overall_bleu = 0
    num_sents = 0

    if not sample_latents:
        nsamples = 1 # no point averaging across multiple samples if running deterministic read write

    for batch in tqdm(eval_dataloader, desc="Evaluating BLEU"):
        x0, x1, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]
        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        num_sents += args.eval_batch_size

        # feed to model
        bleu_sample_avg = np.zeros(len(x1))
        for _ in range(nsamples):
            _, z = model_vae.ae(x0, x1, args.ae_read_write, sample_latents, args.read_iters)
            sent_recs = text_from_latent_code(z, model_vae, args, tokenizer_decoder)
            for i, sent in enumerate(sent_recs):
                sent = sent.replace("\\n", " ")
                words = wordpunct_tokenize(sent)
                sent_recs[i] = words
            for i in range(len(x1)):
                x1_lst = x1[i].tolist()
                sent_true = tokenizer_decoder.decode(x1_lst[1:x1_lst.index(50259)]) #50259 is EOS token
                # word tokenize
                sent_true = sent_true.replace("\\n", " ")
                sent_true = wordpunct_tokenize(sent_true)
                bleu_sample_avg[i] += nltk.translate.bleu_score.sentence_bleu([sent_true], sent_recs[i])
        bleu_sample_avg /= nsamples
        overall_bleu += np.sum(bleu_sample_avg)
    return overall_bleu / num_sents
        
def calc_bleu_denoise(model_vae, eval_dataloader, tokenizer_decoder, args, nsamples=10, sample_latents=True):
    overall_bleu = 0
    num_sents = 0

    if not sample_latents:
        nsamples = 1 # no point averaging across multiple samples if running deterministic read write
    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        _, x1, _, x0_noisy, x1_noisy, x_lengths_noisy = batch
        max_len_values, _ = x_lengths_noisy.max(0)
        x0_noisy = x0_noisy[:,:max_len_values[0]]
        x1_noisy = x1_noisy[:,:max_len_values[1]]
        x0_noisy = x0_noisy.to(args.device)
        x1_noisy = x1_noisy.to(args.device)
        num_sents += args.eval_batch_size

        # feed to model
        bleu_sample_avg = np.zeros(len(x1))
        for _ in range(nsamples):
            _, z = model_vae.ae(x0_noisy, x1_noisy, args.ae_read_write, sample_latents, args.read_iters)
            sent_recs = text_from_latent_code(z, model_vae, args, tokenizer_decoder)
            for i, sent in enumerate(sent_recs):
                sent = sent.replace("\\n", " ")
                words = wordpunct_tokenize(sent)
                sent_recs[i] = words
            for i in range(len(x1)):
                #x1_lst = x1_noisy[i].tolist()
                #print('noisy sent: ' + tokenizer_decoder.decode(x1_lst[1:x1_lst.index(50259)]))
                x1_lst = x1[i].tolist()
                sent_true = tokenizer_decoder.decode(x1_lst[1:x1_lst.index(50259)])
                #print('clean sent: ' + sent_true)
                sent_true = sent_true.replace("\\n", " ")
                sent_true = wordpunct_tokenize(sent_true)
                bleu_sample_avg[i] += nltk.translate.bleu_score.sentence_bleu([sent_true], sent_recs[i])
        bleu_sample_avg /= nsamples
        overall_bleu += np.sum(bleu_sample_avg)
    return overall_bleu / num_sents

def calc_ae_perp(model_vae, eval_dataloader, args):
    rec_loss = 0
    num_sents = 0
    num_words = 0
    for batch in eval_dataloader:
        x0, x1, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)
        num_words += x_lengths[:,1].sum().item()
        num_sents += args.eval_batch_size
        rl, _ = model_vae.ae(x0, x1, args.ae_read_write)
        rec_loss += rl.sum().item()
    perp = np.exp(rec_loss / num_words)
    return rec_loss / num_sents, perp



def calc_iwnll(model_vae, eval_dataloader, args, nsamples=100):
    ns = min(nsamples, 10)
    eval_loss = 0.0
    ############ Perplexity ############
    report_kl_loss = report_rec_loss = report_loss = 0
    report_rec_z_loss = 0
    report_num_words = report_num_sents = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        # pdb.set_trace()
        # not predict start symbol
        report_num_words += x_lengths[:,1].sum().item()
        report_num_sents += args.eval_batch_size

        with torch.no_grad():
            loss, loss_rc, loss_kl, loss_rc_z = model_vae.loss_iw(x0, x1, nsamples=nsamples, ns=ns)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss_rc_z = loss_rc_z.sum()
        loss = loss.sum()

        report_rec_loss += loss_rc.item()
        report_rec_z_loss += loss_rc_z.item()
        report_kl_loss += loss_kl.item()
        report_loss += loss.item()

        # pdb.set_trace()
    test_loss = report_loss / report_num_sents
    
    elbo = (report_kl_loss - report_rec_loss) / report_num_sents
    nll  = - report_rec_loss / report_num_sents
    nll_z = - report_rec_z_loss / report_num_sents
    kl   = report_kl_loss / report_num_sents
    ppl  = np.exp(-report_loss / report_num_words)

    return ppl, elbo, nll, nll_z, kl



def calc_rec(model_vae, eval_dataloader, args, ns=1):

    eval_loss = 0.0
    ############ Perplexity ############
    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0

    i = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        # pdb.set_trace()
        # not predict start symbol
        report_num_words += x_lengths[:,1].sum().item()
        report_num_sents += args.eval_batch_size

        with torch.no_grad():
            loss, loss_rc, loss_kl = model_vae.loss_iw(x0, x1, nsamples=1, ns=1)

        loss_rc = loss_rc.sum()
        report_rec_loss += loss_rc.item()

        i += 1
        if i > 500:
            break


        # pdb.set_trace()

    nll_s  = - report_rec_loss / report_num_sents
    nll_w  = - report_rec_loss / report_num_words

    return nll_s, nll_w



# def calc_mi(model, test_data_batch):
#     mi = 0
#     num_examples = 0
#     for batch_data in test_data_batch:
#         batch_size = batch_data.size(0)
#         num_examples += batch_size
#         mutual_info = model.calc_mi_q(batch_data)
#         mi += mutual_info * batch_size

#     return mi / num_examples



def calc_mi(model_vae, test_data_batch, args):
    # calc_mi_v3
    import math 
    from modules.utils import log_sum_exp

    mi = 0
    num_examples = 0

    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.
    for batch in tqdm(test_data_batch, desc="Evaluating MI, Stage 1"):

        x0, _, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]

        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mu, logvar = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        x_batch, nz = mu.size()

        #print(x_batch, end=' ')

        num_examples += x_batch

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

        neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
        mu_batch_list += [mu.cpu()]
        logvar_batch_list += [logvar.cpu()]


    neg_entropy = neg_entropy / num_examples
    ##print()

    num_examples = 0
    log_qz = 0.
    for i in tqdm(range(len(mu_batch_list)), desc="Evaluating MI, Stage 2"):

        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        
        # [z_batch, 1, nz]
        with torch.no_grad():
            z_samples = model_vae.reparameterize(mu, logvar, 1)

        z_samples = z_samples.view(-1, 1, nz)
        num_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, nz]
        #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
        x_batch, nz = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    log_qz /= num_examples
    mi = neg_entropy - log_qz

    return mi.item()





def calc_au(model_vae, eval_dataloader, args, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating AU, Stage 1"):

        x0, _, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mean, logvar = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating AU, Stage 2"):

        x0, _, _ = batch
        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mean, _ = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    # pdb.set_trace()
    return (au_var >= delta).sum().item(), au_var


def sample_sentences(vae, vocab, device, num_sentences):
    global logging

    vae.eval()
    sampled_sents = []
    for i in range(num_sentences):
        z = vae.sample_from_prior(1)
        z = z.view(1,1,-1)
        start = vocab.word2id['<s>']
        # START = torch.tensor([[[start]]])
        START = torch.tensor([[start]])
        end = vocab.word2id['</s>']
        START = START.to(device)
        z = z.to(device)
        vae.eval()
        sentence = vae.decoder.sample_text(START, z, end, device)
        decoded_sentence = vocab.decode_sentence(sentence)
        sampled_sents.append(decoded_sentence)
    for i, sent in enumerate(sampled_sents):
        logging(i,":",' '.join(sent))

# def visualize_latent(args, vae, device, test_data):
#     f = open('yelp_embeddings_z','w')
#     g = open('yelp_embeddings_labels','w')

#     test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)
#     for i in range(len(test_data_batch)):
#         batch_data = test_data_batch[i]
#         batch_label = test_label_batch[i]
#         batch_size, sent_len = batch_data.size()
#         means, _ = vae.encoder.forward(batch_data)
#         for i in range(batch_size):
#             mean = means[i,:].cpu().detach().numpy().tolist()
#             for val in mean:
#                 f.write(str(val)+'\t')
#             f.write('\n')
#         for label in batch_label:
#             g.write(label+'\n')
#         fo
#         print(mean.size())
#         print(logvar.size())
#         fooo

def visualize_latent(args, epoch, vae, device, test_data):
    nsamples = 1

    with open(os.path.join(args.exp_dir, f'synthetic_latent_{epoch}.txt'),'w') as f:
        test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)
        for i in range(len(test_data_batch)):
            batch_data = test_data_batch[i]
            batch_label = test_label_batch[i]
            batch_size, sent_len = batch_data.size()
            samples, _ = vae.encoder.encode(batch_data, nsamples)
            for i in range(batch_size):
                for j in range(nsamples):
                    sample = samples[i,j,:].cpu().detach().numpy().tolist()
                    f.write(batch_label[i] + '\t' + ' '.join([str(val) for val in sample]) + '\n')


def call_multi_bleu_perl(fname_bleu_script, fname_hyp, fname_ref, verbose=True):
    cmd = "perl %s %s < %s" % (fname_bleu_script, fname_ref, fname_hyp)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, \
        stderr=subprocess.PIPE, shell=True)
    popen.wait()
    try:
        bleu_result = popen.stdout.readline().strip().decode("utf-8")
        if verbose:
            print(bleu_result)
        bleu = float(bleu_result[7:bleu_result.index(',')])
        stderrs = popen.stderr.readlines()
        if len(stderrs) > 1:
            for line in stderrs:
                print(line.strip())
    except Exception as e:
        print(e)
        bleu = 0.
    return bleu




def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__ == '__main__':
    pass
