#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
import logging
import os
import pickle
import random


import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, BertConfig
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForLatentConnector
from pytorch_transformers import BertForLatentConnector, BertTokenizer

from collections import defaultdict
from modules import MemVAE
from utils import (TextDataset_Split, TextDataset_2Tokenizers, BucketingDataLoader)


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    else:
        dataset = TextDataset_Split(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset

def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        if not evaluate:
            args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            file_path=args.train_data_file
        else:
            args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)  
            file_path=args.eval_data_file
        dataloader = BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=False)
    else:
        pass 
    return dataloader


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

def sample_sequence_conditional(model, length, context, past=None, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None):
    generated = context
    i = 0
    finished = []
    for j in range(generated.size(0)):
        finished.append(False)
    finished = torch.tensor(finished).to(device)
    with torch.no_grad():
        while True:
        # for _ in trange(length):
            inputs = {'input_ids': generated, 'past_key_values': past}
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



def latent_code_from_text_single(text, tokenizer_encoder, model_vae, args):
    tokenized1 = tokenizer_encoder.encode(text)
    tokenized1 = [101] + tokenized1 + [102]
    coded1 = torch.Tensor([tokenized1])
    coded1 =torch.Tensor.long(coded1)
    with torch.no_grad():
        x0 = coded1
        x0 = x0.to(args['device'])
        pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
        mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        latent_z = mean.squeeze(1)  
        coded_length = len(tokenized1)
        return latent_z, coded_length
    
def latent_code_from_text(text, tokenizer_encoder, model_vae, args):
    if isinstance(text, list):
        latents = []
        coded_lengths = []
        for sent in text:
            z, cl = latent_code_from_text_single(sent, tokenizer_encoder, model_vae, args)
            latents.append(z)
            coded_lengths.append(cl)
        return torch.vstack(latents).to(args['device']), cl
    else:
        return latent_code_from_text_single(text, tokenizer_encoder, model_vae, args)

def text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder):
    past = latent_z
    context_tokens = tokenizer_decoder.encode('<BOS>')
    context = torch.tensor(context_tokens, dtype=torch.long, device=args['device'])
    context = context.unsqueeze(0).repeat(latent_z.size(0), 1)

    length = 256 # maximum length, but not used 
    out = sample_sequence_conditional(
        model=model_vae.decoder,
        context=context,
        past=past,
        length= length, # Chunyuan: Fix length; or use <EOS> to complete a sentence
        temperature=args['temperature'],
        top_k=args['top_k'],
        top_p=args['top_p'],
        device=args['device'],
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

def text_from_latent_code_mask(latent_z, model_vae, args, prompt, tokenizer_decoder):
    past = latent_z
    context_tokens = tokenizer_decoder.encode('<BOS>' + prompt)
    context = torch.tensor(context_tokens, dtype=torch.long, device=args.device)
    context = context.unsqueeze(0).repeat(latent_z.size(0), 1)

    out = sample_sequence_conditional(
        model=model_vae.decoder,
        context=context,
        past=past,
        length= 256,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
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

def unmask(model_vae, tokenizer_encoder, tokenizer_decoder, text_write_to_mem, text_to_denoise, iterative_step, args, episode_size=1):
    # encode examples to write to memory
    input_encoded, _ = latent_code_from_text(text_write_to_mem, tokenizer_encoder, model_vae, args)
    # here we treat the whole sequence of texts to write as an episode
    input_encoded = input_encoded.reshape(len(text_write_to_mem), 1, model_vae._code_size)
    posterior_memory, dkl_M = model_vae.write(input_encoded=input_encoded)

    # encoding for reconstruction
    input_encoded_noise, _ = latent_code_from_text(text_to_denoise, tokenizer_encoder, model_vae, args)
    input_encoded_noise = input_encoded_noise.reshape(1, 1, model_vae._code_size)

     # iterative reading to denoise latent
    input_recon_list = [] # list of reconstructions with index i being reconstruction with i+1 read iters
    z, dkl_w = model_vae.read(input_encoded_noise, posterior_memory)
    z = z.reshape(1, model_vae._code_size)

    text_split = text_to_denoise.split('[MASK]')

    input_recon = text_from_latent_code_mask(z, model_vae, args, text_split[0], tokenizer_decoder)
    print(input_recon)
    input_recon_list.append(input_recon)

    for _ in range(iterative_step - 1):
        input_recon_encoded, _ = latent_code_from_text(input_recon, tokenizer_encoder, model_vae, args)
        input_recon_encoded = input_recon_encoded.reshape(1, 1, model_vae._code_size)

        z, dkl_w = model_vae.read(input_recon_encoded, posterior_memory)
        z = z.reshape(1, model_vae._code_size)
        input_recon = text_from_latent_code_mask(z, model_vae, args, text_split[0], tokenizer_decoder)
        print(input_recon)
        input_recon_list.append(input_recon)
        
    return input_recon_list



def denoise(model_vae, tokenizer_encoder, tokenizer_decoder, text_write_to_mem, text_to_denoise, iterative_step, args, episode_size=1):
    # encode examples to write to memory
    input_encoded, _ = latent_code_from_text(text_write_to_mem, tokenizer_encoder, model_vae, args)
    # here we treat the whole sequence of texts to write as an episode
    input_encoded = input_encoded.reshape(len(text_write_to_mem), 1, model_vae._code_size)
    posterior_memory, dkl_M = model_vae.write(input_encoded=input_encoded)

    # encoding for reconstruction
    input_encoded_noise, _ = latent_code_from_text(text_to_denoise, tokenizer_encoder, model_vae, args)
    input_encoded_noise = input_encoded_noise.reshape(1, 1, model_vae._code_size)

    # iterative reading to denoise latent
    input_recon_list = [] # list of reconstructions with index i being reconstruction with i+1 read iters
    z, dkl_w = model_vae.read(input_encoded_noise, posterior_memory)
    z = z.reshape(1, model_vae._code_size)

    # here we generate the entire text sequence from latent (?)
    input_recon = text_from_latent_code(z, model_vae, args, tokenizer_decoder)
    print(input_recon)
    input_recon_list.append(input_recon)

    for _ in range(iterative_step - 1):
        input_recon_encoded, _ = latent_code_from_text(input_recon, tokenizer_encoder, model_vae, args)
        input_recon_encoded = input_recon_encoded.reshape(1, 1, model_vae._code_size)

        z, dkl_w = model_vae.read(input_recon_encoded, posterior_memory)
        z = z.reshape(1, model_vae._code_size)
        input_recon = text_from_latent_code(z, model_vae, args, tokenizer_decoder)
        print(input_recon)
        input_recon_list.append(input_recon)
        
    return input_recon_list

def print_samples(sample):
    if isinstance(sample, str):
        print('gen 0: ', sample)
    else:
        for i in range(len(sample)):
            print(f'gen {i+1}: ', sample[i])

def generate_from_memory(model_vae, tokenizer_encoder, tokenizer_decoder, text_write_to_mem, args, num_samples=1, iterative_step=1):
    # encode examples to write to memory
    input_encoded, _ = latent_code_from_text(text_write_to_mem, tokenizer_encoder, model_vae, args)
    # here we treat the whole sequence of texts to write as an episode
    input_encoded = input_encoded.reshape(len(text_write_to_mem), 1, model_vae._code_size)
    posterior_memory, dkl_M = model_vae.write(input_encoded=input_encoded)
    M = posterior_memory[0]

    # sample w from prior
    w = torch.randn(num_samples, 1, model_vae._memory_size).to(args.device)
    z = torch.bmm(w.transpose(0, 1), M).transpose(0, 1)
    z = z.reshape(num_samples, model_vae._code_size)

    sample_list = []
    sample = text_from_latent_code(z, model_vae, args, tokenizer_decoder)
    print('iteration 0')
    print_samples(sample)
    sample_list.append(sample)

    # iteratively read from memory
    for j in range(iterative_step - 1):
        sample_list.append([])
        for i in range(num_samples):
            sample_encoded, _ = latent_code_from_text(sample[i], tokenizer_encoder, model_vae, args)
            sample_encoded = sample_encoded.reshape(1, 1, model_vae._code_size)

            z, dkl_w = model_vae.read(sample_encoded, posterior_memory)
            z = z.reshape(1, model_vae._code_size)
            sample_new = text_from_latent_code(z, model_vae, args, tokenizer_decoder)
            sample_list[-1].append(sample_new)
        print(f'iteration {j+1}')
        print_samples(sample_list[-1])
    return sample_list

# a wrapper function to choose between different play modes
def evaluate_latent_space(args, model_vae, encoder_tokenizer, decoder_tokenizer, prefix=""):

    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=False)

    # Eval!
    logger.info("***** Running recontruction evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
    
    model_vae.eval()

    model_vae =  model_vae.module if hasattr(model_vae, 'module') else model_vae  # Take care of distributed/parallel training

    if args.play_mode == 'reconstrction':
        result = calc_rec(model_vae, eval_dataloader, encoder_tokenizer, decoder_tokenizer, args, ns=100)
        result_file_name = "eval_recontruction_results.txt"
    elif args.play_mode == 'interpolation':
        result = calc_interpolate(model_vae, eval_dataloader, encoder_tokenizer, decoder_tokenizer, args, ns=100)
        result_file_name = "eval_interpolation_results.txt"
    else:
        logger.info("Please specify the corrent play mode [reconstrction, interpolation]")
        

    eval_output_dir = args.output_dir
    output_eval_file = os.path.join(eval_output_dir, result_file_name)

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval {} results *****".format(args.play_mode))
        for key in sorted(result.keys()):
            logger.info("  %s \n %s", key, str(result[key]))
            writer.write("%s \n %s\n" % (key, str(result[key])))

    return result


def calc_rec(model_vae, eval_dataloader, encoder_tokenizer, decoder_tokenizer, args, ns=1):

    count = 0
    result = defaultdict(str)
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        context_tokens = decoder_tokenizer.encode('<BOS>')

        with torch.no_grad():

            text_x0 = encoder_tokenizer.decode(x0[0,:x_lengths[0,0]].tolist(), clean_up_tokenization_spaces=True)[0]
            # result["INPUT TEXT " + str(count)].append(text_x0)

            pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
  
            # Connect hidden feature to the latent space
            # latent_z, loss_kl = model_vae.connect(pooled_hidden_fea)
            mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = mean.squeeze(1)

            past = latent_z
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=past,
                length=x_lengths[0,1], # Chunyuan: Fix length; or use <EOS> to complete a sentence
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                decoder_tokenizer = decoder_tokenizer
            )
            text_x1 = decoder_tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
            text_x1 = text_x1.split()[1:-1]
            text_x1 = ' '.join(text_x1) + '\n'
            result[text_x0] = text_x1

        count += 1
        if count>args.total_sents:
            break
        

    return result




def calc_interpolate(model_vae, eval_dataloader, encoder_tokenizer, decoder_tokenizer, args, ns=1):

    count = 0
    latent_codes = []
    sample_interval = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating interpolation"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x0 = x0.to(args.device)
        x_lengths = x_lengths.to(args.device)


        with torch.no_grad():
            if sample_interval == 0 or sample_interval == args.total_sents:
                text_x0 = encoder_tokenizer.decode(x0[0,:x_lengths[0,0]].tolist(), clean_up_tokenization_spaces=True)[0]
                pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
    
                # Connect hidden feature to the latent space
                mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
                latent_z = mean.squeeze(1)
                
                latent_codes.append(latent_z)

                if sample_interval == 5: 
                    latent_codes.append(latent_z)
                    sample_interval = 0
                    continue
            else: 
                sample_interval += 1
                continue

        count += 1
        if count>args.total_sents:
            break                

    context_tokens = decoder_tokenizer.encode('<BOS>')
    result = defaultdict(str)
    latent_codes_interpolation = []
    num_steps = args.num_interpolation_steps
    for step in range(num_steps+1):
        latent_z = latent_codes[0] + (latent_codes[1] - latent_codes[0]) * step * 1.0/num_steps

        past = latent_z
        out = sample_sequence_conditional(
            model=model_vae.decoder,
            context=context_tokens,
            past=past,
            length=x_lengths[0,1], # Chunyuan: Fix length; or use <EOS> to complete a sentence
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            decoder_tokenizer = decoder_tokenizer
        )
        text_x1 = decoder_tokenizer.decode(out[0,:].tolist(), clean_up_tokenization_spaces=True)
        text_x1 = text_x1.split()[1:-1]
        text_x1 = ' '.join(text_x1) 
        result[step] = text_x1

    return result


def interpolate_w(model_vae, tokenizer_encoder, tokenizer_decoder, args):
    # encode examples to write to memory
    input_encoded, _ = latent_code_from_text(args.sent_memory, tokenizer_encoder, model_vae, args)
    # here we treat the whole sequence of texts to write as an episode
    input_encoded = input_encoded.reshape(len(args.sent_memory), 1, model_vae._code_size)
    posterior_memory, dkl_M = model_vae.write(input_encoded=input_encoded)
    M = posterior_memory[0]

    # encoding for reconstruction
    latent_z1, _ = latent_code_from_text(args.sent_source, tokenizer_encoder, model_vae, args)
    latent_z2, _ = latent_code_from_text(args.sent_target, tokenizer_encoder, model_vae, args)
    latent_z1 = latent_z1.reshape(1, 1, model_vae._code_size)
    latent_z2 = latent_z2.reshape(1, 1, model_vae._code_size)

    # iterative reading to denoise latent
    _, _, w1 = model_vae.read(latent_z1, posterior_memory, get_w=True)
    _, _, w2 = model_vae.read(latent_z2, posterior_memory, get_w=True)

    result = defaultdict(str)

    num_steps = args.num_interpolation_steps + 1
    for step in range(num_steps+1):
        w = w1 + (w2 - w1) * step * 1.0/num_steps

        z = torch.bmm(w.transpose(0, 1), M).transpose(0, 1)
        z = z.reshape(1, model_vae._code_size)
        
        text_interpolate = text_from_latent_code(z, model_vae, args, tokenizer_decoder)
        result[step] = text_interpolate
        print(text_interpolate)

    return result

def interpolate(model_vae, tokenizer_encoder, tokenizer_decoder, args):
    # and then in the main function         
    latent_z1, coded_length1 = latent_code_from_text(args.sent_source, tokenizer_encoder, model_vae, args)
    latent_z2, coded_length2 = latent_code_from_text(args.sent_target, tokenizer_encoder, model_vae, args)

    result = defaultdict(str)

    num_steps = args.num_interpolation_steps + 1
    for step in range(num_steps+1):
        latent_z = latent_z1 + (latent_z2 - latent_z1) * step * 1.0/num_steps
        
        text_interpolate = text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder)
        result[step] = text_interpolate
        print(text_interpolate)

    return result


def analogy(model_vae, tokenizer_encoder, tokenizer_decoder, args):
        
    latent_z1, coded_length1 = latent_code_from_text(args.sent_source, tokenizer_encoder, model_vae, args)
    latent_z2, coded_length2 = latent_code_from_text(args.sent_target, tokenizer_encoder, model_vae, args)
    latent_z3, coded_length3 = latent_code_from_text(args.sent_input, tokenizer_encoder, model_vae, args)
    
    result = defaultdict(str)

    latent_z = latent_z3 + args.degree_to_target * (latent_z2 - latent_z1) 
    
    text_analogy = text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder)
    result[0] = text_analogy
    print(text_analogy)

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=True,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default='Snli', type=str, help="The dataset.")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    parser.add_argument("--total_sents", default=10, type=int, help="Total sentences to test recontruction.")
    parser.add_argument("--num_interpolation_steps", default=10, type=int, help="Total sentences to test recontruction.")
    parser.add_argument("--play_mode", default="interpolation", type=str,
                        help="interpolation or reconstruction.")


    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")


    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gloabl_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")

    # Interact with users
    parser.add_argument("--interact_with_user_input", action='store_true', help="Use user input to interact_with.")
    parser.add_argument("--sent_source", type=str, default="")
    parser.add_argument("--sent_target", type=str, default="")
    parser.add_argument("--sent_input", type=str, default="")
    parser.add_argument("--sent_memory", type=str, nargs="+")
    parser.add_argument("--sent_denoise", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=1)
    
    parser.add_argument("--degree_to_target", type=float, default="1.0")

    ## Variational auto-encoder
    parser.add_argument("--nz", default=32, type=int,
                        help="Latent space dimension.")

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--memory_size", default=32, type=int, help="number of memory slots")
    parser.add_argument("--episode_sizes", nargs='+', type=int, help="episode sizes for training")
    parser.add_argument("--direct_writing", action='store_true')
    parser.add_argument("--ordering", action='store_true')
    parser.add_argument("--pseudoinverse_approx_step", default=7, type=int, help="number of steps for pseudoinverse approximation")
    parser.add_argument("--identity", action='store_true')
    parser.add_argument("--observation_noise_std", default=0, type=float, help="strength of observation noise")
    parser.add_argument("--deterministic_w", action="store_true", help="use deterministic addressing weights")
    parser.add_argument("--w_logvar_setting", default=0, type=int, help="setting to use for w logvar")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)


    args.encoder_model_type = args.encoder_model_type.lower()
    args.decoder_model_type = args.decoder_model_type.lower()


    global_step = args.gloabl_step_eval
    
    # Load full model
    checkpoint = torch.load(args.checkpoint_dir)
    output_encoder_dir = '<<<Please provide an encoder path, e.g. ending in .pth>>>'
    output_decoder_dir = '<<<Please provide a decoder path, e.g. ending in .pth>>>'

    # Load a trained Encoder model and vocabulary that you have fine-tuned
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[args.encoder_model_type]
    encoder_config = encoder_config_class.from_pretrained(args.encoder_model_name_or_path)
    setattr(encoder_config, "latent_size", args.latent_size)
    model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path, latent_size=args.latent_size, config=encoder_config)
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path, do_lower_case=args.do_lower_case)

    model_encoder.to(args.device)
    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    # Load a trained Decoder model and vocabulary that you have fine-tuned
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]
    decoder_config = decoder_config_class.from_pretrained(args.decoder_model_name_or_path)
    setattr(decoder_config, "latent_size", args.latent_size)
    model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, config=decoder_config, latent_size=args.latent_size)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path, do_lower_case=args.do_lower_case)
    model_decoder.to(args.device)
    if args.block_size <= 0:
        args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'

    
    # Evaluation
    model_vae = MemVAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    if "model_state_dict" not in checkpoint: 
        model_vae.load_state_dict(checkpoint)
    else:
        model_vae.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)

    if args.interact_with_user_input:
        if args.play_mode == 'generate':
            if len(args.sent_memory) > 0:
                result = generate_from_memory(model_vae, tokenizer_encoder, tokenizer_decoder, args.sent_memory, args, num_samples=args.num_samples, iterative_step=args.num_iterations)
            else:
                print('Please specify sentences to write to memory')
        if args.play_mode == "denoise":
            if args.sent_memory is not None and len(args.sent_denoise) > 0:
                if '[MASK]' in args.sent_denoise:
                    result = unmask(model_vae, tokenizer_encoder, tokenizer_decoder, args.sent_memory, args.sent_denoise, iterative_step=args.num_iterations, args=args, episode_size=1)
                else:
                    result = denoise(model_vae, tokenizer_encoder, tokenizer_decoder, args.sent_memory, args.sent_denoise, iterative_step=args.num_iterations, args=args, episode_size=1)
        if args.play_mode == 'interpolation':
            if len(args.sent_source) > 0 and len(args.sent_target) > 0:
                if args.sent_memory is not None:
                    result = interpolate_w(model_vae, tokenizer_encoder, tokenizer_decoder, args)
                else:
                    result = interpolate(model_vae, tokenizer_encoder, tokenizer_decoder, args)
            else:
                print('Please check: specify the source and target sentences!')

        if args.play_mode == 'analogy':
            if len(args.sent_source) > 0 and len(args.sent_source) > 0 and len(args.sent_input) > 0:
                result = analogy(model_vae, tokenizer_encoder, tokenizer_decoder, args)
            else:
                print('Please check: specify the source, target and input analogy sentences!')


    else:
        result = evaluate_latent_space(args, model_vae, tokenizer_encoder, tokenizer_decoder, prefix=global_step)


if __name__ == '__main__':
    main()
