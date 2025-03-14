import os
import json
import torch
import requests
import torch.nn.functional as F
import pandas as pd
from lightning_model import MemNetLight
try:
    from lightning_model_counterfactual import MemNetLight as MemNetLight_counterfactual
except ImportError:
    print("Cannot load lightning_model_counterfactual")
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(checkpoint_dir):
    model = MemNetLight.load_from_checkpoint(checkpoint_dir,
                                             optimizer=None, 
                                             decode_rec_strength=0, 
                                             use_ramp_both_sides=False, 
                                             ratio_one=0.0,
                                             use_bitfit=False)
    model.model = model.model.to('cuda')
    return model.model, model.tokenizer_encoder, model.tokenizer_decoder

def latent_code_from_text_single(text, tokenizer_encoder, model_vae, device):
    tokenized1 = tokenizer_encoder.encode(text)
    tokenized1 = [101] + tokenized1 + [102]
    coded1 = torch.Tensor([tokenized1])
    coded1 =torch.Tensor.long(coded1)
    with torch.no_grad():
        x0 = coded1
        x0 = x0.to(device)
        pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
        mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        latent_z = mean.squeeze(1)  
        coded_length = len(tokenized1)
        return latent_z, coded_length
    
def latent_code_from_text(text, tokenizer_encoder, model_vae, device, code_size=None):
    if isinstance(text, list):
        if not len(text):
            return torch.empty((1, code_size)).to(device), None
        latents = []
        coded_lengths = []
        for sent in text:
            z, cl = latent_code_from_text_single(sent, tokenizer_encoder, model_vae, device)
            latents.append(z)
            coded_lengths.append(cl)
        return torch.vstack(latents).to(device), coded_lengths
    else:
        return latent_code_from_text_single(text, tokenizer_encoder, model_vae, device)


def sample_sequence_conditional(model, length, context, past=None, greedy=False, temperature=1,
                                top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None):
    generated = context
    i = 0
    finished = []
    for j in range(generated.size(0)):
        finished.append(False)
    finished = torch.tensor(finished).to(device)
    with torch.no_grad():
        while True:
            inputs = {'input_ids': generated, 'past_key_values': past}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            if greedy:
                next_token = outputs[0][:, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            else:
                next_token_logits = outputs[0][:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            finished = torch.logical_or(finished, next_token[:,0] == decoder_tokenizer.encode('<EOS>')[0])
            if finished.sum() == next_token.size(0) or i == length:
                break
            i += 1
    return generated


def text_from_latent_code_mask(latent_z, model_vae, tokenizer_decoder, length, prompt='', is_vae=True,
                               temperature=0.7, top_k=50, top_p=0.95, device='cuda', greedy=False):
    past = latent_z
    context_tokens = tokenizer_decoder.encode('<BOS>' + prompt)
    context = torch.tensor(context_tokens, dtype=torch.long, device=device)
    if latent_z is not None:
        context = context.unsqueeze(0).repeat(latent_z.size(0), 1)
    else:
        # if No past input then repeat it once
        context = context.unsqueeze(0).repeat(1, 1)
        
    out = sample_sequence_conditional(
        model=model_vae.decoder if is_vae else model_vae,
        context=context,
        past=past,
        length=length,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        decoder_tokenizer = tokenizer_decoder)
    
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


def run_base_model(text, model, tokenizer, do_sample=True, device='cuda', temperature=0.7, length=256, top_k=50, top_p=0.95,):
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    output = model.generate(**encoded_input, temperature=temperature, do_sample=do_sample, max_new_tokens=length, top_k=top_k, top_p=top_p)
    output_text = tokenizer.decode(output[0])
    return output_text


def generate_from_decoder(text_input, model, enc_tok, dec_tok, text_write_to_mem=None, length=256, greedy=False,
                          mode='pyrite', prompt='', z_size=768, deterministic=True, return_memory=False):
    # three modes: (a) 'pyrite' (write to memory and reconstruct from reading), (b) 'vae': use encoder Z output to reconstruct,
    # (c) 'pyrite_generate': free-form generate from pyrite,
    # (d) 'unconstrained': use no past values from the encoder, (d) 'baseline': use the baseline model to generate the text
    # (e) 'baseline_ft': use the baseline model finetuned on Yelp to generate the text
    if mode=='pyrite':
        # First write to the memory
        # if isinstance(text_write_to_mem, list):
        #     latents = []
        #     coded_lengths = []
        #     for sent in text_write_to_mem:
        #         z, cl = latent_code_from_text(sent, enc_tok, model, device='cuda')
        #         latents.append(z)
        #         coded_lengths.append(cl)
        #     memory_write_text_encoded = torch.vstack(latents).to('cuda')
        # else:
        memory_write_text_encoded, cls = latent_code_from_text(text_write_to_mem, enc_tok, model, 
                                                               device='cuda', code_size=model._code_size)
        if cls is None:
            print("Found 0 len in text_to_write")
            memory_write_text_encoded = memory_write_text_encoded.reshape(1, 1, model._code_size)
        else:
            memory_write_text_encoded = memory_write_text_encoded.reshape(len(text_write_to_mem), 1, model._code_size)

        posterior_memory, dkl_M = model.write(input_encoded=memory_write_text_encoded)
        # Then read from the memory
        encoded_read, cls = latent_code_from_text(text_input, enc_tok, model, device='cuda')
        encoded_read = encoded_read.reshape(1, 1, model._code_size)
        if deterministic:
            model.memory.deterministic = True
        zs, _ = model.read(encoded_read, posterior_memory)
        zs = zs.reshape(1, model._code_size)
        output=text_from_latent_code_mask(zs, model, dec_tok, length=length, prompt=prompt, greedy=greedy)

    elif mode=='pyrite_generate':
        # First write to the memory
        memory_write_text_encoded, cls = latent_code_from_text(text_write_to_mem, enc_tok, model, device='cuda')
        memory_write_text_encoded = memory_write_text_encoded.reshape(len(text_write_to_mem), 1, model._code_size)
        posterior_memory, dkl_M = model.write(input_encoded=memory_write_text_encoded)
        # Sample w from prior
        M = posterior_memory[0]
        w = torch.randn(1, 1, model._memory_size).to('cuda')
        z = torch.bmm(w.transpose(0, 1), M).transpose(0, 1)
        zs = z.reshape(1, model._code_size)
        output=text_from_latent_code_mask(zs, model, dec_tok, length=length, prompt=prompt, greedy=greedy)
    elif mode=='vae':
        zs, cls = latent_code_from_text(text_input, enc_tok, model, device='cuda')
        output=text_from_latent_code_mask(zs, model, dec_tok, length=length, prompt=prompt)
    elif mode=='unconstrained':
        zs, cls = latent_code_from_text(text_input, enc_tok, model, device='cuda')
        output=text_from_latent_code_mask(torch.empty_like(zs), model, dec_tok, length=length, prompt=prompt)
    elif mode=='baseline':
        output=run_base_model(prompt, model, dec_tok, length=length) # use dec_tok as tokenizer and enc_tok as None for the baselines
    elif mode=='baseline_ft':
        output=run_base_model(prompt, model, dec_tok, length=length) # use dec_tok as tokenizer and enc_tok as None for the baselines (FT on Yelp)

    if ('pyrite' in mode) and return_memory:
        return output, posterior_memory
    else:
        return output


class BAM:
    url = '<<<Please provide an API endpoint for generation>>>'
    BAM_API_KEY=os.getenv('BAM_API_KEY')
    def __init__(self, model="google/flan-t5-xxl") -> None:
        self.model=model
    
    def ask(self, prompt, temperature=0.7, max_new_tokens=128, greedy=True, stop_sequences=None):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.BAM_API_KEY}'
        }
        decoding_method = 'greedy' if greedy else 'sample'
        data = {
            "model_id": self.model,
            "inputs": [prompt],
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "decoding_method": decoding_method,
                "stop_sequences": stop_sequences
            }
        }
        response=requests.post(self.url, headers=headers, data=json.dumps(data))
        outputf=response.json()['results'][0]['generated_text']
        return outputf
    
    def ask_batch(self, prompt, temperature=0.7, max_new_tokens=128, greedy=True):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.BAM_API_KEY}'
        }
        decoding_method = 'greedy' if greedy else 'sample'
        data = {
            "model_id": self.model,
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "decoding_method":decoding_method
            }
        }
        response=requests.post(self.url, headers=headers, data=json.dumps(data))
        outputf=[x['generated_text'] for x in response.json()['results']]
        return outputf
    

#### Subject 

import spacy

class SubjectMatcher:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")    
        self.db_sentences = []
    
    def add_sentences(self, examples):
        self.db_sentences += examples
    
    def reset(self):
        self.db_sentences = []

    def get_subject(self, sentence):
        doc = self.nlp(sentence)
        for token in doc:
            if "subj" in token.dep_:
                return token.text.lower() 
        return None 

    def check_subject_match(self, test_sentence):
        test_subject = self.get_subject(test_sentence)
        if test_subject is None:
            print(f"No subject found in test sentence: {test_sentence}")
            return 0.0, []

        matching_sentences = [sentence for sentence in self.db_sentences if self.get_subject(sentence) == test_subject]
        return float(len(matching_sentences)>0), matching_sentences
