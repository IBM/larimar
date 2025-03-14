from pprint import pprint
from typing import List, Optional
import scipy
from scipy.stats import hmean
import argparse
import torch
import torch.nn.functional as F
from lightning_model import MemNetLight
from transformers import AutoModelForCausalLM, AutoTokenizer
from read_write_interactive import top_k_top_p_filtering_batch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.nn.utils.rnn import pad_sequence
import unicodedata
import nltk
import collections
import json
from pathlib import Path
import typing
from torch.utils.data import Dataset
from itertools import chain
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import PyriteSentenceScope as pss
from time import time

REMOTE_URL_A = "https://rome.baulab.info/data/dsets/attribute_snippets.json"
REMOTE_URL_C = "https://rome.baulab.info/data/dsets/counterfact.json"
REMOTE_IDF_URL = "https://rome.baulab.info/data/dsets/idf.npy"
REMOTE_VOCAB_URL = "https://rome.baulab.info/data/dsets/tfidf_vocab.json"


def process_results(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in dir_name.iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            for prefix in ["post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # zsRE evaluation metrics
                for key in ["rewrite", "paraphrase", "neighborhood"]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]:
                        continue

                    cur_sum[sum_key].append(np.mean(data[prefix][key]))

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for prefix in ["post"]:
            for k_efficacy, k_generalization, k_specificity in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                (
                    f"{prefix}_rewrite_acc",
                    f"{prefix}_paraphrase_acc",
                    f"{prefix}_neighborhood_acc",
                ),
            ]:
                if k_generalization in cur_sum and k_specificity in cur_sum:
                    cur_sum[f"{prefix}_score"] = (
                        hmean(
                            [
                                cur_sum[k_efficacy][0],
                                cur_sum[k_generalization][0],
                                cur_sum[k_specificity][0],
                            ]
                        ),
                        np.nan,
                    )
                    break

        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        cur_sum.update(metadata)
        # pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


class MENDQADataset:

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        if size is not None:
            raw = raw[:size]

        data = []
        for i, record in enumerate(raw):
            assert ("nq question: " in record["loc"]), f"Neighborhood prompt missing `nq question:`. Check for errors?"

            ans_toks = tok.encode(" " + record["loc_ans"])
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                            "target_tok": ans_toks[i],
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


class AttributeSnippets:
    """
    Contains wikipedia snippets discussing entities that have some property.

    More formally, given a tuple t = (s, r, o):
    - Let snips = AttributeSnippets(DATA_DIR)
    - snips[r][o] is a list of wikipedia articles for all s' such that t' = (s', r, o) is valid.
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        snips_loc = data_dir / "attribute_snippets.json"
        if not snips_loc.exists():
            print(f"{snips_loc} does not exist. Downloading from {REMOTE_URL_A}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL_A, snips_loc)

        with open(snips_loc, "r") as f:
            snippets_list = json.load(f)

        snips = collections.defaultdict(lambda: collections.defaultdict(list))

        for el in snippets_list:
            rid, tid = el["relation_id"], el["target_id"]
            for sample in el["samples"]:
                snips[rid][tid].append(sample)

        self._data = snips
        self.snippets_list = snippets_list

    def __getitem__(self, item):
        return self._data[item]


class CounterFactDataset(Dataset):
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs):
        data_dir = Path(data_dir)
        cf_loc = data_dir / "counterfact.json"
        if not cf_loc.exists():
            print(f"{cf_loc} does not exist. Downloading from {REMOTE_URL_C}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL_C, cf_loc)

        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def get_tfidf_vectorizer(data_dir: str):
    """
    Returns an sklearn TF-IDF vectorizer. See their website for docs.
    Loading hack inspired by some online blog post lol.
    """

    data_dir = Path(data_dir)

    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"
    if not (idf_loc.exists() and vocab_loc.exists()):
        collect_stats(data_dir)

    idf = np.load(idf_loc)
    with open(vocab_loc, "r") as f:
        vocab = json.load(f)

    class MyVectorizer(TfidfVectorizer):
        TfidfVectorizer.tfidf_ = idf

    vec = MyVectorizer()

    # Fit the vectorizer to some dummy data
    vec.fit(['dummy document'])

    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))
    vec._tfidf.n_features_in_ = len(vocab)

    return vec


def collect_stats(data_dir: str):
    """
    Uses wikipedia snippets to collect statistics over a corpus of English text.
    Retrieved later when computing TF-IDF vectors.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"

    try:
        print(f"Downloading IDF cache from {REMOTE_IDF_URL}")
        torch.hub.download_url_to_file(REMOTE_IDF_URL, idf_loc)
        print(f"Downloading TF-IDF vocab cache from {REMOTE_VOCAB_URL}")
        torch.hub.download_url_to_file(REMOTE_VOCAB_URL, vocab_loc)
        return
    except Exception as e:
        print(f"Error downloading file:", e)
        print("Recomputing TF-IDF stats...")

    snips_list = AttributeSnippets(data_dir).snippets_list
    documents = list(chain(*[[y["text"] for y in x["samples"]] for x in snips_list]))

    vec = TfidfVectorizer()
    vec.fit(documents)

    idfs = vec.idf_
    vocab = vec.vocabulary_

    np.save(data_dir / "idf.npy", idfs)
    with open(data_dir / "tfidf_vocab.json", "w") as f:
        json.dump(vocab, f, indent=1)


def latent_code_from_text_single(text, tokenizer_encoder, model_vae, device):
    tokenized1 = tokenizer_encoder.encode(text)
    tokenized1 = [101] + tokenized1 + [102]
    coded1 = torch.Tensor([tokenized1])
    coded1 = torch.Tensor.long(coded1)
    with torch.no_grad():
        x0 = coded1
        x0 = x0.to(device)
        pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
        mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        latent_z = mean.squeeze(1)
        coded_length = len(tokenized1)
        return latent_z, coded_length


def latent_code_from_text(text, tokenizer_encoder, model_vae, device):
    if isinstance(text, list):
        latents = []
        coded_lengths = []
        for sent in text:
            z, cl = latent_code_from_text_single(sent, tokenizer_encoder, model_vae, device)
            latents.append(z)
            coded_lengths.append(cl)
        return torch.vstack(latents).to(device), coded_lengths
    else:
        return latent_code_from_text_single(text, tokenizer_encoder, model_vae, device)


def sample_sequence_conditional(model,
                                length,
                                context,
                                past=None,
                                temperature=1,
                                top_k=0,
                                top_p=0.0,
                                device='cpu',
                                decoder_tokenizer=None,
                                tok_true=None,
                                tok_new=None):
    generated = context
    i = 0
    finished = []
    for j in range(generated.size(0)):
        finished.append(False)
    finished = torch.tensor(finished).to(device)
    with torch.no_grad():
        while True:
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering_batch(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            finished = torch.logical_or(finished, next_token[:, 0] == decoder_tokenizer.encode('<EOS>')[0])
            if finished.sum() == next_token.size(0) or i == length:
                break
            i += 1
    return generated


def text_from_latent_code_mask(latent_z,
                               model_vae,
                               tokenizer_decoder,
                               length,
                               prompt='',
                               is_vae=True,
                               temperature=0.7,
                               top_k=50,
                               top_p=0.95,
                               device='cuda'):
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
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        decoder_tokenizer=tokenizer_decoder)

    texts = []
    eos_token_val = tokenizer_decoder.encode('<EOS>')[0]
    for i in range(latent_z.size(0)):
        tokens = out[i, 1:].tolist()  # filter out BOS
        if eos_token_val in tokens:
            loc = tokens.index(eos_token_val)
            tokens = tokens[:loc]  # end before EOS
        text_x1 = tokenizer_decoder.decode(tokens, clean_up_tokenization_spaces=True)
        # print(text_x1)
        text_x1 = text_x1.split()  # [1:-1]
        text_x1 = ' '.join(text_x1)
        texts.append(text_x1)

    if len(texts) == 1:
        texts = texts[0]

    texts = unicodedata.normalize("NFKD", texts).replace("\n\n", " ")

    return texts


def compute_rewrite_quality_zsre(model, tokenizer, record, posterior_memory, scope_detector=None, no_memory=False):

    _, tok = tokenizer  #enc, dec tokenizer

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok.encode(" " + target_new["str"])
    inp_prompts_og = list(chain(*prob_prompts))

    inp_prompts = [
        el + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]

    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    inp_targets_tok = [
        target_tok[i]
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tokenizer, inp_prompts, inp_targets_tok,
                                            posterior_memory, scope_detector, no_memory=no_memory)

    n_prompts = [el["prompt"].format(record["requested_rewrite"]) for el in neighborhood_prompts]
    n_targets = [el["target"] for el in neighborhood_prompts]
    n_targets_tok = [el["target_tok"] for el in neighborhood_prompts]

    # Predict for neighborhood prompts (dictionary format).
    neighborhood_correct = test_batch_prediction_acc(model, tokenizer, n_prompts, n_targets_tok,
                                                     posterior_memory, scope_detector, no_memory=no_memory)
    # print(f'eval time: {time()-t1}')

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum([l * len(target_tok) for l in map(len, prob_prompts)]).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]

    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct
    ret['rewrite_paraphrase_prompts'] = inp_prompts
    ret['rewrite_paraphrase_targets'] = inp_targets
    ret['neighborhood_prompts'] = n_prompts
    ret['neighborhood_targets'] = n_targets


    return ret


def test_batch_prediction_acc(model, tok, prompts, target, posterior_memory, scope_detector=None, no_memory=False):

    enc_tok, dec_tok = tok

    encoded_read, cls = latent_code_from_text(prompts, enc_tok, model, device='cuda')
    zs, _ = model.read(encoded_read.unsqueeze(1), posterior_memory, deterministic=True)
    zs = zs.squeeze(1)

    if False: #latent_z is None:
        tok.pad_token = tok.eos_token
        prompt_tok = tok(prompts, padding=True, return_tensors="pt").to("cuda")
    else:
        ##### PyriteScopeMemory
        if scope_detector is not None:
            assert len(scope_detector.embeddings) > 0, "ScopeDetector is not initialized"
            scope_info = [(scope_detector.is_indomain(s), s) for s in prompts]
        ##### PyriteScopeMemory -> scope_info holds the indomain detection results for all data

        prompts = [f"<BOS> {d}" for d in prompts]
        encoded = [torch.tensor(dec_tok.encode(d), dtype=torch.long) for d in prompts]
        prompt_tok = pad_sequence(encoded, batch_first=True, padding_value=0).to('cuda')

    if False: #latent_z is None:
        with torch.no_grad():
            logits = model(**prompt_tok).logits

        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
    else:
        if no_memory:
            past = torch.zeros_like(zs)
        else:
            past = zs

        if scope_detector is not None:
            for i, e in enumerate(scope_info):
                if not e[0][0]:
                    past[i] = 0

        with torch.no_grad():
            logits = model.decoder(prompt_tok, past)[0]

        last_non_masked = torch.tensor([len(e)-1 for e in encoded]).cuda()

    to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)

    gathered = torch.gather(logits, 1, to_gather).squeeze(1)
    ans = torch.argmax(gathered, dim=1)

    # correct_id = tok.encode(target, padding=True, return_tensors="pt").to("cuda")
    correct_id = torch.cat([torch.tensor([d], dtype=torch.long, device=torch.device("cuda")) for d in target])
    # correct_id = torch.cat([torch.tensor(tok.encode(d), dtype=torch.long, device=torch.device("cuda")) for d in target])

    return (ans == correct_id).detach().cpu().numpy().tolist()


def test_batch_prediction(
        model,
        tok,
        prefixes,
        target_new,
        target_true,
        posterior_memory,
        sigma_w,
        scope_detector=None,
        paraphrase_prompts=None,
        no_memory=False
):

    if False: # latent_z is None:
        tok.pad_token = tok.eos_token
        prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]

        data = [f"{prefix} {suffix}"
                for prefix in prefixes
                for suffix in [target_new, target_true]]

        prompt_tok = tok(data, padding=True, return_tensors="pt").to("cuda")
    else:
        enc_tok, dec_tok = tok

        prefix_lens = []
        for p in prefixes:
            prefix_lens.append(len(dec_tok.encode(p)) + 1)  # +1 for <BOS>

        encoded_read, cls = latent_code_from_text(prefixes, enc_tok, model, device='cuda')
        zs, _ = model.read(encoded_read.unsqueeze(1), posterior_memory, deterministic=True, sigma_w=sigma_w)
        zs = zs.squeeze()
        zs = zs.repeat_interleave(2, dim=0)  # target_new, target_true for each prompt

        ##### PyriteScopeMemory
        if scope_detector is not None:
            assert len(scope_detector.embeddings) > 0, "ScopeDetector is not initialized"
            data = [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]]
            scope_info = [(scope_detector.is_indomain(s), s) for s in data]
        ##### PyriteScopeMemory -> scope_info holds the indomain detection results for all data

        data = [f"<BOS> {prefix} {suffix}"
                for prefix in prefixes
                for suffix in [target_new, target_true]]

        prompt_tok = pad_sequence([torch.tensor(dec_tok.encode(d), dtype=torch.long) for d in data], batch_first=True, padding_value=0).to('cuda')

    a_tok = dec_tok.encode(target_new)
    b_tok = dec_tok.encode(target_true)

    # a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    if False: # latent_z is None:
        with torch.no_grad():
            logits = model(**prompt_tok)[0]
    else:
        # past = latent_z.repeat(prompt_tok.size(0), 1)
        if no_memory:
            past = torch.zeros_like(zs)
        else:
            past = zs

        if scope_detector is not None:
            for i, e in enumerate(scope_info):
                if not e[0][0]:
                    past[i] = 0

        with torch.no_grad():
            logits = model.decoder(prompt_tok, past)[0]

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            try:
                results[i] += -torch.nn.functional.log_softmax(logits[i, prefix_lens[i // 2] + j - 1, :], dim=0)[cur_tok].item()
            except:
                continue
        results[i] /= cur_len

    gen_para = []

    res =  [{"target_new": results[i].item(),
             "target_true": results[i + 1].item(),
             "prompt_new": data[i],
             "prompt_true": data[i+1]}
            for i in range(0, len(results), 2)]

    return res, gen_para


def compute_rewrite_quality_counterfact(model, tok, record, snips, vec, posterior_memory, sigma_w,
                                        scope_detector=None, instructions=None, counterfact_plus=None,
                                        no_memory=False):
    # First, unpack rewrite evaluation record
    subject, target_new, target_true = (record["requested_rewrite"][x] for x in
                                        ["subject", "target_new", "target_true"])
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    attribute_prompts = record["attribute_prompts"]
    generation_prompts = record["generation_prompts"]

    para_prompts = []
    for p in paraphrase_prompts:
        para_prompts.append(p.split('. ')[-1])
    paraphrase_prompts = para_prompts

    if instructions:
        prompt = rewrite_prompts[0]

        new_rewrite_prompt = f"New Fact: {prompt} {target_new['str']}. "
        # new_rewrite_prompt = new_rewrite_prompt * 4
        rewrite_prompts_instructions = [new_rewrite_prompt + ' ' + prompt]

        paraphrase_prompts_instructions = []
        for p in paraphrase_prompts:
            newp = f'{new_rewrite_prompt} {p}'
            paraphrase_prompts_instructions.append(newp)

        neighborhood_prompts_instructions = []
        for p in neighborhood_prompts:
            newp = f'{new_rewrite_prompt} {p}'
            neighborhood_prompts_instructions.append(newp)

        attribute_prompts_instructions = []
        for p in attribute_prompts:
            newp = f'{new_rewrite_prompt} {p}'
            attribute_prompts_instructions.append(newp)

            # Form a list of lists of prefixes to test.
        prob_prompts = [
            rewrite_prompts_instructions,
            paraphrase_prompts_instructions,
            neighborhood_prompts_instructions,
            attribute_prompts_instructions,
        ]

        generation_prompts_instructions = []
        for p in generation_prompts:
            newp = f'{new_rewrite_prompt} {p}'
            generation_prompts_instructions.append(newp)
        generation_prompts = generation_prompts_instructions
    else:

        if counterfact_plus:
            prompt = rewrite_prompts[0]
            new_rewrite_prompt = f"{prompt} {target_new['str']}. "

            neighborhood_prompts_instructions = []
            for p in neighborhood_prompts:
                newp = f'{new_rewrite_prompt} {p}'
                neighborhood_prompts_instructions.append(newp)

            neighborhood_prompts = neighborhood_prompts_instructions

        # Form a list of lists of prefixes to test.
        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
            attribute_prompts,
        ]

    # t1 = time()
    # Flatten all the evaluated prefixes into one list.
    probs, gen_para = test_batch_prediction(model,
                                          tok,
                                          list(chain(*prob_prompts)),
                                          target_new["str"],
                                          target_true["str"],
                                          posterior_memory,
                                          sigma_w,
                                          scope_detector=scope_detector,
                                          no_memory=no_memory)

    # print(f'eval time {time()-t1}')

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))]

    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]

        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]

        assert (len(consistency_texts) > 0), "Must have consistency texts to evaluate generation"

        gen_stats = test_generation(
            latent_z,
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )

        ret.update(gen_stats)

    return ret, gen_para


def run_base_model(text, model, tokenizer, do_sample=True, device='cuda', temperature=0.7, length=100, top_k=50, top_p=0.95):
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    output = model.generate(**encoded_input, temperature=temperature, do_sample=do_sample, max_new_tokens=length, top_k=top_k, top_p=top_p)
    output_text = tokenizer.decode(output[0])
    return output_text


def test_generation(
        latent_z,
        model,
        tok,
        prefixes: typing.List[str],
        consistency_texts: typing.List[str],
        essence_texts: typing.List[str],
        vec: TfidfVectorizer,
):

    gen_texts = []
    prompt_gen = []
    for p in prefixes:
        if latent_z is None:
            gen_text = run_base_model(p, model, tok, do_sample=True, device='cuda', temperature=0.7, length=100, top_k=50, top_p=0.95)
        else:
            gen_text = text_from_latent_code_mask(latent_z,
                                                  model,
                                                  tok,
                                                  length=100,
                                                  prompt=p)
        gen_texts.append(gen_text)
        prompt_gen.append({'prompt': p, 'gen': gen_text})

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        if latent_z is None:
            ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        else:
            ppl = perplexity_pyrite(latent_z, model, tok, " ".join(essence_texts), max_input_length=100)

        ret.update({"essence_score": ppl, "essence_text": '', 'ppl_text': prompt_gen})

    return ret

def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)([compute_n_gram_entropy(txt) for txt in gen_texts]).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()


def perplexity_pyrite(latent_z, model, tok, text, max_input_length):
    # inputs = tok([ text], return_tensors="pt", max_length=max_input_length, truncation=True).to("cuda")

    # import pdb; pdb.set_trace()

    inputs = torch.tensor(tok.encode('<BOS>' + text)[:max_input_length], dtype=torch.long).to('cuda')
    inputs = inputs.unsqueeze(0)

    with torch.no_grad():
        logits = model.decoder(inputs, latent_z)[0]

    logits = torch.nn.functional.log_softmax(logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs[:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs.size(1) * log_probs.sum()).item()


def perplexity(model, tok, text, max_input_length):
    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()


def gen_ppl(model, tokenizer, text, mode='pyrite', deterministic=True, sigma_w=None):

    # use the first 10 words of the input as prompt
    prompt = text[0][:10]

    if mode in ['pyrite', 'combo', 'unconstrained']:

        enc_tok, dec_tok = tokenizer

        # First write to the memory
        memory_write_text_encoded, cls = latent_code_from_text(text, enc_tok, model, device='cuda')
        memory_write_text_encoded = memory_write_text_encoded.reshape(len(text), 1, model._code_size)
        posterior_memory, dkl_M = model.write(input_encoded=memory_write_text_encoded)

        # Then read from the memory
        encoded_read, cls = latent_code_from_text(text, enc_tok, model, device='cuda')
        encoded_read = encoded_read.reshape(1, 1, model._code_size)

        if deterministic:
            model.memory.deterministic = True
            model.memory._observation_noise_std = 0.0

        zs, _ = model.read(encoded_read, posterior_memory, deterministic=deterministic, sigma_w=sigma_w)
        # zs, _ = model.read(encoded_read, posterior_memory)
        latent_z = zs.reshape(1, model._code_size)

        if mode == 'unconstrained':
            latent_z = torch.zeros_like(latent_z)

        gen_text = text_from_latent_code_mask(latent_z, model, dec_tok, length=300, prompt=prompt)

    else:

        if mode == 'base_icl':
            prompt = f"Fact: {text[0]}. {prompt}"

        gen_text = run_base_model(prompt, model, tokenizer, do_sample=True, device='cuda', temperature=0.7,
                                  length=300, top_k=50, top_p=0.95)

    return gen_text


def eval(model, tokenizer, record, dataset, snips, vec, mode='pyrite', deterministic=True,
         scope_detection_threshold=None, counterfact_plus=None, sigma_w=None):

    request = record["requested_rewrite"]
    text_write_to_mem = [request['prompt'].format(request['subject']) + " " + request['target_new']['str']]
    text_to_query = [request['prompt'].format(request['subject'])]

    if mode == 'pyrite' or mode == 'combo':

        enc_tok, dec_tok = tokenizer

        if deterministic:
            model.memory.deterministic = True
            model.memory._observation_noise_std = 0.0

        # t1 = time()
        # First write to the memory
        memory_write_text_encoded, cls = latent_code_from_text(text_write_to_mem, enc_tok, model, device='cuda')
        memory_write_text_encoded = memory_write_text_encoded.reshape(len(text_write_to_mem), 1, model._code_size)
        posterior_memory, dkl_M = model.write(input_encoded=memory_write_text_encoded)

        # Then read from the memory
        # encoded_read, cls = latent_code_from_text(text_to_query, enc_tok, model, device='cuda')
        # encoded_read = encoded_read.reshape(1, 1, model._code_size)
        #
        # zs, _ = model.read(encoded_read, posterior_memory, deterministic=deterministic, sigma_w=sigma_w)
        # zs = zs.reshape(1, model._code_size)

        scope_detector = None
        if scope_detection_threshold is not None:  # activate scope detection
            scope_detector = pss.ScopeMemory(modelname='sentence-transformers/all-MiniLM-L6-v2',
                                             normalize=False,
                                             dist_threshold=scope_detection_threshold,
                                             verbose=False)
            scope_detector.add_sentences(text_write_to_mem)  # adds memory content to scope

        # print(f'memory time: {time()-t1}')
        if dataset == 'counterfact':
            output = compute_rewrite_quality_counterfact(model, tokenizer, record, snips, vec, posterior_memory, sigma_w,
                                                        scope_detector=scope_detector,
                                                        counterfact_plus=counterfact_plus)
        else:  # 'zsre'
            output = compute_rewrite_quality_zsre(model, tokenizer, record, posterior_memory, scope_detector=scope_detector)

    elif mode == 'unconstrained':

        enc_tok, dec_tok = tokenizer

        if deterministic:
            model.memory.deterministic = True
            model.memory._observation_noise_std = 0.0

        # First write to the memory
        memory_write_text_encoded, cls = latent_code_from_text(text_write_to_mem, enc_tok, model, device='cuda')
        memory_write_text_encoded = memory_write_text_encoded.reshape(len(text_write_to_mem), 1, model._code_size)
        posterior_memory, dkl_M = model.write(input_encoded=memory_write_text_encoded)

        # # Then read from the memory
        # encoded_read, cls = latent_code_from_text(text_to_query, enc_tok, model, device='cuda')
        # encoded_read = encoded_read.reshape(1, 1, model._code_size)
        #
        # if deterministic:
        #     model.memory.deterministic = True
        #     model.memory._observation_noise_std = 0.0
        #
        # zs, _ = model.read(encoded_read, posterior_memory)
        # zs = zs.reshape(1, model._code_size)

        if dataset == 'counterfact':
            output = compute_rewrite_quality_counterfact(model, tokenizer, record, snips, vec, posterior_memory, sigma_w,
                                                         counterfact_plus=counterfact_plus,
                                                         no_memory=True)
        else:  # 'zsre'
            output = compute_rewrite_quality_zsre(model, tokenizer, record, posterior_memory, no_memory=True)


    elif mode == 'base':
        output = compute_rewrite_quality_counterfact(None, model, tokenizer, record, snips, vec)

    else:  # mode == 'base_icl'
        output = compute_rewrite_quality_counterfact(None, model, tokenizer, record, snips, vec, instructions=True)

    return output


def pl_load_pyrite(checkpoint_dir):
    model = MemNetLight.load_from_checkpoint(checkpoint_dir, optimizer=None)
    model.model.to('cuda')
    return model.model, (model.tokenizer_encoder, model.tokenizer_decoder)


def load_pyrite(checkpoint_dir,
                hparams_file=None,
                map_location=None):
    model = MemNetLight.load_from_checkpoint(checkpoint_dir,
                                             strict=False,
                                             optimizer=None,
                                             decode_rec_strength=0,
                                             use_ramp_both_sides=False,
                                             ratio_one=0.0,
                                             hparams_file=hparams_file,
                                             map_location=map_location,
                                             use_bitfit=False)
    model.model.to('cuda')
    return model.model, (model.tokenizer_encoder, model.tokenizer_decoder)



def load_base(name, cache_dir):
    base_model=AutoModelForCausalLM.from_pretrained(name, cache_dir)
    base_model=base_model.to('cuda')
    base_tokenizer=AutoTokenizer.from_pretrained(name, cache_dir)
    return base_model, base_tokenizer


def main(args):
    res_dir = Path(args.res_dir_name)
    if res_dir.exists():
        id_list = [int(str(x).split("_")[1])
                   for x in res_dir.iterdir()
                   if str(x).split("_")[1].isnumeric()]
        run_id = 0 if not id_list else max(id_list) + 1
    else:
        run_id = 0

    if args.mode == 'pyrite' and args.scope_detect_threshold:
        md = 'combo'
    elif args.mode == 'base' or args.mode == 'base_icl':
        md = f'{args.mode}_{args.base_name}'
    else:
        md = args.mode

    run_dir = res_dir / f"run_{str(run_id).zfill(3)}_{md}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    if args.mode in ['base', 'base_icl']:
        model, tokenizer = load_base(args.base_name, args.cache_dir)
    else:
        model, tokenizer = load_pyrite(args.checkpoint)

    final_result = {}

    if args.dataset:
        # Load data
        if args.dataset == 'zsre':
            print("Loading ZSRE dataset")
            enc_tok, dec_tok = tokenizer
            ds = MENDQADataset(args.data_dir, tok=dec_tok, size=args.num_eval_cases)
            snips, vec = None, None
        else:  # countefact
            print("Loading dataset, attribute snippets, tf-idf data")
            snips = AttributeSnippets(args.data_dir)
            vec = get_tfidf_vectorizer(args.data_dir)
            ds = CounterFactDataset(args.data_dir, size=args.num_eval_cases)
            snips = None

        # Iterate through dataset
        for record in tqdm(ds):

            case_id = record["case_id"]
            case_result_path = run_dir / f"case_{case_id}.json"

            if not case_result_path.exists():
                out = eval(model=model,
                           tokenizer=tokenizer,
                           record=record,
                           dataset=args.dataset,
                           snips=snips,
                           vec=vec,
                           mode=args.mode,
                           scope_detection_threshold=args.scope_detect_threshold,
                           counterfact_plus=args.counterfact_plus,
                           sigma_w=args.sigma_w)

                if isinstance(out, tuple):
                    # Execute evaluation suite
                    metrics = {
                        "case_id": case_id,
                        "requested_rewrite": record["requested_rewrite"],
                        "post": out[0],
                        "gen_para": out[1]
                    }
                else:
                    # Execute evaluation suite
                    metrics = {
                        "case_id": case_id,
                        "requested_rewrite": record["requested_rewrite"],
                        "post": out,
                    }

                # Dump metrics in .json
                with open(case_result_path, "w") as f:
                    json.dump(metrics, f, indent=1)

        results = process_results(dir_name=res_dir, runs=[str(run_dir)])

        if args.dataset == 'counterfact':
            # final_result = {'rewrite/efficacy': [results[0]['post_rewrite_success'][0],
            #                                      results[0]['post_rewrite_diff'][0]],
            #                 'paraphrase/generalization': [results[0]['post_paraphrase_success'][0],
            #                                               results[0]['post_paraphrase_diff'][0]],
            #                 'neighborhood/specificity': [results[0]['post_neighborhood_success'][0],
            #                                              results[0]['post_neighborhood_diff'][0]],
            #                 'fluency/ngram_entropy': results[0]['post_ngram_entropy'][0],
            #                 'consistency/reference': results[0]['post_reference_score'][0],
            #                 'ppl': results[0]['post_essence_score'][0],
            #                 'ppl2': -1,
            #                 'model': args.checkpoint}

            final_result = {'rewrite/efficacy': [results[0]['post_rewrite_success'][0],
                                                 results[0]['post_rewrite_diff'][0]],
                            'paraphrase/generalization': [results[0]['post_paraphrase_success'][0],
                                                          results[0]['post_paraphrase_diff'][0]],
                            'neighborhood/specificity': [results[0]['post_neighborhood_success'][0],
                                                         results[0]['post_neighborhood_diff'][0]],
                            'model': args.checkpoint}
        else:
            final_result = {'rewrite/efficacy': results[0]['post_rewrite_acc'][0],
                            'paraphrase/generalization': results[0]['post_paraphrase_acc'][0],
                            'neighborhood/specificity': results[0]['post_neighborhood_acc'][0],
                            'model': args.checkpoint}
    if args.ppl2:
        device = "cuda"
        model_id = "gpt2-large"
        evaluator = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=args.cache_dir).to(device)
        tok_eval = GPT2TokenizerFast.from_pretrained(model_id, cache_dir=args.cache_dir)

        with open(args.ppl2_data_path, 'r') as f:
            sentences = f.readlines()

        all_gen_text = []

        for k in tqdm(range(args.ppl2_num_cases)):
            # t1 = time()
            gen_text = gen_ppl(model, tokenizer, [sentences[k]], mode=args.mode, sigma_w=args.sigma_w)
            # print(f'gen time {time()-t1}')
            # breakpoint()
            out_text = gen_text[:400]
            out_text = out_text.replace("\\n", " ").replace("\n", " ").replace("\\", " ")
            all_gen_text.append(out_text)

        text = " ".join(all_gen_text)
        # encodings = tok_eval(text, return_tensors="pt")
        max_length = 10000
        max_context = evaluator.config.n_positions
        stride = 1000
        seq_len = len(text) #encodings.input_ids.size(1)

        nlls = []
        for begin_loc in tqdm(range(0, seq_len, stride)):

            end_loc = min(begin_loc + max_length, seq_len)
            # trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            # input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            # target_ids = input_ids.clone()
            # target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                encoded_input = tok_eval(text[begin_loc:end_loc], return_tensors='pt')  # this roughly needs to convert to 1024 tokens
                encoded_input = encoded_input.to('cuda')
                inp = encoded_input['input_ids'][:, :max_context]
                outputs = evaluator(input_ids=inp, labels=inp)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            if end_loc == seq_len:
                break

        ppl2 = torch.exp(torch.stack(nlls).mean()).item()
        final_result['ppl2'] = ppl2

    pprint(final_result)
    json.dump(final_result, open(f"{run_dir}/final_result.json", 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir_name", type=str, default='results')
    parser.add_argument("--dataset", type=str, default='', choices=['', 'counterfact', 'zsre'])
    parser.add_argument("--counterfact_plus", action="store_true", help="include counterfact+ idea of neighborhood promtps")
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument("--data_dir", type=str, default='../data')
    parser.add_argument("--num_eval_cases", type=int, default=None)
    parser.add_argument("--mode", type=str, default='pyrite', choices=['pyrite', 'unconstrained', 'base', 'base_icl'])
    parser.add_argument("--base_name", type=str, default='gpt2-xl')
    parser.add_argument("--scope_detect_threshold", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ppl2", action="store_true", help="do ppl based on gpt-large eval of gen text")
    parser.add_argument("--ppl2_data_path", type=str, default='../data/wikipedia/blocksize_64/test.txt')
    parser.add_argument("--ppl2_num_cases", type=int, default=1000)
    parser.add_argument("--sigma_w", type=float, default=None)


    args = parser.parse_args()

    main(args)
