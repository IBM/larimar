import argparse
import lightning as pl
import logging
import torch
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from utils import text_from_latent_code
import nltk
from torchmetrics import Metric

try:
    from apex.optimizers import FusedAdam
except ImportError:
    print('Install apex from https://github.com/NVIDIA/apex/ if you plan to use FusedAdam')

from deepspeed.ops.adam import DeepSpeedCPUAdam

from pytorch_transformers import (AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  GPTJConfig, GPTJForLatentConnector,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

#                                   JinaBertConfig, JinaBertForLatentConnector,
from transformers import get_linear_schedule_with_warmup

from modules import MemVAE, VAE
from utils import (BucketingDataLoader, TextDataset_Split, TextDataset_2Tokenizers, frange_cycle_zero_linear, frange_cycle_both_ramp)
from run_latent_generation_memvae import denoise

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'gptj': (GPTJConfig, GPTJForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
#    'jina_bert': (JinaBertConfig, JinaBertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}


def mask_tokens(inputs, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).to(torch.uint8)
    labels[masked_indices == 1] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
    indices_random = indices_random
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def prepare_enc_dec_tokenizer(encoder_model_type,
                              encoder_model_name_or_path,
                              decoder_model_type,
                              decoder_model_name_or_path,
                              cache_dir,
                              do_lower_case,
                              block_size):
    # ======== ENCODER =========
    _, _, encoder_tokenizer_class = MODEL_CLASSES[encoder_model_type]

    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(encoder_model_name_or_path,
                                                                do_lower_case=do_lower_case,
                                                                cache_dir=cache_dir)

    if block_size <= 0:
        block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model

    block_size = min(block_size, tokenizer_encoder.max_len_single_sentence)

    # ======== DECODER =========
    _, _, decoder_tokenizer_class = MODEL_CLASSES[decoder_model_type]

    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(decoder_model_name_or_path,
                                                                do_lower_case=do_lower_case,
                                                                cache_dir=cache_dir)

    # TODO block_size is the same for encoder and decoder?

    block_size = min(block_size, tokenizer_decoder.max_len_single_sentence)

    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    logger.info(f'Added {num_added_toks} tokens to GPT2')
    assert tokenizer_decoder.pad_token == '<PAD>'

    return tokenizer_encoder, tokenizer_decoder


def prepare_enc_dec(encoder_model_type,
                          encoder_model_name_or_path,
                          decoder_model_type,
                          decoder_model_name_or_path,
                          cache_dir,
                          latent_size,
                          do_lower_case,
                          block_size):

    tokenizer_encoder, tokenizer_decoder = prepare_enc_dec_tokenizer(encoder_model_type,
                                                                      encoder_model_name_or_path,
                                                                      decoder_model_type,
                                                                      decoder_model_name_or_path,
                                                                      cache_dir,
                                                                      do_lower_case,
                                                                      block_size)

    # ======== ENCODER =========
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[encoder_model_type]
    encoder_config = encoder_config_class.from_pretrained(encoder_model_name_or_path,
                                                          cache_dir=cache_dir)
    setattr(encoder_config, "latent_size", latent_size)

    model_encoder = encoder_model_class.from_pretrained(encoder_model_name_or_path,
                                                        from_tf=bool('.ckpt' in encoder_model_name_or_path),
                                                        config=encoder_config,
                                                        latent_size=latent_size,
                                                        cache_dir=cache_dir)

    # ======== DECODER =========
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[decoder_model_type]
    decoder_config = decoder_config_class.from_pretrained(decoder_model_name_or_path,
                                                          cache_dir=cache_dir)
    setattr(decoder_config, "latent_size", latent_size)
    model_decoder = decoder_model_class.from_pretrained(decoder_model_name_or_path,
                                                        from_tf=bool('.ckpt' in decoder_model_name_or_path),
                                                        config=decoder_config, latent_size=latent_size,
                                                        cache_dir=cache_dir)


    # D = pickle.load(open('tmp.pkl', 'rb'))
    # input_ids = torch.tensor(D['input_ids'])
    # past = torch.tensor(D['past'])
    # labels = torch.tensor(D['labels'])
    # label_ignore = D['label_ignore']

    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    assert tokenizer_decoder.pad_token == '<PAD>'

    # outputs = model_decoder(input_ids=input_ids, past=past, labels=labels, label_ignore=label_ignore)

    return model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder


class EvalMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("bleu", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec_z", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec_ae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("kl", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_sents", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_words", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, bleu=0, rec=0, rec_z=0, rec_ae=0, kl=0, loss=0, num_sents=0, num_words=0):
        self.bleu += bleu
        self.rec += rec
        self.rec_z += rec_z
        self.rec_ae += rec_ae
        self.kl += kl
        self.loss += loss
        self.num_sents += num_sents
        self.num_words += num_words

    def compute(self):

        # hack to avoid error if it was unused
        if self.num_sents == 0:
            self.num_sents = 1
        if self.num_words == 0:
            self.num_words = 1

        results = {'bleu': self.bleu / self.num_sents,
                   'elbo': (self.kl-self.rec) / self.num_sents,
                   'nll': - self.rec / self.num_sents,
                   'nll_z': - self.rec_z / self.num_sents,
                   'kl': self.kl / self.num_sents,
                   'ppl': torch.exp(self.loss / self.num_words),
                   'ppl_ae': torch.exp(self.rec_ae / self.num_words),
                   'nll_ae': self.rec_ae / self.num_sents}

        return results


class MemNetLight(pl.LightningModule):

    def __init__(self,
                 encoder_model_type,
                 encoder_model_name_or_path,
                 cache_dir,
                 latent_size,
                 do_lower_case,
                 block_size,
                 decoder_model_type,
                 decoder_model_name_or_path,
                 memory_size,
                 learning_rate,
                 adam_epsilon,
                 weight_decay,
                 warmup_steps,
                 mlm,
                 mlm_probability,
                 direct_writing,
                 ordering,
                 pseudoinverse_approx_step,
                 observation_noise_std,
                 episode_sizes,
                 identity,
                 w_logvar_setting,
                 deterministic_w,
                 dim_target_kl,
                 length_weighted_loss,
                 rec_strength,
                 ae_strength,
                 l2_strength,
                 beta,
                 decode_rec_strength,
                 use_beta_schedule,
                 use_ramp_both_sides,
                 ratio_one,
                 ratio_zero,
                 ratio_increase,
                 fb_mode,
                 optimizer,
                 temperature,
                 top_k,
                 top_p,
                 load_pretrained,
                 perturb,
                 bleu,
                 ae_only,
                 num_samples,
                 ae_read_write,
                 read_iters):
        super().__init__()
        self.save_hyperparameters()

        self.metrics = EvalMetrics()
        print("MemNetLight init()")
        print("encoder_model_type", encoder_model_type)
        print("encoder_model_name_or_path", encoder_model_name_or_path)
        print("cache_dir", cache_dir)
        print("load_pretrained", load_pretrained)

        model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder = prepare_enc_dec(encoder_model_type,
                                                                                             encoder_model_name_or_path,
                                                                                             decoder_model_type,
                                                                                             decoder_model_name_or_path,
                                                                                             cache_dir,
                                                                                             latent_size,
                                                                                             do_lower_case,
                                                                                             block_size)

        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        self.beta_t_list = None

        # ========= MEMORY ===========

        if memory_size > 0:
            self.model = MemVAE(model_encoder,
                                model_decoder,
                                tokenizer_encoder,
                                tokenizer_decoder,
                                memory_size,
                                latent_size,
                                direct_writing,
                                ordering,
                                pseudoinverse_approx_step,
                                observation_noise_std,
                                episode_sizes,
                                identity,
                                w_logvar_setting,
                                deterministic_w,
                                dim_target_kl,
                                length_weighted_loss,
                                rec_strength,
                                ae_strength,
                                l2_strength,
                                beta,
                                decode_rec_strength)

        else:
            self.model = VAE(model_encoder,
                             model_decoder,
                             tokenizer_encoder,
                             tokenizer_decoder,
                             latent_size,
                             fb_mode,
                             dim_target_kl,
                             length_weighted_loss,
                             beta)


        if load_pretrained:
            print(f"LOADING PRETRAINED WEIGHTS..{load_pretrained}")
            load_path = '<<<Please provide a model checkpoint path>>>'
            print("loading...", load_path)
            ckpt = torch.load(load_path)
            self.model.load_state_dict(ckpt, strict=False)


    def training_step(self, batch, batch_idx):

        tokenized_text0, tokenized_text1, tokenized_text_lengths = batch

        # print(f"Processed by GPU {torch.cuda.current_device()}, {batch_idx = }, {len(self.trainer.train_dataloader)}, {isinstance(self.trainer.train_dataloader.sampler, DistributedSampler)}")

        if self.hparams.mlm:
            inputs, labels = mask_tokens(tokenized_text0, self.hparams.tokenizer_encoder, self.hparams.mlm_probability)
        else:
            inputs, labels = tokenized_text0, tokenized_text1

        #TODO: fix here?
        labels = tokenized_text1

        if self.hparams.use_beta_schedule:
            if self.beta_t_list is None:
                # n_iter = self.trainer.train_dataloader.num_examples * self.trainer.max_epochs
                # n_iter = self.trainer.num_training_batches * self.trainer.max_epochs
                n_iter = self.trainer.estimated_stepping_batches
                if self.hparams.use_ramp_both_sides:
                    print("Using RAMP annealing...")
                    self.beta_t_list = frange_cycle_both_ramp(n_iter,
                                                              start=0.0,
                                                              stop=self.hparams.beta,
                                                              n_cycle=10,
                                                              ratio_increase=self.hparams.ratio_increase,
                                                              ratio_zero=self.hparams.ratio_zero,
                                                              ratio_one=self.hparams.ratio_one)
                else:
                    self.beta_t_list = frange_cycle_zero_linear(n_iter,
                                                                start=0.0,
                                                                stop=self.hparams.beta,
                                                                n_cycle=10,
                                                                ratio_increase=self.hparams.ratio_increase,
                                                                ratio_zero=self.hparams.ratio_zero)

            if self.global_step >= len(self.beta_t_list):
                beta_t = self.hparams.beta  # = 1.0
            else:
                beta_t = self.beta_t_list[self.global_step]
        else:
            beta_t = self.hparams.beta

        if self.hparams.memory_size > 0:
            # loss_rec, bijective_loss, loss_kl, l2_loss, loss_rec_decoded, loss

            loss_rec, loss_rec_z, loss_kl, loss_l2, loss_rec_decoded, loss = self.model(inputs, labels, train=True, beta_t=beta_t)

            loss_rec = loss_rec.mean()
            loss_rec_z = loss_rec_z.mean()
            loss_l2 = loss_l2.mean()
            loss_kl = loss_kl.mean()
            loss_rec_decoded = loss_rec_decoded.mean()
            loss = loss.mean()
        else:
            loss_rec, loss_kl, loss = self.model(inputs, labels)

            loss_rec_z = loss_rec.mean()
            loss_kl = loss_kl.mean()
            loss = loss.mean()
            loss_rec = 0
            loss_l2 = 0
            loss_rec_decoded = 0

        self.log_dict({'train/REC': loss_rec,
                       'train/REC_Z': loss_rec_z,
                       'train/KL': loss_kl,
                       'train/L2': loss_l2,
                       'train/REC_DECODER': loss_rec_decoded,
                       'train/LOSS': loss}, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    # def on_after_backward(self) -> None:
    #     print(f"==============================")
    #     print(f"Processed by GPU {torch.cuda.current_device()}")
    #     print("on_after_backward enter")
    #     for n, p in self.model.named_parameters():
    #         if p.grad is None:
    #              print(f'{n=}')
    #     print("on_after_backward exit")
    #     print(f"==============================\n\n\n\n\n\n\n\n")

    def on_train_epoch_end(self) -> None:
        # parser = argparse.ArgumentParser()
        # args = parser.parse_args()
        args = {}
        args['device'] = self.device
        args['temperature'] = self.hparams.temperature
        args['top_k'] = self.hparams.top_k
        args['top_p'] = self.hparams.top_p

        if self.hparams.memory_size > 0:
            # def denoise(model_vae, tokenizer_encoder, tokenizer_decoder, text_write_to_mem, text_to_denoise, iterative_step, args, episode_size=1):
            print(">>> denoise")
            rec_list = denoise(self.model,
                               self.tokenizer_encoder,
                               self.tokenizer_decoder,
                               ["All the food is great here."],
                               "All the food is great here.",
                               1,
                               args)

            self.logger.experiment.add_text(f'Reconstruction of \"All the food is great here\"', rec_list[0], global_step=self.global_step)

            args['temperature'] = 0.7
            rec_list = denoise(self.model,
                               self.tokenizer_encoder,
                               self.tokenizer_decoder,
                               ["All the food is great here."],
                               "All the food is great here.",
                               1,
                               args)

            self.logger.experiment.add_text(f'Reconstruction of \"All the food is great here\"', rec_list[0], global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        if self.trainer.train_dataloader is None:
            if self.hparams.bleu:
                if self.hparams.perturb:
                    self.eval_blue_denoise(batch,
                                           self.hparams.ae_only,
                                           self.hparams.num_samples,
                                           self.hparams.ae_read_write,
                                           self.hparams.read_iters,
                                           self.hparams.temperature,
                                           self.hparams.top_k,
                                           self.hparams.top_p)
                else:
                    self.eval_blue_recon(batch,
                                           self.hparams.ae_only,
                                           self.hparams.num_samples,
                                           self.hparams.ae_read_write,
                                           self.hparams.read_iters,
                                           self.hparams.temperature,
                                           self.hparams.top_k,
                                           self.hparams.top_p)
            else:
                self.eval_perp(batch,
                               self.hparams.ae_read_write,
                               self.hparams.num_samples)
        else:
            self.eval_blue_recon(batch,
                                 ae_only=True,
                                 num_samples=1,
                                 ae_read_write=False,
                                 read_iters=1,
                                 temperature=self.hparams.temperature,
                                 top_k=self.hparams.top_k,
                                 top_p=self.hparams.top_p)

            self.eval_blue_recon(batch,
                                 ae_only=False,
                                 num_samples=1,
                                 ae_read_write=True,
                                 read_iters=1,
                                 temperature=self.hparams.temperature,
                                 top_k=self.hparams.top_k,
                                 top_p=self.hparams.top_p)

            self.eval_perp(batch,
                           num_samples=1,
                           ae_read_write=True)

            self.eval_perp(batch,
                           num_samples=1,
                           ae_read_write=False)

    def eval_blue_recon(self,
                        batch,
                        ae_only,
                        num_samples,
                        ae_read_write,
                        read_iters,
                        temperature,
                        top_k,
                        top_p):

        x0, x1, x_lengths = batch

        sample_latents = (not ae_only)
        if not sample_latents:
            nsamples = 1  # no point averaging across multiple samples if running deterministic read write
        else:
            nsamples = num_samples


        num_sents = x1.shape[0]

        max_len_values, _ = x_lengths.max(0)
        x0sub = x0[:, :max_len_values[0]]
        x1sub = x1[:, :max_len_values[1]]

        # feed to model
        bleu_sample_avg = np.zeros(len(x1sub))
        for _ in range(nsamples):
            _, z = self.model.ae(x0sub, x1sub, ae_read_write, sample_latents, read_iters)
            sent_recs = text_from_latent_code(z,
                                              self.model,
                                              temperature,
                                              top_k,
                                              top_p,
                                              self.tokenizer_decoder)

            for i, sent in enumerate(sent_recs):
                sent = sent.replace("\\n", " ")
                words = wordpunct_tokenize(sent)
                sent_recs[i] = words

            for i in range(len(x1sub)):
                x1_lst = x1sub[i].tolist()
                sent_true = self.tokenizer_decoder.decode(x1_lst[1:x1_lst.index(50259)])  # 50259 is EOS token
                # word tokenize
                sent_true = sent_true.replace("\\n", " ")
                sent_true = wordpunct_tokenize(sent_true)
                bleu_sample_avg[i] += nltk.translate.bleu_score.sentence_bleu([sent_true], sent_recs[i])

        bleu_sample_avg /= nsamples
        metric = self.metrics(bleu=np.sum(bleu_sample_avg), num_sents=num_sents)

        if ae_read_write:
            self.log('eval/ae_rw/BLEU_recon', metric['bleu'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        else:
            self.log('eval/BLEU_recon', metric['bleu'], on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def eval_blue_denoise(self,
                          batch,
                          ae_only,
                          num_samples,
                          ae_read_write,
                          read_iters,
                          temperature,
                          top_k,
                          top_p):
        _, x1, _, x0_noisy, x1_noisy, x_lengths_noisy = batch

        sample_latents = (not ae_only)
        if not sample_latents:
            nsamples = 1  # no point averaging across multiple samples if running deterministic read write
        else:
            nsamples = num_samples

        max_len_values, _ = x_lengths_noisy.max(0)
        x0_noisy = x0_noisy[:, :max_len_values[0]]
        x1_noisy = x1_noisy[:, :max_len_values[1]]

        num_sents = x1.shape[0]

        # feed to model
        bleu_sample_avg = np.zeros(len(x1))
        for _ in range(nsamples):
            _, z = self.model.ae(x0_noisy, x1_noisy, ae_read_write, sample_latents, read_iters)
            sent_recs = text_from_latent_code(z,
                                              self.model,
                                              temperature,
                                              top_k,
                                              top_p,
                                              self.tokenizer_decoder)

            for i, sent in enumerate(sent_recs):
                sent = sent.replace("\\n", " ")
                words = wordpunct_tokenize(sent)
                sent_recs[i] = words

            for i in range(len(x1)):
                x1_lst = x1[i].tolist()
                sent_true = self.tokenizer_decoder.decode(x1_lst[1:x1_lst.index(50259)])
                sent_true = sent_true.replace("\\n", " ")
                sent_true = wordpunct_tokenize(sent_true)
                bleu_sample_avg[i] += nltk.translate.bleu_score.sentence_bleu([sent_true], sent_recs[i])
        bleu_sample_avg /= nsamples

        metric = self.metrics(bleu=np.sum(bleu_sample_avg), num_sents=num_sents)

        if ae_read_write:
            self.log('eval/ae_rw/BLEU_denoise', metric['bleu'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        else:
            self.log('eval/BLEU_denoise', metric['bleu'], on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def eval_perp(self, batch, ae_read_write, num_samples):

        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0sub = x0[:, :max_len_values[0]]
        x1sub = x1[:, :max_len_values[1]]

        rl, _ = self.model.ae(x0sub, x1sub, ae_read_write)

        # if self.hparams.ae_only:
        #     metrics = self.metrics(rec_ae=rl.sum().item(),
        #                            num_sents=x0.shape[0],
        #                            num_words=x_lengths[:, 1].sum().item())
        #
        # else:
        ns = min(num_samples, 10)
        loss, loss_rc, loss_kl, loss_rc_z = self.model.loss_iw(x0sub, x1sub, nsamples=num_samples, ns=ns)

        metrics = self.metrics(rec=loss_rc.sum(),
                                   rec_z=loss_rc_z.sum(),
                                   rec_ae=rl.sum().item(),
                                   kl=loss_kl.sum(),
                                   loss=loss.sum(),
                                   num_sents=x0.shape[0],
                                   num_words=x_lengths[:, 1].sum().item())

        if ae_read_write:
            self.log('eval/ae_rw/ELBO', metrics['elbo'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/ae_rw/NLL', metrics['nll'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/ae_rw/NLL_Z', metrics['nll_z'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/ae_rw/KL', metrics['kl'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/ae_rw/PPL', metrics['ppl'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/ae_rw/PPL_ae', metrics['ppl_ae'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/ae_rw/NLL_ae', metrics['nll_ae'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        else:
            self.log('eval/ELBO', metrics['elbo'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/NLL', metrics['nll'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/NLL_Z', metrics['nll_z'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/KL', metrics['kl'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/PPL', metrics['ppl'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/PPL_ae', metrics['ppl_ae'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log('eval/NLL_ae', metrics['nll_ae'], on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.optimizer == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        elif self.hparams.optimizer == 'fusedadam':
            optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        elif self.hparams.optimizer == 'deepspeed':
            optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        else:
            raise ValueError('unknown optimizer')

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
