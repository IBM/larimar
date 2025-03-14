import torch
import torch.nn as nn
import numpy as np
from .memory import GPM
from .utils import log_sum_exp
import math
import torch.nn.functional as F
import pickle

import logging
logger = logging.getLogger(__name__)

class MemVAE(nn.Module):
    """VAE with memory prior"""
    def __init__(self,
                 encoder,
                 decoder,
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
                 decode_rec_strength): #

        super(MemVAE, self).__init__()

        self._memory_size = memory_size # K
        self._code_size = latent_size # C
        self._direct_writing = direct_writing
        self._ordering = ordering
        self._pseudoinverse_approx_step = pseudoinverse_approx_step
        self._observation_noise_std = observation_noise_std
        self.episode_sizes = episode_sizes # episode size used during training
        self.dim_target_kl = dim_target_kl
        self.length_weighted_loss = length_weighted_loss
        self.rec_strength = rec_strength
        self.ae_strength = ae_strength
        self.l2_strength = l2_strength
        self.beta = beta

        self.encoder = encoder
        self.decoder = decoder
        self.memory = GPM(code_size=self._code_size,
                          memory_size=self._memory_size,
                          direct_writing=self._direct_writing,
                          ordering=self._ordering,
                          pseudoinverse_approx_step=self._pseudoinverse_approx_step,
                          observation_noise_std=self._observation_noise_std,
                          identity=identity,
                          w_logvar_setting=w_logvar_setting,
                          deterministic=deterministic_w)


        #self.nz = args.latent_size
        self.eos_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.eos_token])[0]
        self.pad_token_id = tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]

        self.decode_rec_strength=decode_rec_strength

        # connector: from Bert hidden units to the latent space
        # self.linear = nn.Linear(args.nz, 2 * args.nz, bias=False)
    
    def write(self, input_encoded):
        posterior_memory, dkl_M = self.memory.write_to_memory(input_encoded)
        return posterior_memory, dkl_M

    def read(self, input_encoded, posterior_memory, reduce_kl=True, get_w=False, deterministic=False, sigma_w=None):
        return self.memory.read_with_encoded_input(input_encoded=input_encoded,
                                                   memory_state=posterior_memory,
                                                   reduce_kl=reduce_kl, get_w=get_w, 
                                                   deterministic=deterministic,
                                                   sigma_w=sigma_w)

    def _bijective_loss(self, labels, latent_z):
        outputs = self.decoder(input_ids=labels, past_key_values=latent_z, labels=labels)
        #bijective_loss = -torch.mean(torch.sum(input * torch.log(input_recon + 1e-12), dim=1) +
        #                             torch.sum((1 - input) * torch.log(1 - input_recon + 1e-12), dim=1))
        return outputs[0]

    def mem_connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        mu, _ = self.encoder.linear(bert_fea).chunk(2, -1)
        batch_size, nz = mu.size()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        return mu_expd

    def ae(self, x0, x1, read_write=False, sample=False, read_iters=1):
        bert_fea = self.encoder(x0)[1]
        z, _ = self.encoder.linear(bert_fea).chunk(2, -1)
        if read_write:
            z = z.reshape(1, x0.size(0), self._code_size)
            if self._ordering:
                z = self.memory._z_attention(z)
            posterior_mem, _ = self.write(z)
            for i in range(read_iters):
                if sample:
                    z, _ = self.read(z, posterior_mem)
                else:
                    z, _ = self.read(z, posterior_mem, deterministic=True)
            z = z.squeeze()
        outputs = self.decoder(input_ids=x1, past_key_values=z, labels=x1)
        return outputs[0], z

    def forward(self, inputs, labels, episode_size=1, train=False, beta_t=None):

        if train:
            # randomly sample an episode size from training episode sizes
            episode_size = min(inputs.shape[0], np.random.choice(self.episode_sizes))

        batch_size = inputs.shape[0] // episode_size

        attention_mask=(inputs > 0).float()
        # logger.info(inputs)
        # logger.info(attention_mask)
        # logger.info(labels)
        reconstrution_mask=(labels != 50257).float() # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)

        outputs = self.encoder(inputs, attention_mask)
        pooled_hidden_fea = outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)

        latent_z = self.mem_connect(pooled_hidden_fea)
        latent_z = latent_z.squeeze(1)
        bijective_loss = self._bijective_loss(labels, latent_z)

        pre_reshape_z = latent_z

        latent_z = latent_z.reshape(episode_size, batch_size, self._code_size)

        if self._ordering:
            latent_z = self.memory._z_attention(latent_z)

        posterior_memory, dkl_M = self.write(latent_z)
        z, loss_kl = self.read(latent_z, posterior_memory, reduce_kl=False)
        kl_mask = (loss_kl > self.dim_target_kl).float()
        loss_kl = (kl_mask * loss_kl)
        loss_kl = torch.sum(loss_kl, -1).squeeze().reshape(episode_size * batch_size)
        z = z.reshape(episode_size * batch_size, self._code_size)
        l2_loss = torch.mean((z - pre_reshape_z)**2, dim=-1)
        # past = self.decoder.linear(latent_z)

        # pickle.dump({'input_ids': labels.detach().cpu().numpy(),
        #              'past':z.reshape(episode_size * batch_size, self._code_size).detach().cpu().numpy(),
        #              'labels': labels.detach().cpu().numpy(),
        #              'label_ignore': self.pad_token_id},
        #             open('tmp.pkl', 'wb'))

        # Decoding
        past_z=z.reshape(episode_size * batch_size, self._code_size)
        outputs = self.decoder(input_ids=labels, past_key_values=past_z, labels=labels)
        loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        # Decoding without past
        if self.decode_rec_strength > 0:
            outputs_decoded = self.decoder(input_ids=labels, 
                                           past_key_values=torch.zeros_like(past_z), # torch.empty_like(past_z), 
                                           labels=labels)
            loss_rec_decoded = outputs_decoded[0]  # model outputs are always tuple in pytorch-transformers (see doc)
        else:
            loss_rec_decoded = torch.tensor([0.0]*batch_size)

        if beta_t is None:
            beta_t = self.beta

        if self.length_weighted_loss:
            loss_rec /= sent_length
            bijective_loss /= sent_length
            loss_rec_decoded /= sent_length
            loss = self.rec_strength * loss_rec + self.ae_strength * bijective_loss + \
                   self.l2_strength * l2_loss + beta_t * loss_kl
        else:
            loss = self.rec_strength * loss_rec + self.ae_strength * bijective_loss + \
                   self.l2_strength * l2_loss + beta_t * loss_kl
            
        if self.decode_rec_strength > 0:
            loss += self.decode_rec_strength * loss_rec_decoded

        # if loss is None or torch.isnan(loss).any():
        #     print(">>> Error ")
        #     print(outputs_decoded[0])
        #     print(sent_length)
        #     print(loss_rec, bijective_loss, loss_kl, l2_loss, loss_rec_decoded, loss)
        #     exit()
            
        return loss_rec, bijective_loss, loss_kl, l2_loss, loss_rec_decoded, loss

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        outputs = self.decoder(input_ids=x, past_key_values=z, labels=x)
        loss_rec = outputs[0]
        return -loss_rec
    
    def eval_prior_dist(self, w_sample):
        device = w_sample.device
        loc = torch.zeros(self._memory_size).to(device)
        scale = torch.ones(self._memory_size).to(device)
        prior = torch.distributions.normal.Normal(loc, scale)
        return prior.log_prob(w_sample).sum(dim=-1)

    def eval_cond_ll(self, x, z):
        """compute log p(x|w) = log p(x|z,w) + log p(z|w) = log p(x|z) + log p(z|w)
        """
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = x.unsqueeze(1).repeat(1, z_shape[1], 1).contiguous().view(x_shape[0]*z_shape[1], x_shape[-1]) 
            z = z.contiguous().view(x_shape[0]*z_shape[1], z_shape[-1]) 

        return self.log_probability(x, z)
    
    def eval_inference_dist(self, w, param):
        """this function computes log q(w | x)
        Args:
            w: tensor
                different w points that will be evaluated, with
                shape [batch, nsamples, K]
        Returns: Tensor1
            Tensor1: log q(w|x) with shape [batch, nsamples]
        """
        K = w.size(2)
        mu, logvar = param
        #logvar = logvar.squeeze()

        var = logvar.exp()

        # (batch_size, nsamples, K)
        dev = w - mu
        #print(logvar.shape)

        # (batch_size, nsamples)
        if len(logvar) == 1:
            # same logvar for each dimension
            log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                0.5 * (K * math.log(2 * math.pi) + K * logvar)
        else:
            log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                0.5 * (K * math.log(2 * math.pi) + logvar.sum(dim=-1))
        #print('output shape', log_density.shape)
        return log_density
    
    def loss_iw(self, x0, x1, nsamples=50, ns=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """
        # encoding into bert features
        bert_fea = self.encoder(x0)[1]
        batch_size = x0.size(0)

        # (batch_size, nz)

        start_z, _ = self.encoder.linear(bert_fea).chunk(2, -1)

        log_gen_z_only = self.eval_cond_ll(x1, start_z)
        
        latent_z = start_z.reshape(1, batch_size, self._code_size)

        if self._ordering:
            latent_z = self.memory._z_attention(latent_z)
        
        # expand to have ns samples
        latent_z = latent_z.repeat(1, ns, 1)

        # write to memory
        posterior_memory, _ = self.write(latent_z)

        # get KL
        _, KL = self.read(latent_z, posterior_memory, reduce_kl=True)

        # mu, logvar = mu.squeeze(0), logvar.squeeze(0)
        ll_tmp, rc_tmp = [], []
        for _ in range(int(nsamples / ns)):
            # (batch, nsamples, nz)
            z, w_sample, w_mean, w_logvar = self.read(latent_z, posterior_memory, get_w=True)
                     
            # [batch, nsamples]
            w_sample = w_sample.squeeze().reshape(ns, batch_size, self._memory_size).transpose(0,1)
            w_mean = w_mean.squeeze().reshape(ns, batch_size, self._memory_size).transpose(0,1)
            #print(w_mean[0])
            z = z.reshape(ns, batch_size, self._code_size).transpose(0,1)
            #w_sample = w_sample.squeeze().reshape(batch_size, ns, self._memory_size)
            #w_mean = w_mean.squeeze().reshape(batch_size, ns, self._memory_size)
            #z = z.reshape(batch_size, ns, self._code_size)
            if len(w_logvar.size()) > 1:
                w_logvar = w_logvar.squeeze().reshape(ns, batch_size, self._memory_size).transpose(0,1)
            
            log_prior = self.eval_prior_dist(w_sample)
            log_gen = self.eval_cond_ll(x1, z)
            log_infer = self.eval_inference_dist(w_sample, (w_mean, w_logvar))
            # pdb.set_trace()
            log_gen = log_gen.unsqueeze(0).contiguous().view(z.shape[0],-1)


            # pdb.set_trace()
            rc_tmp.append(log_gen)
            ll_tmp.append(log_gen + log_prior - log_infer)
            
        log_prob_iw = log_sum_exp(torch.cat(ll_tmp, dim=-1), dim=-1) - math.log(nsamples)
        log_gen_iw = torch.mean(torch.cat(rc_tmp, dim=-1), dim=-1)
        return log_prob_iw, log_gen_iw , KL / ns, log_gen_z_only
