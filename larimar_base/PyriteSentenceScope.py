import numpy as np
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer
import spacy
from transformers import AutoModel
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
import read_write_interactive as rwi
from argparse import Namespace
import torch
import demo_utils as du
from transformers import BertTokenizer, BertModel

def _cosine_distance(x, y):
    # we are assuming all vecs are L2 norm = 1
    return 1 - (x @ y.T) / (norm(x, axis=1) * norm(y, axis=1))


class PyriteSentenceEmbedder:
    """
    A wrapper to use pyrite encoder as sentence embedder. Make it API-behave like other HF models
    """
    args = Namespace(latent_size=768,
                     max_seq_length=256,
                     nz=768,
                     memory_size=256,
                     episode_sizes=[8],
                     direct_writing=True,
                     ordering=False,
                     pseudoinverse_approx_step=7,
                     identity=False,
                     w_logvar_setting=0,
                     deterministic_w=False,
                     observation_noise_std=0,
                     block_size=256,
                     temperature=1.0,
                     top_k=0,
                     top_p=1,
                     )
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, checkpoint: str,
                 verbose: bool = True, device=args.device):
        self.verbose = verbose
        self.checkpoint = checkpoint
#        self.pyrite_model, self.tokenizer_encoder, self.tokenizer_decoder = rwi.load_model(checkpoint, memsize, psteps)
        self.pyrite_model, self.tokenizer_encoder, self.tokenizer_decoder = du.load_model(checkpoint, device=device)
        self.pyrite_model.to(device)
        if verbose: print(f"INFO: Using {device} for model/tokenizer")

    def encode(self, sent: str, convert_to_tensor: bool = True) -> np.ndarray:
        e, _ = rwi.latent_code_from_text(sent, self.tokenizer_encoder, self.pyrite_model, args=self.args)
        e = e.detach().cpu().numpy() if not convert_to_tensor else e
        return e


class OriginalBertSentenceEmbedder:
    """
    A wrapper to use Bert as sentence embedder. Make it API-behave like other HF models
    """
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 verbose: bool = True):
        self.verbose = verbose
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.model = BertModel.from_pretrained("bert-large-cased")

    def encode(self, sent: str, output_key='pooler_output') -> np.ndarray:
        encoded_input = self.tokenizer(sent, return_tensors='pt', padding=True)
        output = self.model(**encoded_input)
        return output[output_key].detach().cpu().numpy()

class ScopeMemory:

    def __init__(self, modelname: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 dist_func: callable = _cosine_distance, normalize: bool = False,
                 split_sentences: bool = True, split_sent_policy: str = "take_best",
                 verbose: bool = True, device='cuda', mode: str="embedder",
                 dist_threshold = None, **kwargs):
        """

        Args:
            modelname:
            dist_func:
            normalize:
            split_sentences:
            split_sent_policy:
            verbose:
            device:
            mode: "embedder"  = simple sentence embedding and cosine sim; "pyrite" = experimental setting
            **kwargs:
        """
        self.verbose = verbose
        self.device = device
        self.mode = mode
        self.encoder_args = {}
        self.dist_threshold = dist_threshold
        if self.dist_threshold is not None and self.verbose:
            print(f"INFO (ScopeMemory): detection threshold set to {self.dist_threshold}")
        if 'MiniLM' in modelname:
            # 'sentence-transformers/all-MiniLM-L6-v2'
            self.model = SentenceTransformer(modelname)
            self.encoder_args = {'convert_to_tensor': False, 'show_progress_bar': False}
        elif 'jina' in modelname:
            # from https://huggingface.co/jinaai/jina-embeddings-v2-small-en
            # 'jinaai/jina-embeddings-v2-small-en'
            self.model = AutoModel.from_pretrained(modelname,
                                                   trust_remote_code=True)  # trust_remote_code is needed to use the encode method
            self.encoder_args = {'convert_to_tensor': False, 'show_progress_bar': False}
        elif 'optimus' in modelname or 'pyrite' in modelname:
            self.model = PyriteSentenceEmbedder(modelname, **kwargs)
            self.encoder_args = {'convert_to_tensor': False}
        elif 'all-mpnet-base' in modelname:
            self.model = SentenceTransformer(modelname)
            self.encoder_args = {'convert_to_tensor': False, 'show_progress_bar': False}
        elif 'bert-large-cased' in modelname:
            self.model = OriginalBertSentenceEmbedder()
            self.encoder_args = {}
        else:
            raise ValueError("ERROR: Unknown model type.")
        if verbose: print(f"INFO (ScopeMemory): Using sentence embedder: {modelname} in mode: {self.mode}")
        if self.mode != "embedder" and verbose:
            print(f"INFO: non-default mode {self.mode} active")

        self.sentence_splitter = None
        self.split_sent_policy = split_sent_policy
        if split_sentences:
            if verbose: print(f"INFO (ScopeMemory): Test sentence splitting active (policy: {self.split_sent_policy})")
            self.sentence_splitter = spacy.load('en_core_web_sm')
        self.sentences: List = []
        self.embeddings = None
        self.pyrite_embeddings = None   # used in mode==pyrite
        self.pyrite_posterior_memory = None  # ditto
        self.pyrite_deltas = None    # ditto
        self.normer = None
        self.normalize = normalize
        if self.verbose and self.normalize:
            print(f"INFO (ScopeMemory): Normalization active.")
        self.dist_func = dist_func
        self.pyrite_stdscaler = None

    def _calc_normer(self):
        assert self.embeddings is not None, "not initialized"
        d = self.dist_func(self.embeddings, self.embeddings)
        d = d[np.triu_indices(d.shape[0])]
        self.normer = StandardScaler().fit(d.reshape(-1, 1))

    def _add_sentences_pyrite_mode(self, text_write_to_mem: List[str]):
        memory_write_text_encoded, cls = du.latent_code_from_text(text_write_to_mem,
                                                                  self.model.tokenizer_encoder,
                                                                  self.model.pyrite_model,
                                                                  device=self.device,
                                                                  code_size=self.model.pyrite_model._code_size)
        self.pyrite_embeddings = memory_write_text_encoded.detach().cpu().numpy()
        if cls is None:
            print("Found 0 len in text_to_write")
            memory_write_text_encoded = memory_write_text_encoded.reshape(1, 1, self.model.pyrite_model._code_size)
        else:
            memory_write_text_encoded = memory_write_text_encoded.reshape(len(text_write_to_mem), 1,
                                                                          self.model.pyrite_model._code_size)
        self.pyrite_posterior_memory, dkl_M = self.model.pyrite_model.write(input_encoded=memory_write_text_encoded)

        self.model.pyrite_model.memory.deterministic = True

        # store differences
        for zi in range(len(text_write_to_mem)):
            zq = memory_write_text_encoded[zi].unsqueeze(0)
            zs, _ = self.model.pyrite_model.read(zq, self.pyrite_posterior_memory)
            zs = zs.reshape(1, self.model.pyrite_model._code_size).detach().cpu().numpy()
            zq = zq.reshape(1, self.model.pyrite_model._code_size).detach().cpu().numpy()
            delta = zs - zq
            if self.pyrite_deltas is None:
                self.pyrite_deltas = delta
            else:
                self.pyrite_deltas = np.concatenate([self.pyrite_deltas, delta])

        # use these as standardization population
        self.pyrite_stdscaler = StandardScaler().fit(norm(self.pyrite_deltas, axis=-1).reshape(-1,1))


    def add_sentences(self, sentences: List):
        assert isinstance(sentences, list)
        self.sentences += sentences
        if self.mode == 'embedder':
            auxemb = self.model.encode(sentences, **self.encoder_args)
        elif self.mode == 'pyrite':
            return self._add_sentences_pyrite_mode(sentences)  # this is a branch out - pyrite stuff handled separately
        auxemb = np.expand_dims(auxemb, axis=0) if len(auxemb.shape) < 2 else auxemb
        if self.embeddings is None:
            self.embeddings = auxemb
        else:
            self.embeddings = np.concatenate([self.embeddings, auxemb])
        if self.normalize:
            self._calc_normer()
        return self

    def reset(self):
        self.embeddings = None
        self.pyrite_embeddings = None
        self.pyrite_deltas = None
        self.sentences = []
        self.normer = None

    def _split_sentences(self, s: str) -> List[str]:
        return [s] if self.sentence_splitter is None else self.sentence_splitter(s).sents

    def _knn_distance(self, x: np.ndarray, k: int = 1):
        assert self.embeddings is not None, "not initialized"
        assert isinstance(x, np.ndarray)
        x = np.expand_dims(x, axis=0) if len(x.shape) < 2 else x
        d = self.dist_func(x, self.embeddings)
        if self.normalize:
            od = d.shape
            d = self.normer.transform(d.reshape([-1, 1])).reshape(od)
        return np.mean(np.sort(d, axis=1)[:, :k], axis=1)

    def knn_distance(self, sent: Union[list, str], k: int = 1) -> List[float]:
        sent = [sent] if not isinstance(sent, list) else sent
        if self.mode == "pyrite":
            return self._knn_distance_pyrite(sent, k=k)
        assert self.embeddings is not None
        outd = []
        for s in sent:
            auxs = []
            if self.sentence_splitter is not None:
                auxs = [s for s in self._split_sentences(s)]
            auxd = []
            for ss in auxs:
                e = self.model.encode(str(ss), **self.encoder_args)
                auxd.append(self._knn_distance(e, k=k))
            sent_score = np.min(auxd) if self.split_sent_policy == "take_best" else np.mean(auxd)
            outd.append(sent_score)
        return outd

    def _knn_distance_pyrite(self, sent: Union[list, str], k: int = 1) -> List[np.ndarray]:
        sent = [sent] if not isinstance(sent, list) else sent
        assert self.pyrite_deltas is not None
        outd = []
        for s in sent:
            auxs = []
            if self.sentence_splitter is not None:
                auxs = [str(s) for s in self._split_sentences(s)]
            auxd = []
            for ss in auxs:
                # Then read from the memory
                encoded_read, cls = du.latent_code_from_text(ss,
                                                                          self.model.tokenizer_encoder,
                                                                          self.model.pyrite_model,
                                                                          device=self.device,
                                                                          code_size=self.model.pyrite_model._code_size)
                encoded_read = encoded_read.reshape(1, 1, self.model.pyrite_model._code_size)
                self.model.pyrite_model.memory.deterministic = True
                zs, _ = self.model.pyrite_model.read(encoded_read, self.pyrite_posterior_memory)
                zs = zs.reshape(1, self.model.pyrite_model._code_size).detach().cpu().numpy()
                encoded_read = encoded_read.reshape(1, self.model.pyrite_model._code_size).detach().cpu().numpy()
                delta = zs-encoded_read
                d = norm(delta)
                #transform(norm(self.pyrite_deltas, axis=-1).reshape(-1,1))
                if self.normalize:
                    od = d.shape
                    d = self.pyrite_stdscaler.transform(d.reshape([-1, 1])).reshape(od)
                auxd.append(d)
            sent_score = np.min(auxd) if self.split_sent_policy == "take_best" else np.mean(auxd)
            outd.append(sent_score)
        return outd

    def is_indomain(self, sent: str, dist_threshold: float=None, k: int = 1) -> (bool, float):
        """
        Returns True if 'sent' is similar to memory, based on 'dist_thr' given (if thr==None, threshold
        specified at instantiation time will be used). Also returns distance.
        Args:
            sent:
            thr:
            k:

        Returns:
            (Bool, float)
        """
        assert dist_threshold is not None or self.dist_threshold is not None, "Threshold has not been set."
        t = dist_threshold if dist_threshold is not None else self.dist_threshold
        d = self.knn_distance(sent, k=k)
        assert len(d)==1, "weird"
        d = d[0]
        det = True if d<=t else False
        return det, d



if __name__ == "__main__":
    sentences = ["I'm happy", "I am being joyful", "I am pretty content",
                 "happy am I", "Howard glacier is in India."]
    test_sentences = ['A Maran Week is annually held since 1990. Howard glacier is located in India.',
                      "I am filled with happiness", "krakken be not"]

    args = {'device': 'cpu'}
    # args = {'device': 'cuda'}

    model = 'sentence-transformers/all-MiniLM-L6-v2'
    # model = 'jinaai/jina-embeddings-v2-small-en'
    # model = 'jinaai/jina-embeddings-v2-base-en'
    # model = 'bert-large-cased'
    mode = "embedder"
    # mode = "pyrite"
    # sm = ScopeMemory(modelname=model, mode=mode, normalize=True, **args)
    # sm.add_sentences(sentences)
    # # sm.add_sentences([sentences[0]])
    # k = 1
    # for s in test_sentences:
    #     print(f"RESULT: sentence = \'{s}\', dnn(k={k}) = {sm.knn_distance(s, k=k)}")
    # sm.reset()
    # sm.add_sentences(sentences)
    # sm.add_sentences([sentences[0]])
    # k = 2
    # print(f"RESULT: dnn(k={k}) = {sm.knn_distance(test_sentences, k=2)}")

    sm2 = ScopeMemory(modelname=model, mode=mode, normalize=False, dist_threshold=0.5)
    sm2.add_sentences(sentences)
    det, d = sm2.is_indomain(sentences[0])
    print("debug")
