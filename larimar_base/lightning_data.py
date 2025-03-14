import lightning as pl
from utils import BucketingDataLoaderPL
from lightning_model import prepare_enc_dec_tokenizer


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 train_data_file,
                 train_batch_size,
                 eval_data_file,
                 eval_batch_size,
                 max_seq_length,
                 perturb,
                 use_labels,
                 dataset,
                 use_philly,
                 num_data_workers,
                 batches_per_bucket,
                 block_size,
                 encoder_model_type,
                 encoder_model_name_or_path,
                 decoder_model_type,
                 decoder_model_name_or_path,
                 cache_dir,
                 do_lower_case,
                 num_chunks):

        super().__init__()

        self.train_data_file = train_data_file
        self.train_batch_size = train_batch_size
        self.max_seq_length = max_seq_length
        self.eval_data_file = eval_data_file
        self.eval_batch_size = eval_batch_size
        self.perturb = perturb
        self.use_labels = use_labels
        self.dataset = dataset
        self.use_philly = use_philly
        self.block_size = block_size
        self.num_data_workers = num_data_workers
        self.batches_per_bucket = batches_per_bucket
        self.num_chunks = num_chunks

        tokenizer_encoder, tokenizer_decoder = prepare_enc_dec_tokenizer(encoder_model_type,
                                                                         encoder_model_name_or_path,
                                                                         decoder_model_type,
                                                                         decoder_model_name_or_path,
                                                                         cache_dir,
                                                                         do_lower_case,
                                                                         block_size)

        self.tokenizer = [tokenizer_encoder, tokenizer_decoder]

    def setup(self, stage=None):

        if stage == 'fit':
            self.traindl = BucketingDataLoaderPL(self.train_data_file,
                                               self.train_batch_size,
                                               self.max_seq_length,
                                               self.tokenizer,
                                               self.block_size,
                                               self.use_labels,
                                               self.dataset,
                                               self.use_philly,
                                               self.num_chunks,
                                               self.num_data_workers,
                                               batches_per_bucket=100,
                                               perturb=self.perturb,
                                               shuffle=True)
        if stage in ("fit", "validate"):
            self.valdl = BucketingDataLoaderPL(self.eval_data_file,
                                             self.eval_batch_size,
                                             self.max_seq_length,
                                             self.tokenizer,
                                             self.block_size,
                                             self.use_labels,
                                             self.dataset,
                                             self.use_philly,
                                             self.num_chunks,
                                             self.num_data_workers,
                                             batches_per_bucket=100,
                                             perturb=self.perturb,
                                             shuffle=False)
        else:
            return

    def train_dataloader(self):
        return self.traindl.get()

    def val_dataloader(self):
        return self.valdl.get()
