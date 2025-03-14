
####################################################################################################
# config yaml file
####################################################################################################
config_file=configs/config_train_larimar.yaml


####################################################################################################
# model_* 
####################################################################################################
model_encoder_model_name_or_path="bert-large-cased"
model_decoder_model_name_or_path="gpt2-large"
model_decode_rec_strength=1.0
model_optimizer=adamw
model_learning_rate=5e-5
model_observation_noise_std=0.000001
model_beta=0.5

episode_length=16
model_episode_sizes=[${episode_length}]



####################################################################################################
# trainer_*
####################################################################################################
trainer_devices=8
trainer_max_epochs=5
trainer_precision=32-true
trainer_strategy=ddp
trainer_callbacks_init_args_every_n_train_steps=20000
trainer_callbacks_init_args_save_top_k=3



####################################################################################################
# data_*
####################################################################################################
data_train_batch_size=16
data_num_chunks=false



####################################################################################################
# directories and files
####################################################################################################

# cache directory
model_cache_dir=../cache

# trained model directory  
decoder_name_save=$(echo "gpt2" | sed 's/\//-/g')
loss_type=decoder_loss
top_larimar_model_dir=../train/larimar/checkpoints
larimar_model_description=${model_encoder_model_name_or_path}-${decoder_name_save}-large-wiki-ep-${episode_length}_${loss_type}_${model_observation_noise_std}
trainer_default_root_dir=${top_larimar_model_dir}/${larimar_model_description}
trainer_logger_init_args_save_dir=${trainer_default_root_dir}


# training data directory and files
block_size=64
top_training_data_dir=../data
training_data_dir=${top_training_data_dir}/wikipedia/blocksize_${block_size}
data_train_data_file=${training_data_dir}/train.txt
data_eval_data_file=${training_data_dir}/test.txt





####################################################################################################
# train
####################################################################################################
python main_pl.py fit \
       --config ${config_file} \
       --model.cache_dir=${model_cache_dir} \
       --model.encoder_model_name_or_path ${model_encoder_model_name_or_path} \
       --model.decoder_model_name_or_path ${model_decoder_model_name_or_path} \
       --model.optimizer ${model_optimizer} \
       --model.learning_rate ${model_learning_rate} \
       --model.episode_sizes ${model_episode_sizes} \
       --model.decode_rec_strength ${model_decode_rec_strength} \
       --model.observation_noise_std ${model_observation_noise_std} \
       --model.beta ${model_beta} \
       --trainer.devices ${trainer_devices} \
       --trainer.max_epochs ${trainer_max_epochs} \
       --trainer.precision ${trainer_precision} \
       --trainer.strategy ${trainer_strategy} \
       --trainer.default_root_dir  ${trainer_default_root_dir} \
       --trainer.logger.init_args.save_dir ${trainer_logger_init_args_save_dir} \
       --trainer.callbacks.init_args.every_n_train_steps ${trainer_callbacks_init_args_every_n_train_steps}  \
       --trainer.callbacks.init_args.save_top_k ${trainer_callbacks_init_args_save_top_k} \
       --data.train_batch_size ${data_train_batch_size} \
       --data.train_data_file ${data_train_data_file} \
       --data.eval_data_file ${data_eval_data_file} \
       --data.num_chunks  ${data_num_chunks}
