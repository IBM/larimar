mode=pyrite
dataset=counterfact
cache_dir=../cache
checkpoint=../models/larimar-1.3b-c3.ckpt
data_dir=../data/counterfact
res_dir_name=../eval/results
num_eval_cases=2000
scope_detect_threshold=0.3

# scope detection
for num_rephrases in 0 1 2
do
    python counterfact_eval_rephrase.py \
	   --mode ${mode} \
	   --dataset ${dataset} \
	   --cache_dir  ${cache_dir} \
	   --checkpoint ${checkpoint} \
	   --data_dir   ${data_dir} \
	   --res_dir_name ${res_dir_name} \
	   --num_eval_cases ${num_eval_cases} \
	   --num_rephrases ${num_rephrases} \
	   --remove_distraction \
	   --scope_detect_threshold ${scope_detect_threshold}
done



# no scope
for num_rephrases in 0 1 2
do
python counterfact_eval_rephrase.py \
       --mode ${mode} \
       --dataset ${dataset} \
       --cache_dir  ${cache_dir} \
       --checkpoint ${checkpoint} \
       --data_dir   ${data_dir} \
       --res_dir_name ${res_dir_name} \
       --num_eval_cases ${num_eval_cases} \
       --num_rephrases ${num_rephrases} \
       --remove_distraction
done



