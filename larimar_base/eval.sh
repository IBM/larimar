mode=pyrite
dataset=counterfact
cache_dir=../cache
checkpoint=../models/larimar-1.3b-c3.ckpt
data_dir=../data/counterfact
res_dir_name=../eval/results
num_eval_cases=2000
scope_detect_threshold=0.3

# scope detection
python counterfact_eval.py \
       --mode ${mode} \
       --dataset ${dataset} \
       --cache_dir  ${cache_dir} \
       --checkpoint ${checkpoint} \
       --data_dir   ${data_dir} \
       --res_dir_name ${res_dir_name} \
       --num_eval_cases ${num_eval_cases} \
       --scope_detect_threshold ${scope_detect_threshold}
       
# no scope
python counterfact_eval.py \
       --mode ${mode} \
       --dataset ${dataset} \
       --cache_dir  ${cache_dir} \
       --checkpoint ${checkpoint} \
       --data_dir   ${data_dir} \
       --res_dir_name ${res_dir_name} \
       --num_eval_cases ${num_eval_cases} \




