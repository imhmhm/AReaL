<<<<<<< HEAD:wisemlops/demo_if_local_sglang.sh
host_ip=$(hostname -I | awk '{print $1}')
export no_proxy="${host_ip}${no_proxy:+,$no_proxy}"
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

MTP_CODE_DIR=/home/ma-user/work/dataset/dataset_zhh_guiyang/github/AReaL
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH

export MTP_DATASET_HOME=/home/ma-user/work/dataset/dataset_zhh_guiyang
AREAL_EXPERIMENT_NAME=if-grpo-qwen25_1_5b_910b
TRIAL_NAME=$(date +%y%m%d%H%M%S)

export NLTK_DATA=$MTP_DATASET_HOME/nltk_data

python examples/instruction_following/instruction_following_rl.py \
    --config examples/instruction_following/if_grpo_npu.yaml \
    allocation_mode=sglang:d4p1t1+d4p1t1 \
    scheduler.type=local \
    actor.optimizer.lr=1.70e-5 \
    rollout.max_head_offpolicyness=2 \
    cluster.fileroot=$MTP_DATASET_HOME/ckpt/areal \
    cluster.name_resolve.nfs_record_root=$MTP_DATASET_HOME/outputs/areal/name_resolve \
    actor.path=$MTP_DATASET_HOME/hf_model/Qwen2.5-1.5B-Instruct \
    actor.scheduling_spec.0.env_vars.TASK_QUEUE_ENABLE=1 \
    actor.kl_ctl=0.1 \
    train_dataset.path=$MTP_DATASET_HOME/hf_data/IF_multi_constraints_upto5 \
    +train_dataset.format=instruction_following \
    +train_dataset.split=train \
    valid_dataset.path=$MTP_DATASET_HOME/hf_data/IFEval \
    +valid_dataset.format=instruction_following \
    +valid_dataset.split=train \
    +stats_logger.swanlab.mode=local \
    experiment_name=$AREAL_EXPERIMENT_NAME \
=======
ray stop --force
rm -rf /tmp/ray
ulimit -n 32768
host_ip=$(hostname -I | awk '{print $1}')
export no_proxy="${host_ip}${no_proxy:+,$no_proxy}"


## ====== MTP dir ====== ##
MTP_DATASET_HOME=/home/ma-user/work/dataset/huashan_zhh_wulan_pfs
MTP_CODE_DIR=/home/ma-user/work/dataset/dataset_zhh_wulan/github/AReaL
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH


## ====== Ascend ====== ##
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh


## ====== cluster ====== ##
export NNODES=${MA_NUM_HOSTS:-1}
export NPUS_PER_NODE=${MA_NUM_GPUS:-$(npu-smi info -l | grep -c "NPU ID")}


## ====== hccl ====== ##
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050


## ====== areal ====== ##
HF_MODEL_DIR=$MTP_DATASET_HOME/hf_model/Qwen2.5-1.5B-Instruct
AREAL_ALLOCATION_MODE=${AREAL_ALLOCATION_MODE:-'sglang:d2p1t1+d2p1t1'}
AREAL_EXPERIMENT_NAME=gsm8k-grpo-qwen25_1_5b-910b
TRIAL_NAME=$(date +%y%m%d%H%M%S)


python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_npu.yaml \
    allocation_mode=$AREAL_ALLOCATION_MODE \
    scheduler.type=local \
    actor.optimizer.lr=1.70e-5 \
    rollout.max_head_offpolicyness=2 \
    cluster.fileroot=$MTP_DATASET_HOME/ckpt/areal \
    cluster.name_resolve.nfs_record_root=$MTP_DATASET_HOME/outputs/areal/name_resolve \
    actor.path=$HF_MODEL_DIR \
    actor.scheduling_spec.0.env_vars.TASK_QUEUE_ENABLE=1 \
    train_dataset.path=$MTP_DATASET_HOME/hf_data/gsm8k \
    valid_dataset.path=$MTP_DATASET_HOME/hf_data/gsm8k \
    +stats_logger.swanlab.mode=local \
    experiment_name=$AREAL_EXPERIMENT_NAME \
>>>>>>> 63b44651a1b5c7ce437e7a2f2b21fc4d2815fa79:wisemlops/debug_local_sglang.sh
    trial_name=$TRIAL_NAME