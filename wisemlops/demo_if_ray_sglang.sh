ray stop --force
rm -rf /tmp/ray
ulimit -n 32768
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
# export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2:$LD_PRELOAD


## ====== MTP dir ====== ##
# MTP_DATASET_HOME=/opt/huawei/dataset/data_sfs
# MTP_CODE_DIR=/cache/algorithm
# cp -r /opt/huawei/schedule-train/algorithm $MTP_CODE_DIR
# cd $MTP_CODE_DIR
export MTP_DATASET_HOME=/home/ma-user/work/dataset/dataset_zhh_guiyang
MTP_CODE_DIR=/home/ma-user/work/dataset/dataset_zhh_guiyang/github/AReaL
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH


## ====== Ascend info ====== ##
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
npu-smi info
echo "========================= driver ========================="
cat /usr/local/Ascend/driver/version.info
echo "========================= toolkit ========================="
cat /usr/local/Ascend/cann/aarch64-linux/ascend_toolkit_install.info


## ====== cluster ====== ##
export NNODES=${MA_NUM_HOSTS:-1}
export NPUS_PER_NODE=${MA_NUM_GPUS:-$(npu-smi info -l | grep -c "NPU ID")}
# export NPUS_PER_NODE=4
export WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
echo "**** WORLD_SIZE: $WORLD_SIZE"

export NODE_RANK=$(($NNODES > 1 ? ${VC_TASK_INDEX:-0} : 0 ))

if [ ${MA_VJ_NAME} ]; then
  MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MASTER_RANK:-0}.${MA_VJ_NAME}"
  CURRENT_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${ORIGINAL_VC_TASK_INDEX:-$NODE_RANK}.${MA_VJ_NAME}"
else
  MASTER_ADDR=localhost
  CURRENT_ADDR=localhost
fi
CURRENT_IP=$(ip -4 addr show $(ip -o -4 route show to default | awk '{print $5}') | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export no_proxy="${CURRENT_IP}${no_proxy:+,$no_proxy}"

echo "============================"
ip -4 addr show $(ip -o -4 route show to default | awk '{print $5}') | grep -oP '(?<=inet\s)\d+(\.\d+){3}'
echo "============================"
ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}'
echo "============================"
hostname -i | awk '{print $1}'
echo "============================"
python -c "import socket; print(socket.gethostbyname('${CURRENT_ADDR}'))"
echo "============================"

## ====== device ====== ##
# export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3


## ====== hccl ====== ##
# export HCCL_IF_BASE_PORT=64000
export HCCL_ASYNC_ERROR_HANDLING=0
# export HCCL_EXEC_TIMEOUT=3600
# export HCCL_CONNECT_TIMEOUT=3600
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

## ====== file based barrier ====== ##
export FLAG_PATH=$MTP_DATASET_HOME/outputs/flags/${MA_VJ_NAME:-webstudio_run}
_timestamp=$(date +%y%m%d%H%M%S)
mkdir -p "$FLAG_PATH/$NODE_RANK/$_timestamp"

all_ranks_ready() {
    for ((_rank=0; _rank<NNODES; _rank++)); do
        if [ ! -d "${FLAG_PATH}/$_rank" ]; then
            echo "NODE_RANK $_rank is not ready..."
            return 1
        fi
    done
    return 0
}

while ! all_ranks_ready; do
    sleep 5
done
echo "All nodes are ready. Proceeding..."


## ====== sglang ====== ##
HF_MODEL_NAME=ailab_slm_0_5b____v2
## sglang adaptation of ailab_slm
cp -r $MTP_DATASET_HOME/sglang_model/ailab_slm_v0_5_9.py /home/ma-user/install/sglang/python/sglang/srt/models/ailab_slm_v0_5_9.py
## soft link the ckpt from training dir to huggingface dir on the local machine
mkdir -p /cache/hf_model
cp -r $MTP_DATASET_HOME/hf_model/$HF_MODEL_NAME /cache/hf_model/$HF_MODEL_NAME
ln -sf $MTP_DATASET_HOME/ckpt/mcore/mtp_251202_154830__ailab_slm_0_5b____v2__mtp_251124_135004__ailab_slm_0_5b____v2__slm_archv2_muon_pretrain_8t_v12_lr5e3_muon_cooldown_mtp01_v6v4_320b/mg2hf/iter_0012500/model.safetensors /cache/hf_model/$HF_MODEL_NAME/model.safetensors


## ====== areal ====== ##
AREAL_EXPERIMENT_NAME=if-grpo-ailab_0_5b-251202_154830-910b
AREAL_ALLOCATION_MODE=${AREAL_ALLOCATION_MODE:-'sglang:d4p1t1+d4p1t1'}

AREAL_JOB_TIMESTAMP=$(find "$FLAG_PATH" -mindepth 2 -maxdepth 2 -type d -name "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]" -printf "%f\n" | sort | tail -n1)
AREAL_RAY_LOGS_DIR=$MTP_DATASET_HOME/logs/$AREAL_EXPERIMENT_NAME/$AREAL_JOB_TIMESTAMP/ray_logs
AREAL_TENSORBOARD_DIR=$MTP_DATASET_HOME/tensorboard/areal/$AREAL_EXPERIMENT_NAME/$AREAL_JOB_TIMESTAMP
mkdir -p $AREAL_RAY_LOGS_DIR
export NLTK_DATA=$MTP_DATASET_HOME/nltk_data


## ====== ray start ====== ##
if [ "$MASTER_ADDR" = "$CURRENT_ADDR" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE', "NODE_RANK_'${NODE_RANK}'": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          # +stats_logger.tensorboard.path=$AREAL_TENSORBOARD_DIR \
          # actor.kl_ctl=0.1 \
          # +gconfig.enable_thinking=True \
          python examples/instruction_following/instruction_following_rl.py \
            --config examples/instruction_following/if_grpo_npu.yaml \
            scheduler.type=ray \
            allocation_mode=$AREAL_ALLOCATION_MODE \
            cluster.n_nodes=$NNODES \
            cluster.n_gpus_per_node=$NPUS_PER_NODE \
            +gconfig.stop_token_ids=[340] \
            total_train_epochs=1 \
            actor.kl_ctl=0.01 \
            actor.optimizer.lr=3.0e-5 \
            rollout.max_head_offpolicyness=2 \
            cluster.fileroot=$MTP_DATASET_HOME/ckpt/areal \
            cluster.name_resolve.nfs_record_root=$MTP_DATASET_HOME/outputs/areal/name_resolve \
            actor.path=/cache/hf_model/$HF_MODEL_NAME \
            actor.scheduling_spec.0.env_vars.TASK_QUEUE_ENABLE=1 \
            train_dataset.path=$MTP_DATASET_HOME/hf_data/IF_multi_constraints_upto5 \
            +train_dataset.format=instruction_following \
            +train_dataset.split=train \
            valid_dataset.path=$MTP_DATASET_HOME/hf_data/IFEval \
            +valid_dataset.format=instruction_following \
            +valid_dataset.split=train \
            evaluator.freq_epochs=null \
            evaluator.freq_steps=91 \
            +evaluator.eval_before_train=True \
            +stats_logger.swanlab.mode=local \
            experiment_name=$AREAL_EXPERIMENT_NAME \
            trial_name=$AREAL_JOB_TIMESTAMP
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE', "NODE_RANK_'${NODE_RANK}'": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Node $NODE_RANK Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

## copy ray logs
cp -r /tmp/ray/session_latest/logs $AREAL_RAY_LOGS_DIR/$NODE_RANK

echo "**** END ****"

sleep 600