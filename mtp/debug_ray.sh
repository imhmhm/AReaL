ray stop --force
rm -rf /tmp/ray
ulimit -n 32768
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
# export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2:$LD_PRELOAD


## ====== MTP dir ====== ##
export MTP_DATASET_HOME=/home/ma-user/work/dataset/dataset_zhh_guiyang
# MTP_CODE_DIR=/opt/huawei/schedule-train/algorithm
MTP_CODE_DIR=/home/ma-user/work/dataset/dataset_zhh_guiyang/github/AReaL
export PYTHONPATH=$MTP_CODE_DIR:$PYTHONPATH
# cp -r /opt/huawei/schedule-train/algorithm $MTP_CODE_DIR
# cd $MTP_CODE_DIR


## ====== Ascend info ====== ##
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
npu-smi info
echo "========================= driver ========================="
cat /usr/local/Ascend/driver/version.info
echo "========================= toolkit ========================="
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
# cat /usr/local/Ascend/cann/aarch64-linux/ascend_toolkit_install.info


## ====== cluster ====== ##
export NNODES=${MA_NUM_HOSTS:-1}
export NPUS_PER_NODE=${MA_NUM_GPUS:-$(npu-smi info -l | grep -c "NPU ID")}
export WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
echo "**** WORLD_SIZE: $WORLD_SIZE"

export NODE_RANK=$(($NNODES > 1 ? ${VC_TASK_INDEX:-0} : 0 ))

if [ ${MA_VJ_NAME} ]; then
  MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MASTER_RANK:-0}.${MA_VJ_NAME}"
  CURRENT_IP="${MA_VJ_NAME}-${MA_TASK_NAME}-${ORIGINAL_VC_TASK_INDEX:-$NODE_RANK}.${MA_VJ_NAME}"
  # CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
else
  MASTER_ADDR=localhost
  CURRENT_IP=localhost
fi


## ====== device ====== ##
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8


## ====== hccl ====== ##
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050


# ## ====== torch_npu ====== ##
# export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"


# ## ====== vllm ====== ##
# export TASK_QUEUE_ENABLE=1
# export VLLM_ASCEND_ENABLE_NZ=0
# export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=0


## ====== file based barrier ====== ##
export FLAG_PATH=$MTP_DATASET_HOME/outputs/flags/${MA_VJ_NAME:-webstudio_run}
_timestamp=$(date +%y%m%d_%H%M%S)
mkdir -p "$FLAG_PATH/$NODE_RANK/$_timestamp"

all_ranks_ready() {
    for ((_rank=0; _rank<NNODES; _rank++)); do
        if [ ! -d "${FLAG_PATH}/$_rank" ]; then
            echo "NODE_RANK $_rank not ready"
            return 1
        fi
    done
    return 0
}

while ! all_ranks_ready; do
    sleep 5
done
echo "All nodes are ready. Proceeding..."


## ====== areal ====== ##
AREAL_EXPERIMENT_NAME=gsm8k-grpo-qwen25_1_5b_910b
TRIAL_NAME=$(date +%y%m%d_%H%M%S)


# ## ====== verl ====== ##
# VERL_JOB_TIMESTAMP=$(find "$FLAG_PATH" -mindepth 2 -maxdepth 2 -type d -name "[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]" -printf "%f\n" | sort | tail -n1)
# export HF_MODEL_NAME='Qwen2.5-0.5B-Instruct'
# export VERL_PROJECT_NAME='verl_v061'
# export VERL_EXPERIMENT_NAME=${VERL_JOB_TIMESTAMP}__${HF_MODEL_NAME}

# export VERL_FILE_LOGGER_ROOT=$MTP_DATASET_HOME/logs
# VERL_RAY_LOGS_DIR=$VERL_FILE_LOGGER_ROOT/$VERL_PROJECT_NAME/$VERL_EXPERIMENT_NAME/ray_logs
# mkdir -p $VERL_RAY_LOGS_DIR
# export TENSORBOARD_DIR=$MTP_DATASET_HOME/tensorboard/$VERL_PROJECT_NAME/$VERL_EXPERIMENT_NAME
# export VERL_CKPT_DIR=$MTP_DATASET_HOME/ckpt


# ## ====== 替换yaml中的环境变量 ====== ##
# DEFAULT_YAML=""
# YAML=${1:-$DEFAULT_YAML}
# echo "**** Use $YAML"
# envsubst < configs/$YAML.yaml > configs/temp.yaml
# mv configs/temp.yaml configs/$YAML.yaml
# cat configs/$YAML.yaml
# echo


## ====== ray start ====== ##
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
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
          python examples/math/gsm8k_rl.py \
              --config examples/math/gsm8k_grpo_npu.yaml \
              scheduler.type=ray \
              cluster.fileroot=$MTP_DATASET_HOME/areal/experiments \
              cluster.name_resolve.nfs_record_root=$MTP_DATASET_HOME/outputs/areal/name_resolve \
              actor.path=$MTP_DATASET_HOME/hf_model/Qwen2.5-1.5B-Instruct \
              train_dataset.path=$MTP_DATASET_HOME/hf_data/gsm8k \
              valid_dataset.path=$MTP_DATASET_HOME/hf_data/gsm8k \
              +stats_logger.tensorboard.path=$MTP_DATASET_HOME/outputs/areal/tensorboard/${AREAL_EXPERIMENT_NAME}/${TRIAL_NAME} \
              experiment_name=$AREAL_EXPERIMENT_NAME \
              trial_name=$TRIAL_NAME
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

# ## copy ray logs
# cp -r /tmp/ray/session_latest/logs $VERL_RAY_LOGS_DIR/$NODE_RANK

echo "**** END ****"

sleep 600