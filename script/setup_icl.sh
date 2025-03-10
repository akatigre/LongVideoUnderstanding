IMAGE_NAME=qwenllm/qwenvl:2.5-cu121
CONTAINER_NAME=qwen2.5-vl
CODE_DIR=/home/server08/yoonjeon_workspace/VideoPrefill
DATA_DIR=/home/server08/hdd1/yoonjeon_workspace/ood_qa
# if [ ! -e ${QWEN_CHECKPOINT_PATH}/config.json ]; then
#     echo "Checkpoint config.json file not found in ${QWEN_CHECKPOINT_PATH}, exit."
#     exit 1
# fi
if sudo docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} exists, skip pulling."
else
    sudo docker pull ${IMAGE_NAME} || {
        echo "Pulling image ${IMAGE_NAME} failed, exit."
        exit 1
    }
fi

sudo docker run --gpus all \
    --name ${CONTAINER_NAME} \
    --rm \
    -v /home/server08/.cache/huggingface/:/root/.cache/huggingface \
    -v ${DATA_DIR}:${DATA_DIR} \
    -v ${CODE_DIR}:${CODE_DIR} \
    -w ${CODE_DIR} \
    --shm-size=8g \
    -it ${IMAGE_NAME} 