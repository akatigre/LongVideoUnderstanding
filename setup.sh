# Python 3.9.19，cuda12.2，torch2.3.0 flash_attn==2.5.8 minference==0.1.5.post1
pip3 install transformers==4.47.1 # https://github.com/microsoft/MInference/issues/112
pip3 install protobuf
pip3 install triton==3.1.0
pip3 install -U torch==2.5.1 torchvision==0.18.0 torchaudio==2.5.1

# pip3 install -U https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip3 install https://github.com/microsoft/MInference/releases/download/v0.1.5.post1/minference-0.1.5.post1+cu122torch2.3-cp39-cp39-linux_x86_64.whl

git clone https://github.com/EvolvingLMMs-Lab/LongVA.git && mv LongVA/longva/longva . && rm -rf LongVA
git clone https://github.com/microsoft/MInference.git && pip3 install -e MInference
pip3 install packaging ninja decord
