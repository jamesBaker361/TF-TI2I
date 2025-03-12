conda create -n tfti2i python=3.9 -y
conda activate tfti2i
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade diffusers[torch]
pip install transformers
pip install protobuf
pip install sentencepiece