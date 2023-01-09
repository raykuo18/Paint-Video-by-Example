conda create -n $1 python=3.8
conda activate $1
pip3 install cudatoolkit torch torchvision torchaudio
pip3 install "modelscope[cv,multi-modal]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
# pip uninstall mmcv
pip install -U openmim 
mim install mmcv-full
pip3 install -r requirements.txt