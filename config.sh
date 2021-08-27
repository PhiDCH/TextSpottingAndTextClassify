mv productInfer.py ../productInfer.py
cd ..

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
pip install mmdet
git clone https://github.com/PhiDCH/mmocr
cd mmocr
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

git clone https://github.com/PhiDCH/PAN.pytorch pan
cd pan/post_processing/
rm -rf pse.so
make
pip install polygon3
pip install pyclipper
pip install colorlog
python -m pip install Pillow==6.2
cd ../..
gdown --id 1FpVf64mfAyaHQ70AV2rUfY3SydLwFq0W -O pan/pretrain/pannet_wordlevel.pth

gdown https://drive.google.com/uc?id=1zWsVDHC-7FlRNTOxHO5fIj2t2kYHu-nA
mkdir textClassify
unzip data.zip -d textClassify/
rm data.zip

git clone https://github.com/PhiDCH/CRAFT-pytorch
cd CRAFT-pytorch 
gdown --id 1RyaO4cxaV62L78AEAvrrPeaX9_JLLqXB -O craft_mlt_25k.pth
cd ..


