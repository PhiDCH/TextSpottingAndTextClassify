mv productInfer.py ../productInfer.py
cd ..

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
pip install mmdet
git clone https://github.com/open-mmlab/mmocr.git
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
gdown --id 1WCoawJV2jSTJz_unaQRA31sBbB8Ptah3 -O pan/pretrain/pannet_wordlevel.pth

gdown https://drive.google.com/uc?id=1zWsVDHC-7FlRNTOxHO5fIj2t2kYHu-nA
mkdir textClassify
unzip data.zip -d textClassify/
rm data.zip


