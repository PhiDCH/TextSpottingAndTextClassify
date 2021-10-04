mv productInfer.py ../productInfer.py
cd ..

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
pip install mmdet
git clone https://github.com/PhiDCH/mmocr mmocr1
cd mmocr1
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
rm -rf demo docker docs docs_zh_CN mmocr.egg-info resources tests tools 
rm MANIFEST.in README.md README_zh-CN.md model-index.yml setup.cfg setup.py
cd ..

git clone https://github.com/PhiDCH/PAN.pytorch pan
cd pan/post_processing/
rm -rf pse.so
make
pip install polygon3
pip install pyclipper
pip install colorlog
python -m pip install Pillow==6.2
cd ..
rm -rf base config data_loader imgs result test_img trainer utils
rm PAN.ipynb README.MD train.py eval.py config.json
cd ..
gdown --id 1V7X_ZS4ZhuZhCxbhhKO8B_1PGWQyKfdI -O pan/pretrain/pannet_wordlevel.pth

gdown https://drive.google.com/uc?id=1zWsVDHC-7FlRNTOxHO5fIj2t2kYHu-nA
mkdir textClassify
unzip data.zip -d textClassify/
rm data.zip


git clone https://github.com/PhiDCH/CRAFT-pytorch CRAFTpytorch
cd CRAFTpytorch 
gdown --id 1RyaO4cxaV62L78AEAvrrPeaX9_JLLqXB -O craft_mlt_25k.pth
cd ..

git clone https://github.com/PhiDCH/text-recognition textRecognition
cd textRecognition/
gdown --id 1U4KQQ36LCS4HBr3hKXQC-D7exRb2aLci -O best_accuracy.pth
cd ..
