@echo off
python -m pip install --upgrade pip
pip install tensorflow
pip install opencv-python
pip install sklearn
pip install numpy
pip install keyboard
pip install primesense
cd externals
git clone --recursive -j8 -o "openpose" https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
echo "You have to build openpose with cmake."