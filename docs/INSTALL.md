# Installation


## Requirements
- Linux with Python ≥ <font color='red'>3.10</font>.
- PyTorch ≥ <font color='red'>2.0</font> and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -e .`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd psalm/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops
sh make.sh
```

### Example conda environment setup

```bash
conda create --name psalm python=3.10 -y
conda activate psalm
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone https://github.com/zamling/PSALM.git
cd PSALM
pip install -e .
pip install opencv-python addict
cd psalm/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops
sh make.sh
