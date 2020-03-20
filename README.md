# Analysis of Image-To-Image GANs

This project looks to investigate the performance of various Image-To-Image Generative Adversarial Networks across a set number of datasets on a set number of metrics. This project is currently ongoing and this README will be updated to reflect progress on this project.

## Running this project

1. Install [Conda](https://docs.anaconda.com/anaconda/install/)
2. Clone this repository and recursively add all submodules
3. Create the conda environment
4. Start Jupyter Lab and Open [analyze.ipynb](/analyze.ipynb)

```bash
git --recurse-submodules clone git@github.com:sagars729/GAN.git
cd GAN
conda env create -f gan_env.yaml
conda activate gan
jupyter lab
```
