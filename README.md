# **SAGE: Spatially aware Genes Selection and Dual-view Embedding Fusion for Domain Identification in Spatial Transcriptomics**

![figure1](./docs/imgs/SAGE_Logo.png)

## Overview
`SAGE` is a unified framework for identifying spatially informative genes and segmenting spatial tissue domains from spatial transcriptomics data across resolutions. `SAGE` consists of two integrated components: a hybrid **gene selection module** that combines statistical filtering with topic modeling, and **SAGE neural network** for learning spatial representations from both physical and transcriptomic similarity graphs.

![figure1](./docs/imgs/SAGE_overview_Fig_1.png)

Using `SAGE` you can do:
* **Identify spatially informative genes** by integrating variability, topic modeling, and spatial autocorrelation.

* **Segment spatial domain**s using a dual-view graph neural network that captures both physical proximity and gene expression similarity.

* **Discover functional gene modules and spatial co-expression patterns** for biological interpretation.


## Installation 

The installation was tested on a machine with a 40-core Intel(R) Xeon(R) Silver 4210R CPU, 128GB of RAM, and an NVIDIA 3060Ti GPU with 4GB of RAM, using Python 3.12.5, Scanpy 1.10.2, PyTorch 2.4.0. If possible, please run SAGE on CUDA.



### Step1. Install SAGE in the virtual environment by conda

- First, install conda: https://docs.anaconda.com/anaconda/install/index.html
- Then, create a envs named SAGE with python 3.12.5

```
conda create -n SAGE python=3.12.5
conda activate SAGE
```

### Step2.  Install dependency packages

you can install the dependency packages using `pip` by:

```
pip install -r requirements.txt
```

### Step3.  Install SAGE

```
pip install setuptools==58.2.0
python setup.py build
python setup.py install
```

Here, the environment configuration is completed！

## Tutorials

[How to use SAGE](https://github.com/yihe-csu/SAGE/wiki/How-to-use-SAGE): Before using SAGE, activate the `SAGE_ENV` environment and set required environment variables.

[Tutorial 1](https://github.com/yihe-csu/SAGE/wiki/Tutorial-1:-Application-on-10x-Visium-human-dorsolateral-prefrontal-cortex-(DLPFC)-dataset): Application-on-10x-Visium-human-dorsolateral-prefrontal-cortex-(DLPFC)-dataset.


## Reference and Citation
He, Y., Wang, S., Xu, Y., Wang, J. et al. SAGE: Spatially Aware Genes Selection and Multi-view Embedding Fusion for Domain Identification in Spatial Transcriptomics.

## Improvements
We welcome any comments about `SAGE`, and if you find bugs or have any ideas, feel free to leave a comment [FAQ](https://github.com/yihe-csu/SAGE/labels/FAQ).
`SAGE` doesn't fully test on `macOS`.

---

## 

