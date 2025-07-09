from setuptools import setup, find_packages

setup(
    name     = 'SAGE',
    version  = '1.1.8',
    keywords =["spatial transcriptomics", "Graph neural networks", "Contrastive learning"],
    description = 'Spatially Aware Genes Selection and Dual-view Embedding Fusion for Domain Identification in Spatial Transcriptomics',
    license  = 'MIT License',
    url      = 'https://github.com/yihe-csu/SAGE',
    author   = 'Yi He',
    author_email = 'heyi@stu.csust.edu.cn',
    packages     = find_packages(),
    include_package_data = True,
    platforms    = 'any',
        install_requires = [
            "leidenalg==0.10.2",
            "matplotlib==3.10.3",
            "numpy==2.3.1",
            "pandas==2.3.1",
            "POT==0.9.4",
            "python-igraph==0.11.6",
            "rpy2==3.5.16",
            "scanpy==1.11.3",
            "scikit-learn==1.7.0",
            "seaborn==0.13.2",
            "torch==2.5.0+cu124",
            "torch-geometric==2.6.1",
            "tqdm==4.66.5",
            "umap-learn==0.5.6"
        ],
    )