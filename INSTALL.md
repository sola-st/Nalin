#### Python Packages

The required packages are listed in _requirements.txt_. The packages may be installed using the command ```pip install -r requirements.txt```. 
Additionally, install the [PyTorch](https://pytorch.org/get-started/locally/) package (We have tested on PyTorch version 1.10.1).  

```shell
pip install -r requirements.txt
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

💡 The above command will install the **CPU** version of PyTorch. If you want CUDA support, please change the command accordingly as mentioned in the [link](https://pytorch.org/get-started/locally).

#### Jupyter Notebook Dataset

We use the dataset from a CHI’18 [paper](https://dl.acm.org/doi/10.1145/3173574.3173606) that has analyzed more than 1.3 million publicly available Jupyter 
Notebooks from GitHub. Download the dataset using the [link](https://library.ucsd.edu/dc/collection/bb6931851t).
We provide a sample of about 2000 Jupyter notebooks (_benchmark/jupyter_notebook_datasets/sample.zip_) obtained from this dataset for testing. 

#### Embedding

Download the embedding file from the [link](https://u.pcloud.link/publink/show?code=XZyeJaXZrnrbvwzBcYSOWYgzsn4usJ6DOqPy) and put in the _benchmark_ folder.
