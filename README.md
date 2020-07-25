[![Documentation Status](https://readthedocs.org/projects/pysigma/badge/?version=latest)](https://pysigma.readthedocs.io/en/latest/?badge=latest)


# PySigma

Python implementation of the Sigma cognitive and graphical architecture grounded on PyTorch tensor processing.

For documentations and discussions of design crateria, please check the [wiki page](https://bitbucket.org/TownesZhouLCC/pysigma/wiki/Home) of this repo.

## Install

PySigma requires Python version 3.7. It is recommended that you use a conda virtual environment to install necessary packages for PySigma:
```bash
conda create -n pysigma python=3.7
```
To activate the conda virtual environment, type the command:
```bash
conda activate pysigma
```
After activating the virtual environment, you need to install required packages using pip:
```bash
pip install -r requirements.txt
```
Note that the `requirements.txt` file only includes cpu-based PyTorch. To enable GPU acceleration, you need to install the correct version that supports CUDA. Follow [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) on how to do this. 


Thanks, and have fun!  
