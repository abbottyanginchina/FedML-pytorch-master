
# Federated Learning Framework for Experiment using Pytorch [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4321561.svg)](https://doi.org/10.5281/zenodo.4321561)


Note: The scripts will be slow without the implementation of parallel computing. 

## Dataset
Only experiments on MNIST and EMNIST is produced by far. The kinds of distribution of the dataset are as follow:

* Prepares IID training datasets for each client

* Prepares NIID-1 training datasets for each client (Overlapping sample sets)

* Prepares NIID-2 training datasets for each client (Unequal data distribution)

* Prepares NIID-1+2 training datasets for each client(Overlapping sample sets + Unequal data distribution)

## Models
There are three different models in this project: 
* Small
* Medium
* Large


## Requirements
python == 3.8  
pytorch == 1.8

## Run
Federated learning is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [parameters.py](utils/parameters.py). 

For example:
> python main_fed.py --dataset MNIST --distribution_type IID --model_size SMALL



