# Federated Learning on Non-IID Data with Local-drift Decoupling and Correction
Code for paper - **[Federated Learning on Non-IID Data with Local-drift Decoupling and Correction]**

We provide code to run FedDC, FedAvg, 
[FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w), 
[Scaffold](https://openreview.net/pdf?id=B7v4QMR6Z9w), and [FedProx](https://arxiv.org/abs/1812.06127) methods.


## Prerequisite
* Install the libraries listed in requirements.txt
    ```
    pip install -r requirements.txt
    ```

## Datasets preparation
**We give datasets for the benchmark, including CIFAR10, CIFAR100, MNIST, EMNIST-L and the synthetic dataset.**




You can obtain the datasets when you first time run the code on CIFAR10, CIFAR100, MNIST, synthetic datasets.
EMNIST needs to be downloaded from this [link](https://www.nist.gov/itl/products-and-services/emnist-dataset).


For example, you can follow the following steps to run the experiments:

```python example_code_mnist.py```
```python example_code_cifar10.py```
```python example_code_cifar100.py```

1. Run the following script to run experiments on the MNIST dataset for all above methods:
    ```
    python example_code_mnist.py
    ```
2. Run the following script to run experiments on CIFAR10 for all above methods:
    ```
    python example_code_cifar10.py
    ```
3. Run the following script to run experiments on CIFAR100 for all above methods:
    ```
    python example_code_cifar10.py
    ```
4. To show the convergence plots, we use the tensorboardX package. As an example to show the results which stored in "./Folder/Runs/CIFAR100_100_23_iid_":
    ```
    tensorboard --logdir=./Folder/Runs/CIFAR10_100_23_iid
    ```
5. Get the url, and then enter the url in to the web browser, for example "http://localhost:6006/".

   
## Generate IID and Dirichlet distributions:
Modify the DatasetObject() function in the example code.
CIFAR-10 IID, 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=17, rule='iid', unbalanced_sgm=0, data_path=data_path)
```
CIFAR-10 Dirichlet (0.3), 100 partitions, balanced data
```
data_obj = DatasetObject(dataset='CIFAR10', n_client=100, seed=47, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)
```

    
## FedDC 
The FedDC method is implemented in ```utils_methods_FedDC.py```. The baseline methods are stored in ```utils_methods.py```.

### Citation

```
@inproceedings{
gao2022federated,
title={FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction},
author={Liang Gao and Huazhu Fu and Li Li and Yingwen Chen and Ming Xu and Cheng-Zhong Xu},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
year={2022}
}
```
