# Genetic Algortihm
Autor: Luis Cossio

Implementation of genetic programming for the task of finding an aproximation of a function and to solve the problema of des chiffres et des lettres


## Description
- 

## Structure
The structure of the project is:
  ```
  genetic_programming
  ├── Genetic_program.py
  ├── Individuals.py
  ├── test_individuals.py
  ├── ex_des_chiffres_et_des_lettres.py
  ├── ex_function_estimation.py
  ├── requirements.txt
  └── README.md
  ```

## Deployment
To use the executable file ex_des_chiffres_et_des_lettres.py use in the folder a command like this one:
```
python ex_des_chiffres_et_des_lettres.py --population 50 --inputs 1 2 3 4 5 6 7 8 --output 23 --mutation 0.3 --epochs 50 --end-condition 0
```


## Description of Algorithm
*   Training and fitting calculation is done by feeding the GA with samples of the image and expected ground truth:

  
*   A sample is consider as an image of blood cells from the BCCD dataset and the corresponding mask where there is a bounding box marking where there is a white blood cell. 
  ![example](./resources images/image_and_mask.png)

*   The Genetic algorithm attempts to found a series of filters that can replicate the ground truth of the samples. 

*   A filter consist of a tensor of shape ``[row_kernel,col_kernel,channels_in,channels_out]``. An individual in this context consist of N filters, that defines N convolutional layers.  

*   The filters operates over a batch of input images in a Convolutional neural network (CNN) like manner, where each filter defines a layer. A convolution layer consist of the next operations:
    
    1- Convolution 2D between tensor and filter 
    
    2- Activation Function (Relu for all layers but the last one)
    
    3- Pooling 
    
    4- Pseudo Normalization
    
*   The last Layer uses a sigmoid function to model and output mask defined between [0,1]. Said result is then re-scaled so the maximum value of the output mask is 1.0. This is done because some output masks achieved a max value below 1.0, for example 0.74, but given that said value is the maximum in all the mask it's locigal to take as 1.0, and scale all of the rest result accordingly.    
     
*   Score is then calculated as the -(cross-entropy) of the mask output, with the predicted mask.  
    
## Results 
The initial best filter used in the test dataset produce the next image:
  ![example](./resources images/result.png)


The performance of the method for differents GA parameters (mutation rate and population size), in the test dataset was:

  ![example](./resources images/heatmap.png)
## Libraries
numpys matplotlib scipy scikit-learn


## Installation
## Setting up BCCD repository
In order to download the dataset and files use the kaggle dataset link [BCCD](https://www.kaggle.com/surajiiitm/bccd-dataset). 
Download and install in such a way that this repository folder follows the structure explain previously. 
 
### Install python
In order to install the programing language follow the installation instructions in the official site:

https://www.python.org/downloads/ 

### Install Pip
In case of not having an installation tool, it is recommended install pip, following the instruccions of the official site:

https://pip.pypa.io/en/stable/reference/pip_install/

### Install libraries
To install the libraries just run the next code in the commando line using pip:
```
pip install -r requirements.txt
```

## Info
Language: Python
 
Version: python 3.7