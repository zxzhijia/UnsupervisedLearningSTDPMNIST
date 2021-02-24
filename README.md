##  Updated for Py3
# Brian 2 version source code for paper "Unsupervised Learning of Digit recognition using STDP"

This small project is to translate Brian 1 code from Paper "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" written by PU Diehl to Brian 2 version. 


## Prerequisite
1. Brian 2 
2. MNIST datasets, which can be downloaded from http://yann.lecun.com/exdb/mnist/. 
   * The data set includes four gz files. Extract them after you downloaded them.

## Testing with pretrained weights:

1. Run the main file "Diehl&Cook_spiking_MNIST_Brian2.py". It might take hours depending on your computer 
2. After the previous step is finished, evaluate it by running "Diehl&Cook_MNIST_evaluation.py".

## Training a new network:

1. modify the main file "Diehl&Cook_spiking_MNIST_Brian2.py" by changing line 214 to "test_mode=False" and run the code. 
2. The trained weights will be stored in the folder "weights", which can be used to test the performance.
3. In order to test your training, change line 214 back to "test_mode=True". 
4. Run the "Diehl&Cook_spiking_MNIST_Brian2.py" code to get the results. 
