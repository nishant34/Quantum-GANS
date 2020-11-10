# Quantum-GANS
* Implementation of Quantum Gans using Pennylane capable of creating quantum simualator and quantum nodes and backpropagating through them .
* Angles of rotation gates are the paramters in case of quantum circuits .

# Setting up the environment to work in:

* Install Anaconda
* Then create and activate a conda environement
```javascript
conda create -n myenv
source activate myenv

```
Then following requiremenst are there.
* python 3.6 or gretaer will work(lower versions should also work).
* pytorch 1.14 

Many simulators are there or QPUs which can be installed depending on what works with the system
To install pennylane :
```javascript
pip install pennylane
pip install pennylane-sf
```
SF is the strawberry fields simulator.

To install forest simulator :
```javascript
pip install pennylane-forest
```
To install cirq simulator :
```javascript
pip install pennylane-cirq
```



# Training the Quantum GANS:
The structure is as follows:
First specify appropriate locations and parameters in the common.py file
To train the model run:
```javascript
python train.py
```
Tensorboard logs will be generated in the tensorboard_dir.
To see the tensorboard visualisations:
```javascript
python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]
```
A link will appear on the command line  and on opening it in the browser tensorboard visualisations will be visible.
For circuits define quantum nodes so that backpropagation is possible through them.




