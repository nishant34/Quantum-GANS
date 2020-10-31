import os
import torch
import numpy as np
import pennylane as qml
from torch.autograd import Variable
import torch.nn as nn
from train_helper import *
np.random.seed()

np.random.seed()
dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=10)
class Generator(nn.Module):
    def __init__(self,size_system):
        super(Generator,self)
        self.size = size_system
        self.eps=0.01
        #
        #self.gate_count = 0
        #self.weights = nn.Parameter(torch.randn(2),requires_grad = True) 
        self.weights1 = np.array([np.pi] + [0] * 8) + \
                   np.random.normal(scale=self.eps, size=(9,))
        self.weights = Variable(torch.tensor(self.weights1))

    def reset_angles(self):
        theta = np.random.random(len(self.gate_count))
        for i in range(self.gate_count):
            self.weights[i] = theta[i]
    
    def set_qcircuit(self,qc):
        self.qc = qc()

    def gen_circuit(self):
        qml.Hadamard(wires=0)
        qml.RX(self.weights[0].item(), wires=0)
        qml.RX(self.weights[1].item(), wires=1)
        qml.RY(self.weights[2].item(), wires=0)
        qml.RY(self.weights[3].item(), wires=1)
        qml.RZ(self.weights[4].item(), wires=0)
        qml.RZ(self.weights[5].item(), wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(self.weights[6].item(), wires=0)
        qml.RY(self.weights[7].item(), wires=0)
        qml.RZ(self.weights[8].item(), wires=0)            

    def update_gen(self,step):
        cost_function_to_optimize = self.loss_to_optimize
        parameters_to_optimize = self.weights    
        optimizer= get_optimizer(parameters_to_optimize)

        cost_function_to_optimize.backward()
        optimizer.step()

class Discriminator(object):
    def __init__(self,size_system):
        self.size = size_system
        #self.qc = qml.device("forest.qpu",device="Aspen-1-2Q-B")
        #@qml.qnode(dev,interface='torch')
        #self.gate_count = gate_count
        self.weights_real = nn.Parameter(torch.randn(2),requires_grad = True)
        self.weights_fake =  nn.Parameter(torch.randn(2),requires_grad = True)
        self.w = Variable(torch.tensor(np.random.normal(size=(9,))))
        

    def reset_angles(self):
        theta = np.random.random(len(self.gate_count))
        for i in range(self.gate_count):
            self.weights_real[i] = theta[i]
            self.weights_fake[i] = theta[i]
    
    def set_qcircuit(self,qc):
        self.qc = qc()

    def disc_circuit(self):
        qml.Hadamard(wires=0)
        qml.RX(self.w[0], wires=0)
        qml.RX(self.w[1], wires=1)
        qml.RY(self.w[2], wires=0)
        qml.RY(self.w[3], wires=1)
        qml.RZ(self.w[4], wires=0)
        qml.RZ(self.w[5], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(self.w[6], wires=1)
        qml.RY(self.w[7], wires=1)
        qml.RZ(self.w[8], wires=1)            
    
    def update_disc(self,step):
        cost_function_to_optimize = self.real_loss_to_optimize + self.fake_loss_to_optimize
        parameters_to_optimize = self.weights    
        optimizer= get_optimizer(parameters_to_optimize)

        cost_function_to_optimize.backward()
        optimizer.step()
    @qml.qnode(dev,interface='torch')
    def generate_output(self):
        return qml.expval(qml.PauliZ(1))
        

