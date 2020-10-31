import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from common import *
import logging


def makedirs(dir_name):
  file_path = os.path.join(global_dir,dir_name)    
  os.mkdir(file_path)

dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=10)
#@qml.qnode(dev,interface='torch')

def add_losses_to_generator(gen,disc):
  gen.gen_circuit()
  gen.loss_to_optimize = -disc.generate_output()   
  
#@qml.qnode(dev,interface='torch')
def add_loss_to_discriminator(disc,real=False):
  
  if real==False:
    disc.disc_circuit()
    disc.fake_loss_to_optimize += -disc.generate_output()
    return 
    
  else:
    get_real_data([phi,theta,omega])
    disc.disc_circuit()
    disc.real_loss_to_optimize = disc.generate_output()
 

def get_real_data(angles):
    qml.Hadmard(wires=0)
    qml.Rot(*angles,wires=0)

def global_update_step(gen,disc,step):
  update_disc1 = True
  count_disc = 0
#a global iupdate step for the complete discriminator plus generator model 
  if gen.loss_to_optimize<2*disc.loss_to_optimize:
    disc.update_disc()
  
  elif disc.loss_to_optimize<2*gen :
   gen.update_gen()

  else:
    if update_disc1==True:
      disc.update_disc(disc)
      count_disc+=1
      if count_disc==5:
        update_disc1 =  False
        
    
    else:
      gen.update_gen()
      count_gen+=1
      if count_gen==5:
        update_disc1=True
        count_disc=0
        count_gen=0


def get_optimizer(param_list):
  #returns optimizer given a parameter list with defined learning rate and weight decay

  optimizer = torch.optim.Adam(param_list,lr = lr,weight_decay=weight_decay)
  return optimizer


def add_tensorboard_summary(writer,gen,disc,step):
   writer.add_scalar('Disc_real_loss',disc.real_loss,step)
   writer.add_scalar('Disc_fake_loss',disc.fake_loss,step)
   writer.add_scalar('Disc_total_loss',disc.loss_tot_optimize,step)
   writer.add_scalar('Gen_loss',gen.loss_to_optimize,step)
   writer.add_scalar('Gan_loss',disc.loss_to_optimize+gen.loss_to_optimize,step)
   
def logger():
  logger = logging.getlogger()
  return logger




def get_initial_zero_state(size):
  zero_state = np.zeros(2**size)
  zero_state[0]=1
  zero_state=  np.asmatrix(zero_state).T
  return zero_state

def flatten(x):
  return x.view(s.shape[0],-1)

#def get_entangled_state():



def load_model(model,dir_name,load_file_name):

  if not os.path.isdir(dir_name):
    print("not exists")
    return 
  load_path = dir_name + '/' + load_file_name
  model.load.state_dict(load_path)

def save_model(model,dir_name,save_file_name):
  if not os.path.isdir():
    makedirs(dir_name)
  save_path = dir_name+'/'+save_file_name
  model.save_dict(save_path)

