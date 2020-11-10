
import pennylane as qml
import numpy as  np
import time
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from model_qwgan import *
from common import *
from train_helper import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter,summary
#import pennylane as qml


def get_real_data(angles):
    qml.Hadamard(wires=0)
    qml.Rot(*angles,wires=0)

def main():
 #initialise data to feed to the model
 real_angles = [np.pi/3,np.pi/4,np.pi/5]

  

 #define model
 #model_gen = Generator(n_qbits)
 #model_disc  = Discriminator(n_qbits)
 
 
 #initialise summary
 writer = SummaryWriter(log_dir=tensorboard_log_path)
 
 #parse model
 
 #initialise losses 
 add_losses_to_generator()
 add_loss_to_discriminator(real=False)
 add_loss_to_discriminator(real=True)
 model_gen = gen
 model_disc  = disc


 #load data
 
 if initial_load==True:
     model_disc_load_name = 'model__' + format(initial_load_epoch,'05') + '__disc.pth'
     load(model_disc,model_load_path,model_disc_load_name)
     model_gen_load_name = 'model__' + format(initial_load_epoch,'05') + '__gen.pth'
     load(model_gen,model_load_path,model_gen_load_name)
     

 #start epochs
 Logger = logging.getLogger("LG")
 Logger.info("Total Epochs:{}".format(num_epochs))
 Logger.info("Learning_rate:{}".format(lr))
 Logger.info("Beta:{}".format(Beta))
 Logger.info("Number of Qbits:{}".format(n_qbits))
 Logger.info("Loss_function_type:{}".format(loss_function_type))
 for epoch in range(num_epochs):
     print("global_epoch:{}".format(epoch),"Unrolling begins")
     #if device=='cuda'
     disc_train = True
     count_disc = 0
     count_gan = 0
     
     for step in range(num_steps_per_epoch):
      if disc_train:        
        #model_disc.update_disc(step)
        count_disc+=1
        if count_disc>=num_steps_for_unrolling:
            disc_train = False
            count_disc=0
      else:
          #model_gen.update_gen(step)
          count_gen+=1
          if count_gen>=num_steps_for_unrolling:
              disc_train = True
              count_gen = 0
      print("----------------------------------------------------------------------------") 
      print("disc_loss:{}".format(model_disc.fake_loss_to_opimize+model_disc.real_loss_to_optimize),"disc_fake_loss:{}".format(model_disc.fake_loss_to_opimize),"disc_real_loss:{}".format(model_disc.fake_loss_to_opimize),"generator_loss:{}".format(model_gen.loss_to_optimize))
      add_tensorboard_summary(writer,model_gen,model_disc,epoch)

     if epoch%epoch_per_check==0:
          model_disc_save_name = 'model__' + format(epoch,'05') + '__disc.pth'
          save_model(model_disc,model_save_path,model_disc_load_name)
          model_gen_save_name = 'model__' + format(epoch,'05') + '__gen.pth'
          save_model(model_gen,model_save_path,model_gen_save_name)
          print("model svaed at epoch:{}".format(epoch))
     
         

if __name__=="__main__":
  main()


 


 
    

