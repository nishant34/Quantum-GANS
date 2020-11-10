import numpy as np
import os
import time

global_dir = 'Desktop/quantum'

log_dir = global_dir + '/' + 'log_files'

initial_load = False
n_qbits = 2
lr = 0.00001
Beta = 0.97
num_epochs= 200
num_steps_per_epoch = 50
epoch_per_check = 5
model_save_path = 'Dektop/quantum/saved_model'
tensorboard_log_path = '/Desktop/quantum/logging/'
model_load_path = model_save_path

loss_function_type = "BCE"
weight_decay = 0.1
num_steps_for_unrolling = 5
 
