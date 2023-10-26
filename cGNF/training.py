
# cGNF training function wrapper with some important arguments

import os
import math
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.cuda
import torch.backends.cudnn as cudnn
from timeit import default_timer as timer
import pickle

# install the modules
from cGNF.GNF_Modules.Normalizers import *
from cGNF.GNF_Modules.Conditionners import *
from cGNF.GNF_Modules.NormalizingFlowFactories import buildFCNormalizingFlow
from cGNF.GNF_Modules.NormalizingFlow import *

cond_types = {"DAG": DAGConditioner, "Coupling": CouplingConditioner,
              "Autoregressive": AutoregressiveConditioner}  # types of conditioners
norm_types = {"affine": AffineNormalizer, "monotonic": MonotonicNormalizer}  # types of transformers/normalizers

def train(path="", dataset_name="" , path_save="",
          trn_batch_size=128, val_batch_size=1024, learning_rate=1e-4, seed=None, nb_epoch=10000,
          emb_net=[100, 90, 80, 70, 60],
          int_net=[60, 50, 40, 30, 20], nb_estop=50, val_freq =1):

    # set cuda device for pytorch if available
    device = "cpu" if not (torch.cuda.is_available()) else "cuda:0"

    # Set the seed for random number generation. If not provided, select a random seed between 1 and 20000
    if seed is None:
        seed = random.randint(1, 20000)
    print(f"Running simulation with seed {seed}")

    # setting the seed for random number generators in python, numpy and pytorch for reproducible results
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # ensures that the CUDA backend for PyTorch (CuDNN) uses deterministic algorithms.
        torch.backends.cudnn.benchmark = True # enables the CuDNN autotuner, which selects the best algorithm for CuDNN operations given the current hardware setup.
    else:
        print("GPU device not available!")

    if not (os.path.isdir(path_save)):  # checks if a directory with the name 'path_save' exists.
        os.makedirs(path_save)  # if not, creates a new directory with this name. This is where the logs and model weights will be saved.

    # Load dataset for training and inference
    with open(path + dataset_name + '.pkl', 'rb') as f:
        data = pickle.load(f)

    # Load data from the dataset
    mu = data['mu']
    sig = data['sig']
    trn = data['trn']
    val = data['val']
    A = data['A']
    cat_dims = data['cat_dims']
    dataset_filepath = data['dataset_filepath']

    # Convert numpy data to torch tensors and move to device
    data_mu = torch.from_numpy(mu).float().to(device)
    data_sigma = torch.from_numpy(sig).float().to(device)
    print(f"data_mu = \n{data_mu.cpu().numpy()}, \ndata_sigma = \n{data_sigma.cpu().numpy()}")

    #load data saved in csv file
    df_filename = dataset_filepath + '.csv'
    df = pd.read_csv(df_filename)

    print(list(df.columns))
    print("Data loaded.")

    # Create PyTorch TensorDataset from numpy data
    d_trn = TensorDataset(torch.from_numpy(trn).float())
    d_val = TensorDataset(torch.from_numpy(val).float())

    # Default settings
    train = True
    file_number = None # file_number: the starting point of the epoch training loop.
    nb_flow = 1# nb_flow: the number of flow layers in the normalizing flow model. We are using one-step function computation for Monotonic Nomalizer.
    workers = 0 # worker: It tells the data loader instance how many sub-processes to use for data loading.
    pin_memory = False # pin_memory: setting pin_memory to 'True' can boost performance by reducing host to device (CPU to GPU) transfer time.
    cond_type = 'DAG' # cond_type: the type of conditioner to use ('DAG', 'Coupling', or 'Autoregressive').
    nb_step_dual = 50 # nb_step_dual: number of step between updating Acyclicity constraint and sparsity constraint, which is used in dual optimization.
    l1 = .5 # l1: Maximum weight for l1 regularization (Lasso) for the DagConditioner.
    gumble_T = .5 # gumble_T: Temperature of the gumble distribution.
    norm_type = 'monotonic' # norm_type: the type of normalizer to use ('affine' or 'monotonic').
    nb_steps = 50 # nb_steps: the number of steps to take for optimization (integration) in Normalizer.
    solver = "CC" # Clenshaw-Curtis quadrature optimization algorithm (integral solver) in the normalizer.

    # Create pytorch dataloaders for the respective datasets to process the data in batches
    print(f"Batch size = trn:{trn_batch_size:7d},  val:{val_batch_size:7d}")
    l_trn = DataLoader(d_trn,
                       batch_size=trn_batch_size,
                       num_workers=int(workers),
                       shuffle=True,  #
                       pin_memory=pin_memory,
                       drop_last=False)  # create train dataloader
    l_val = DataLoader(d_val,
                       batch_size=val_batch_size,
                       num_workers=int(workers),
                       shuffle=True,  # False,
                       pin_memory=pin_memory,
                       drop_last=False)  # create validation dataloader

    print(f"Number of samples = trn:{len(d_trn):7d},  val:{len(d_val):7d}")
    print(f"Number of batches = trn:{len(l_trn):7d},  val:{len(l_val):7d}")
    epoch_iters = len(l_trn) # Equals to 'Number of samples' (N) // 'Batch size' (B)

    print(f"Dataset_mean = {mu}")
    print(f"Dataset_sigma = {sig}")

    # Import modules from GNF for the monotonic normalizer/transformer and graphical conditioner
    dim = torch.from_numpy(trn).shape[1] # Retrieve the number of variables (number of columns) in your input data trn
    conditioner_type = cond_types[cond_type]
    conditioner_args = {"in_size": dim, "hidden": emb_net[:-1], "out_size": emb_net[-1]}
    if conditioner_type is DAGConditioner:
        conditioner_args['l1'] = l1
        conditioner_args['gumble_T'] = gumble_T
        conditioner_args['nb_epoch_update'] = nb_step_dual
        conditioner_args["hot_encoding"] = False # a process of converting categorical variables into a form that could be provided to machine learning algorithms to improve prediction.
        conditioner_args['A_prior'] = A.to(device)

    #print(f'{conditioner_args}')

    normalizer_type = norm_types[norm_type]
    if normalizer_type is MonotonicNormalizer:
        normalizer_args = {"integrand_net": int_net, "cond_size": emb_net[-1], "nb_steps": nb_steps,
                           "solver": solver,
                           "mu": data_mu, "sigma": data_sigma,
                           # standardize the input data, comment to learn cGNF on unstandardized data
                           # "cat_dims": None
                           "cat_dims": cat_dims  # categorical dimensions for Gaussian dequantization
                           }
    else:
        normalizer_args = {}
    #print(f'{normalizer_args}')

    if file_number is None:
        file_number = 0

    # Creating the cGNF model
    model = buildFCNormalizingFlow(nb_flow, conditioner_type, conditioner_args, normalizer_type, normalizer_args)
    print(f'{model.getConditioners()[0]}')
    print(f'{model.getNormalizers()[0]}')
    _best_valid_loss = np.inf  # initializing the variable '_best_valid_loss' with the value of positive infinity (np.inf).

    ## Initializing an instance of the AdamW optimizer in Pytorch
    # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # AdamW decouples the weight decay from the optimization steps (as done in Adam optimizer).
    # filter() ensures that the optimizer only tries to optimize the parameters that are intended to be updated, i.e., the ones that have their 'requires_grad attribute' set to 'True'.

    n_iters_val = (epoch_iters // val_freq)  # check validation loss after # epoch iteration

    model.to(device)  # move model to GPU(CPU)

    # Initialize counter for early stopping
    n_estop = 0
    # Initialize counter for total iterations in training
    n_iters = 0

    # Start the epcoh training, and go from epoch 'file_number + 1' to epoch 'file_number + 1 + nb_epoch'. The model will be updated a total of 'Number of epochs' (E) * 'Number of batches' (B) iterations.
    for epoch in range(file_number + 1, file_number + 1 + nb_epoch):

        ll_tot = 0 # Initialize total loss
        start = timer() # Measure time at the start of the epoch

        # Update constraints if the conditioner is a DAGConditioner
        if conditioner_type is DAGConditioner:
            with torch.no_grad(): #  indicate the block of code that follows will not require gradient (requires_grad = True) computation.
                for conditioner in model.getConditioners():
                    conditioner.constrainA(zero_threshold=0.) # Constrain A applies a constraint to the adjacency matrix (A) of the DAGConditioner. With the threshold is set to 0, any weights exactly equal to zero will be pruned, making the corresponding edges disappear from the graph.

        # Training loop
        if train:
            model.train() # Set the model to training mode, which is built in Pytorch
            if n_estop > nb_estop: # Early stopping if n_estop exceeds the specified early stopping threshold
                break

            # Loop over all batches of data in the training set
            for it, cur_x in enumerate(l_trn): # 'it' is the index or batch number, and 'cur_x' is the data in the current batch.
                n_iters += 1
                cur_x = cur_x[0].to(device) # Move the current batch of data to the device

                # If the normalizer is a MonotonicNormalizer, update the number of steps
                if normalizer_type is MonotonicNormalizer:
                    for normalizer in model.getNormalizers():
                        normalizer.nb_steps = nb_steps + torch.randint(0, 10, [1])[0].item() # the number of steps 'nb_steps' is added by a random integer between 0 and 10.

                # Forward pass through the model
                z, jac = model(cur_x)

                # Compute the loss
                loss = model.loss(z, jac)

                # If the loss is NaN or infinity, save the model and exit
                if math.isnan(loss.item()) or math.isinf(loss.abs().item()):
                    torch.save(model, path_save,'NANmodel.pt')
                    print("Error NAN in loss")
                    exit()

                # Add the loss to the total loss
                ll_tot += loss.detach()

                # Zero the gradients, perform backpropagation, and update the weights
                opt.zero_grad() # clears old gradients from the last step
                loss.backward(retain_graph=True) # Backward pass: compute the derivative of the loss w.r.t. the parameters using backpropagation.
                opt.step() # causes the optimizer to take a step based on the gradients of the parameters.

                # Calculate validation loss at specified intervals of the training
                if n_iters % n_iters_val == 0: #the model's validation loss is evaluated every 'n_iters_val' steps of training (the number of iteration is the integer multiple of 'n_iters_val').
                    print("------- iteration: {:d} --------". format(n_iters))
                    n_estop += n_iters_val / epoch_iters #n_estop is to track how many epochs have passed without improvement on the validation set. for n_estop to reach 50, the current model would have to go through  '50 % n_iters_val / epoch_iters' epochs without improving its validation loss.
                    print("Current n_estop:", n_estop)
                    # Valid loop
                    model.eval() # Set the model to evaluation mode
                    ll_val = 0. # Initialize validation loss

                    with torch.no_grad():
                        # Update the number of steps for the normalizer during validation
                        if normalizer_type is MonotonicNormalizer:
                            for normalizer in model.getNormalizers():
                                normalizer.nb_steps = nb_steps + 20
                        # Loop over all batches of data in the validation set
                        for iv, cur_x in enumerate(l_val): # 'iv' is the index or batch number, and 'cur_x' is the data in the current batch.
                            cur_x = cur_x[0].to(device) # Move the current batch of data to the device
                            z, jac = model(cur_x) # Forward pass through the model
                            ll = (model.z_log_density(z) + jac) # Compute the log-likelihood
                            ll_val += ll.mean().item() # Add the log-likelihood to the total validation loss

                        # Compute the average validation loss
                        ll_val /= iv + 1

                        end = timer()# Measure time at the end of the epoch
                        print(
                            "epoch: {:d} - batch: {:d} - Train loss: {:4f} - Current best validation loss: {:4f} - Valid log-likelihood: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                            format(epoch, it+1, ll_tot.item() / (it + 1), _best_valid_loss, ll_val, end - start))

                        if -ll_val < _best_valid_loss: # If the current model performs better than all previous model, n_estop will be reset to 0. Else, n_estop will not reset and will instead increment.
                            n_estop = 0 # It's incremented each time validation happens and reset to 0 each time a new best validation loss is found. If the value of n_estop exceeds the early stopping threshold nb_estop, the training is stopped early.
                            _best_valid_loss = -ll_val # replace the best validation loss with the current value
                            print("------- New best validation loss --------")
                            print("New best validation loss: {:4f} ".format(-ll_val))
                            print("Saving best model...")
                            torch.save(model, os.path.join(path_save,'_best_model.pt'))  # save the current best validation model.
                            torch.save(opt.state_dict(), os.path.join(path_save,'_best_optimizer.pt'))  # save the current best validation optimizer state.

                    torch.save(model, os.path.join(path_save,'model.pt'))  # save the current model at validation intervals.
                    torch.save(opt.state_dict(), os.path.join(path_save, 'ADAM.pt'))

            torch.save(model, os.path.join(path_save, 'model.pt'))  # save the current model before termination.
            torch.save(opt.state_dict(), os.path.join(path_save,'ADAM.pt'))

            model.train()  # switch back from eval to train mode
            ll_tot /= it + 1
            model.step(epoch, ll_tot)  # store/update model training loss
        else:
            ll_tot = 0.

    print(f'\n\nModel and logs saved in {path_save} folder.\n\n')

    return model, data  # return the trained cGNF model and the dataset object for further inference
