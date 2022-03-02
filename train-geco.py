import argparse
import importlib
import torch
import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from models import get_model
from losses import get_loss
from logs import get_logger
from utils_mnist import summarize_batch

from utils import sample_from_normal

# configuration for 'mnist' experienment
# --experiment 'mnist' --num_epochs 100 --context_dim 64 --num_stochastic_layers 3 --z_dim 2 --x_dim 2 --h_dim 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Arguments for training procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='synthetic',
    help='options that tells what experiment to replicate: synthetic|mnist|omniglot|youtube')

parser.add_argument('--nll', type=str, default='gaussian', help='type of loglikelihood')

# Dataloaders options
parser.add_argument('--train_num_datasets_per_distr', type=int, default=2500,
    help='number of training datasets per distribution for synthetic experiment')

parser.add_argument('--test_num_datasets_per_distr', type=int, default=500,
    help='number of test datasets per distribution for synthetic experiment')

parser.add_argument('--train_num_persons', type=int, default=1395,
    help='number of persons in the training datasets for youtube experiment')

parser.add_argument('--test_num_persons', type=int, default=100,
    help='number of persons in the testing datasets for youtube experiment')

parser.add_argument('--num_data_per_dataset', type=int, default=200,
    help='number of samples per dataset')

parser.add_argument('--batch_size', type=int, default=16, help='size of batch') #16

# Path for data directory if using the youtube experiment
parser.add_argument('--data_dir', type=str, default=None, help='location of sampled youtube data')

parser.add_argument('--test_mnist', action='store_true', help='whether to test on mnist')

# Optimization options
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')

# beta1 is the exponential decay rate for the first moment estimates (e.g. 0.9), used in the Adam optimizer.
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for optimizer')

parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')

# Architecture options
parser.add_argument('--context_dim', type=int, default=3, help='context dimension')

parser.add_argument('--masked', action='store_true',
    help='whether to use masking during training')

parser.add_argument('--type_prior', type=str, default='standard',
    help='either use standard gaussian prior or prior conditioned on labels')

parser.add_argument('--num_stochastic_layers', type=int, default=1,
    help='number of stochastic layers')

parser.add_argument('--z_dim', type=int, default=32,
    help='dimension of latent variables')

parser.add_argument('--x_dim', type=int, default=1, help='dimension of input')

parser.add_argument('--h_dim', type=int, default=1, help='dimension of h after shared encoder')

# Logging options
parser.add_argument('--tensorboard', action='store_true', help='whether to use tensorboard')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--save_dir', type=str, default='model_params')
parser.add_argument('--save_freq', type=int, default=5)  # 20

opts = parser.parse_args()

# If using youtube dataset, check that a data directory is specified
if opts.experiment == 'youtube' and opts.data_dir is None:
    exit("Must specify a directory for the youtube dataset")

#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))

train_dataset = dataset_module.get_dataset(opts, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=6)

test_dataset = dataset_module.get_dataset(opts, split='val')
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=6)
test_batch = next(iter(test_dataloader))

# Initialize the Neural Statistician model in models.py
model = get_model(opts).to(device)

loss_dict = get_loss(opts)
logger = get_logger(opts)
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

alpha = 0.99
lbd_step = 100
iter_num = 0
pretrain = 1

lambd = torch.FloatTensor([1]).to(device) 

def RE(x, mu, tol):
    return torch.sum(torch.pow(mu - x, 2), dim = 1) - tol**2

train_hist = {'loss':[], 'reconstr':[], 'KL':[]}
test_hist = {'loss':[], 'reconstr':[], 'KL':[]}
lambd_hist = []

for epoch in tqdm.tqdm(range(300)):
    model.train()
    
    train_hist['loss'].append(0)
    train_hist['reconstr'].append(0)
    train_hist['KL'].append(0)


    for data_dict in train_dataloader:
        data = data_dict['datasets'].to(device)

        optimizer.zero_grad()
        output_dict = model.forward(data, train=True)

        reconstruction_mu = output_dict['means_x']
        reconstruction_logsigma = output_dict['logvars_x']

        losses = {}
        for key in loss_dict:
            losses[key] = loss_dict[key].forward(output_dict)

        constraint = torch.mean(RE(data, reconstruction_mu.view_as(data), 0.01))

        KL_div = losses['KL']
        loss = KL_div + lambd*constraint + losses['NLL']

        # Can weigh the contribution from each term
        #losses['sum'] = (1 + alpha)*losses['NLL'] + losses['KL']/(1 + alpha)

        # Compute gradients, and take step backward
        #losses['sum'].backward()
        loss.backward()



        for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)

        optimizer.step()

        with torch.no_grad():
                if epoch == 0 and iter_num == 0:
                    constrain_ma = constraint
                else:
                    constrain_ma = alpha * constrain_ma.detach_() + (1 - alpha) * constraint
                if iter_num % lbd_step == 0 and epoch > pretrain:
#                     print(torch.exp(constrain_ma), lambd)
                    lambd *= torch.clamp(torch.exp(constrain_ma), 0.9, 1.1)

        train_hist['loss'][-1] += loss.data.cpu().numpy()[0]/len(train_dataloader)
        train_hist['reconstr'][-1] += constraint.data.cpu().numpy()/len(train_dataloader)
        train_hist['KL'][-1] += KL_div.data.cpu().numpy()/len(train_dataloader)
        iter_num += 1
        
        # Save model outputs and losses from the training.
        logger.log_data(output_dict, losses)


        logger.log_image(output_dict, 'train')

        lambd_hist.append(lambd.data.cpu().numpy()[0])

        model.train(False)
        test_hist['loss'].append(0)
        test_hist['reconstr'].append(0)
        test_hist['KL'].append(0)
    
    with torch.no_grad():
        model.eval()
        
        for data_dict in test_dataloader:

            data = data_dict['datasets'].to(device)

            output_dict = model.sample_conditional(data, num_samples_per_dataset=5)
            
            #losses = {}
            #for key in loss_dict:
            #    losses[key] = loss_dict[key].forward(output_dict)
            
            losses = {'NLL': loss_dict['NLL'].forward(output_dict)}
            

            reconstruction_mu = output_dict['means_x']
            reconstruction_logsigma = output_dict['logvars_x']
            
            #constraint = torch.mean(RE(data, reconstruction_mu.view_as(data), 0.01))
            #KL_div = losses['KL']
            #loss = KL_div + torch.mm(lambd,constraint)

            #test_hist['loss'][-1] += loss.data.cpu().numpy()[0]/len(test_dataloader)
            #test_hist['reconstr'][-1] += constraint.data.cpu().numpy()/len(test_dataloader)
            #test_hist['KL'][-1] += KL_div.data.cpu().numpy()/len(test_dataloader)

            logger.log_data(output_dict, losses, split='test')




        logger.log_image(output_dict, 'test')


    if epoch % 100 == 0:
        logger.save_model(model, str(epoch))


    logger.save_model(model, 'last')
    #draw_hist(train_hist, valid_hist)
