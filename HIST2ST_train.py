import os
import torch
import random
import argparse
import pickle as pk
import pytorch_lightning as pl
from utils import *
from HIST2ST import *
from predict import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='the id of gpu.')
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=12000, help='random seed.')
parser.add_argument('--epochs', type=int, default=350, help='number of epochs.')
parser.add_argument('--name', type=str, default='hist2ST', help='prefix name.')
parser.add_argument('--data', type=str, default='her2st', help='dataset name:{"her2st","cscc"}.')
parser.add_argument('--logger', type=str, default='../logs/my_logs', help='logger path.')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout.')

parser.add_argument('--bake', type=int, default=5, help='the number of augmented images.')
parser.add_argument('--lamb', type=float, default=0.5, help='the loss coef of self-distillation.')


parser.add_argument('--nb', type=str, default='F', help='zinb or nb loss.')
parser.add_argument('--zinb', type=float, default=0.25, help='the loss coef of zinb.')

parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')
parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN .')
parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.')

parser.add_argument('--tag', type=str, default='5-7-2-8-4-16-32', 
                    help='hyper params: kernel-patch-depth1-depth2-depth3-heads-channel,'
                         'depth1-depth2-depth3 are the depth of Convmixer, Multi-head layer in Transformer, and GNN, respectively'
                         'patch is the value of kernel_size and stride in the path embedding layer of Convmixer'
                         'kernel is the kernel_size in the depthwise of Convmixer module'
                         'heads are the number of attention heads in the Multi-head layer'
                         'channel is the value of the input and output channel of depthwise and pointwise. ')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),args.tag.split('-'))

trainset = pk_load(args.fold,'train',False,args.data,neighs=args.neighbor, prune=args.prune)
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
testset = pk_load(args.fold,'test',False,args.data,neighs=args.neighbor, prune=args.prune)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
label=None
if args.fold in [5,11,17,23,26,30] and args.data=='her2st':
    label=testset.label[testset.names[0]]

genes=785
if args.data=='cscc':
    args.name+='_cscc'
    genes=171

log_name=''
if args.zinb>0:
    if args.nb=='T':
        args.name+='_nb'
    else:
        args.name+='_zinb'
    log_name+=f'-{args.zinb}'
if args.bake>0:
    args.name+='_bake'
    log_name+=f'-{args.bake}-{args.lamb}'
log_name=f'{args.fold}-{args.name}-{args.tag}'+log_name+f'-{args.policy}-{args.neighbor}'
logger = TensorBoardLogger(
    args.logger, 
    name=log_name
)
print(log_name)

model = Hist2ST(
    depth1=depth1, depth2=depth2, depth3=depth3,
    n_genes=genes, learning_rate=args.lr, label=label, 
    kernel_size=kernel, patch_size=patch,
    heads=heads, channel=channel, dropout=args.dropout,
    zinb=args.zinb, nb=args.nb=='T',
    bake=args.bake, lamb=args.lamb, 
    policy=args.policy, 
)
trainer = pl.Trainer(
    gpus=[args.gpu], max_epochs=args.epochs,
    logger=logger,check_val_every_n_epoch=2,
)

trainer.fit(model, train_loader, test_loader)
torch.save(model.state_dict(),f"./model/{args.fold}-Hist2ST{'_cscc' if args.data=='cscc' else ''}.ckpt")
# model.load_state_dict(torch.load(f"./model/{args.fold}-Hist2ST{'_cscc' if args.data=='cscc' else ''}.ckpt"),)
pred, gt = test(model, test_loader,'cuda')
R=get_R(pred,gt)[0]
print('Pearson Correlation:',np.nanmean(R))
clus,ARI=cluster(pred,label)
print('ARI:',ARI)
