import os
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ipdb;
st=ipdb.set_trace

from models import DPINet, DPINet2, GNS, GNSRigid, GNSRigidH
from data import PhysicsFleXDataset, collate_fn

from utils import count_parameters, get_query_dir

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

data_root = get_query_dir("dpi_data_dir")
out_root=get_query_dir("out_dir")


parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--training_fpt', type=float, default=1)

parser.add_argument('--nf_relation', type=int, default=300)
parser.add_argument('--nf_particle', type=int, default=200)
parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--model_name', default='DPINet2')
parser.add_argument('--floor_cheat', type=int, default=0)
parser.add_argument('--env', default='')
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--statf', default="")
parser.add_argument('--noise_std', type=float, default='0')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gen_stat', type=int, default=0)

parser.add_argument('--subsample_particles', type=int, default=1)

parser.add_argument('--log_per_iter', type=int, default=1000)
parser.add_argument('--ckp_per_iter', type=int, default=10000)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--augment_worldcoord', type=int, default=0)

parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--n_instance', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--forward_times', type=int, default=2)

parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()

phases_dict = dict()

random.seed(args.seed)
torch.manual_seed(args.seed)
# preparing phases_dict
if args.env == "TDWdominoes":
    args.n_rollout = None# how many data
    # don't use, determined by data

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3
    args.dt = 0.01

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = -1
    args.time_step = 301 #??
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_stages_types = ["leaf-leaf", "leaf-root", "root-root", "root-leaf"]

    args.neighbor_radius = 0.08

    phases_dict = dict() # load from data
    #["root_num"] = [[]]
    #phases_dict["instance"] = ["fluid"]
    #phases_dict["material"] = ["fluid"]
    args.outf = args.outf.strip()

    args.outf = os.path.join(out_root,'dump/dump_TDWdominoes/' + args.outf)


else:
    raise AssertionError("Unsupported env")

writer = SummaryWriter(os.path.join(args.outf, "log"))
data_root = os.path.join(data_root, "train")
args.data_root = data_root
if "," in args.dataf:
    #list of folder
    args.dataf = [os.path.join(data_root, tmp.strip()) for tmp in args.dataf.split(",") if tmp != ""]

else:
    args.dataf = args.dataf.strip()
    if "/" in args.dataf:
        args.dataf = 'data/' + args.dataf
    else: # only prefix
        args.dataf = 'data/' + args.dataf + '_' + args.env
    os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)


# generate data
datasets = {phase: PhysicsFleXDataset(
    args, phase, phases_dict, args.verbose_data) for phase in ['train', 'valid']}

for phase in ['train', 'valid']:
    datasets[phase].load_data(args.env)

use_gpu = torch.cuda.is_available()
assert(use_gpu)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=args.batch_size,
    shuffle=True if x == 'train' else False,
    #num_workers=args.num_workers,
    collate_fn=collate_fn)
    for x in ['train', 'valid']}

# define propagation network
if args.env == "TDWdominoes":
    if args.model_name == "DPINet":
        model = DPINet(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
    if args.model_name == "DPINet2":
        """
        original DPI, but don't apply different fc for different objects?
        originla dpi only has one object, so they are trying to apply different fc for
        different type of relation.
        But I have several objects, do many relations are actually the same
        to do: add relationship type
        """
        model = DPINet2(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
    elif args.model_name == "GNS":
        """
        deep mind model, hierarchy, use only relative information
        """
        args.pstep = 10
        args.n_stages = 1
        args.noise_std = 3e-4
        args.n_stages_types = ["leaf-leaf"]

        model = GNS(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)

    elif args.model_name == "GNSRigid":
        """
        deep mind model, hierarchy, use only relative information
        """
        args.pstep = 10
        args.n_stages = 1
        args.noise_std = 3e-4
        args.n_stages_types = ["leaf-leaf"]

        model = GNSRigid(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
    elif args.model_name == "GNSRigidH":
        """
        DPI2, use only relative information
        """
        args.noise_std = 3e-4

        model = GNSRigidH(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
    else:
        raise ValueError(f"no such model {args.model_name} for env {args.env}")
else:
    model = DPINet(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
if use_gpu:
    model = model.cuda()
print("Number of parameters: %d" % count_parameters(model))


opt_mode = "dpi"

if opt_mode == "dpi":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
elif opt_mode == "gns":
    min_lr = 1e-6
    lr = tf.train.exponential_decay(optimizer,
                                    decay_steps=int(5e6),
                                    decay_rate=0.1) + min_lr
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr - min_lr)

if args.resume_epoch > 0 or args.resume_iter > 0:
    # load local parameters
    args_load = model.load_local(os.path.join(args.outf, "args_stat.pkl"))
    args_current = vars(args)

    exempt_list = ["dataf", "lr", "num_workers", "resume_epoch", "resume_iter"]

    for key in args_load:
        if key in exempt_list:
            continue

        assert(args_load[key] == args_current[key]), f"{key} is mismatched in loaded args and current args: {args_load[key]} vs {args_current[key]}"

    # check args_load
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
    print("Loading saved ckp from %s" % model_path)
    #torch.save(model.state_dict(), model_path)
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # schedular.load_state_dict(checkpoint['scheduler_state_dict'])

    checkpoint = torch.load(model_path)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path))

# criterion
criterionMSE = nn.MSELoss()

# optimizer

optimizer.zero_grad()
if use_gpu:
    model = model.cuda()
    #criterionMSE = criterionMSE.cuda()

# save args, stat
model.save_local(args, os.path.join(args.outf, "args_stat.pkl"))

st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf
train_iter = 0
current_loss = 0

max_nparticles = 0
for epoch in range(st_epoch, args.n_epoch):

    phases = ['train', 'valid'] if args.eval == 0 else ['valid']
    for phase in phases:
        import time

        model.train(phase=='train')
        previous_run_time = time.time()
        start_time = time.time()

        losses = 0.
        for i, data in enumerate(dataloaders[phase]):

            #start_time = time.time()
            #print("previous run time", start_time - previous_run_time)
            attr, state, rels, n_particles, n_shapes, instance_idx, label, phases_dict_current= data

            if n_particles > max_nparticles:
                max_nparticles = n_particles

            Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]

            Rr, Rs, Rr_idxs = [], [], []
            for j in range(len(rels[0])):
                Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]

                Rr_idxs.append(Rr_idx)
                Rr.append(torch.sparse.FloatTensor(
                    Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))

                Rs.append(torch.sparse.FloatTensor(
                    Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

            data = [attr, state, Rr, Rs, Ra, Rr_idxs, label]

            #print("data prep:", time.time() - start_time)

            with torch.set_grad_enabled(phase=='train'):
                if use_gpu:
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = Variable(data[d][t].cuda())
                        else:
                            data[d] = Variable(data[d].cuda())
                else:
                    for d in range(len(data)):
                        if type(data[d]) == list:
                            for t in range(len(data[d])):
                                data[d][t] = Variable(data[d][t])
                        else:
                            data[d] = Variable(data[d])

                attr, state, Rr, Rs, Ra, Rr_idxs, label = data

                # st_time = time.time()
                predicted = model(
                    attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                    node_r_idx, node_s_idx, pstep, rels_types,
                    instance_idx, phases_dict_current, args.verbose_model)

            loss = criterionMSE(predicted, label) / args.forward_times
            current_loss = np.sqrt(loss.item() * args.forward_times)
            #loss =  F.l1_loss(predicted, label)
            losses += np.sqrt(loss.item())

            if phase == 'train':
                train_iter += 1
                loss.backward()
                if i % args.forward_times == 0 and i!=0:
                    # update parameters every args.forward_times
                    optimizer.step()
                    optimizer.zero_grad()
            #print("backprop:", time.time() - start_time)
            if i % args.log_per_iter == 0:
                n_relations = 0
                for j in range(len(Ra)):
                    n_relations += Ra[j].size(0)
                print('%s %s [%d/%d][%d/%d] n_relations: %d, Loss: %.6f, Agg: %.6f' %
                      (phase, args.outf, epoch, args.n_epoch, i, len(dataloaders[phase]),
                       n_relations, current_loss, losses / (i + 1)))
                print("total time:", time.time() - start_time)
                start_time = time.time()


                lr = get_lr(optimizer)
                if phase == "train":
                    writer.add_scalar(f'lr', lr, train_iter)
                writer.add_histogram(f'{phase}/label_x', label[:,0], train_iter)
                writer.add_histogram(f'{phase}/label_y', label[:,1], train_iter)
                writer.add_histogram(f'{phase}/label_z', label[:,2], train_iter)
                writer.add_histogram(f'{phase}/predicted_x', predicted[:,0], train_iter)
                writer.add_histogram(f'{phase}/predicted_y', predicted[:,1], train_iter)
                writer.add_histogram(f'{phase}/predicted_z', predicted[:,2], train_iter)
                writer.add_scalar(f'{phase}/loss', current_loss, train_iter)
            previous_run_time = time.time()

            if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                            '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

        print("total time:", time.time() - previous_run_time)
        losses /= len(dataloaders[phase])
        print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
              (phase, epoch, args.n_epoch, losses, best_valid_loss))
        if phase == 'valid':
            scheduler.step(losses)
            if(losses < best_valid_loss):
                best_valid_loss = losses
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                            '%s/net_best.pth' % (args.outf))

