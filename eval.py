import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data import load_data, load_data_dominoes, prepare_input, normalize, denormalize, recalculate_velocities, \
                 correct_bad_chair, remove_large_obstacles, subsample_particles_on_large_objects
from models import DPINet, DPINet2, GNS, GNSRigid, GNSRigidH
from utils import mkdir, get_query_dir
from utils_geom import calc_rigid_transform
import ipdb
_st = ipdb.set_trace

assert(torch.cuda.is_available())
data_root = get_query_dir("dpi_data_dir")
label_source_root = get_query_dir("dpi_data_dir")
model_root=get_query_dir("out_dir")
out_root = os.path.join(model_root, "eval")
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--subsample', type=int, default=3000)
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--training_fpt', type=float, default=1)

parser.add_argument('--nf_relation', type=int, default=300)
parser.add_argument('--nf_particle', type=int, default=200)
parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--modelf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--mode', default='test')
parser.add_argument('--statf', default="")
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--test_training_data_processing', type=int, default=0)
parser.add_argument('--ransac_on_pred', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--model_name', default='DPINet2')

parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--n_instances', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)
parser.add_argument('--augment_worldcoord', type=int, default=0)
parser.add_argument('--floor_cheat', type=int, default=0)
# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

#visualization
parser.add_argument('--saveavi', type=int, default=0)
parser.add_argument('--save_pred', type=int, default=1)

args = parser.parse_args()

phases_dict = dict()

if args.env == "TDWdominoes":
    args.n_rollout = 2# how many data
    data_names = ['positions', 'velocities']
    args.time_step = 200
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3
    args.dt = 0.01

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = -1
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_stages_types = ["leaf-leaf", "leaf-root", "root-root", "root-leaf"]

    args.neighbor_radius = 0.08
    args.gen_data = False

    phases_dict = dict() # load from data
    #["root_num"] = [[]]

    model_name = copy.deepcopy(args.modelf)
    args.modelf = 'dump/dump_TDWdominoes/' + args.modelf

    args.modelf = os.path.join(model_root, args.modelf)
else:
    raise AssertionError("Unsupported env")


evalf_root = os.path.join(out_root, args.evalf + '_' + args.env, model_name)
mkdir(os.path.join(out_root, args.evalf + '_' + args.env))
mkdir(evalf_root)

mode = args.mode

data_root_ori = data_root
scenario = args.dataf
args.data_root = data_root

prefix = args.dataf
args.dataf = os.path.join(data_root, mode, args.dataf)

stat = [np.zeros((3,3)), np.zeros((3,3))]

if args.statf:
    stat_path = os.path.join(data_root_ori, args.statf)
    print("Loading stored stat from %s" % stat_path)
    stat = load_data(data_names[:2], stat_path)

    for i in range(len(stat)):
        stat[i] = stat[i][-args.position_dim:, :]
        # print(data_names[i], stat[i].shape)

use_gpu = torch.cuda.is_available()


if args.model_name == "DPINet2":
    """
    original DPI, but don't apply different fc for different objects?
    originla dpi only has one object, so they are trying to apply different fc for
    different type of relation.
    But I have several objects, do many relations are actually the same
    to do: add relationship type
    """
    model = DPINet2(args, stat, phases_dict, residual=True, use_gpu=use_gpu)
elif args.model_name == "GNS":
    """
    deep mind model, hierarchy, use only relative information
    """
    args.pstep = 10
    args.n_stages = 1
    args.noise_std = 3e-4
    args.n_stages_types = ["leaf-leaf"]

    model = GNS(args, stat, phases_dict, residual=True, use_gpu=use_gpu)

elif args.model_name == "GNSRigid":
    """
    deep mind model, hierarchy, use only relative information
    """
    args.pstep = 10
    args.n_stages = 1
    args.noise_std = 3e-4
    args.n_stages_types = ["leaf-leaf"]

    model = GNSRigid(args, stat, phases_dict, residual=True, use_gpu=use_gpu)
elif args.model_name == "GNSRigidH":
    """
    deep mind model, hierarchy, use only relative information
    """
    args.noise_std = 3e-4

    model = GNSRigidH(args, stat, phases_dict, residual=True, use_gpu=use_gpu)
else:
    raise ValueError(f"no such model {args.model_name} for env {args.env}" )

if args.epoch == 0 and args.iter == 0:
    model_file = os.path.join(args.modelf, 'net_best.pth')
else:
    model_file = os.path.join(args.modelf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

# check args file
args_load = model.load_local(os.path.join(args.modelf, 'args_stat.pkl'))
args_current = vars(args)
exempt_list = ["dataf", "lr", "n_rollout", "time_step", "eval", "data_root"]
for key in args_load:
    if key in exempt_list or key not in args_current:
        continue
    assert(args_load[key] == args_current[key]), f"{key} is mismatched in loaded args and current args: {args_load[key]} vs {args_current[key]}"

print("Loading network from %s" % model_file)
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint['model_state_dict'])

#model.load_state_dict(torch.load(model_file))
model.eval()

criterionMSE = nn.MSELoss()

if use_gpu:
    model.cuda()



# list all the args
# only evaluate on human data now

infos = np.arange(100)
data_name = args.dataf.split("/")[-1]


if args.save_pred:
    #if not args.gt_only:
    predf = open(os.path.join(evalf_root,  mode + "-"+ scenario +".txt"), "a")
    pred_gif_folder = os.path.join(evalf_root,  mode + "-"+ scenario)

    if args.ransac_on_pred:
        predf = open(os.path.join(evalf_root, "ransacOnPred-" + mode + "-"+ scenario + ".txt"), "a")
        pred_gif_folder = os.path.join(evalf_root,  "ransacOnPred-" + mode + "-"+ scenario)
    mkdir(pred_gif_folder)

accs = []
recs = []

mode = args.mode
dt = args.training_fpt * args.dt



gt_preds = []
#import ipdb; ipdb.set_trace()

label_file = os.path.join(label_source_root, mode, "labels", f"{scenario}.txt")
gt_labels = []
with open(label_file, "r") as f:
    for line in f:
        trial_name, label = line.strip().split(",")
        gt_labels.append((trial_name[:-5], (label == "True")))
gt_labels = gt_labels

for trial_id, trial_cxt in enumerate(gt_labels):
    #if trial_id < 200:
    #    continue
    trial_name, label_gt = trial_cxt

    trial_name = os.path.join(args.dataf, trial_name)
    #for idx in range(len(infos)):
    gt_node_rs_idxs = []


    if scenario == "Support":
        max_timestep = 205
    elif scenario == "Link":
        max_timestep = 140
    elif scenario == "Contain":
        max_timestep = 125
    elif scenario in ["Collide", "Drape"]:
        max_timestep = 55
    else:
        max_timestep = 105

    args.time_step = len([file for file in os.listdir(trial_name) if file.endswith(".h5")])

    print("Rollout %d / %d" % (trial_id, len(gt_labels)))
    #des_dir = os.path.join(args.evalf, 'rollout_%d' % idx)
    #os.system('mkdir -p ' + des_dir)

    #trying to identify the length
    #import ipdb; ipdb.set_trace()

    timesteps  = [t for t in range(0, args.time_step - int(args.training_fpt), int(args.training_fpt))]
    # ground truth
    #assert(max_timestep > len(timesteps)), str(max_timestep) + "," + str(len(timesteps))
    total_nframes = max_timestep #len(timesteps)

    if args.env == "TDWdominoes":
        pkl_path = os.path.join(trial_name, 'phases_dict.pkl')
        with open(pkl_path, "rb") as f:
            phases_dict = pickle.load(f)
    phases_dict["trial_dir"] = trial_name

    # get red_id and yellow_id
    if scenario in ["Dominoes", "Collide", "Drop"]:
        red_id = 1
        yellow_id = 0
    elif scenario in ["Drape"]:

        instance_idx = phases_dict["instance_idx"]
        yellow_id = 0
        red_id = len(instance_idx) - 1 -1
    elif scenario in ["Roll"]:
        yellow_id = 0
        if "ramp" in trial_name:
            red_id = 2
        else:
            red_id = 1
    else:
        if "red_id" not in phases_dict:
            print(arg_name, trial_id_name)
            import ipdb; ipdb.set_trace()
        red_id = phases_dict["red_id"]
        yellow_id = phases_dict["yellow_id"]



    if args.test_training_data_processing:
        is_bad_chair = correct_bad_chair(phases_dict)
        # remove obstacles that are too big
        is_remove_obstacles = remove_large_obstacles(phases_dict)

        # downsample large object
        is_subsample = subsample_particles_on_large_objects(phases_dict, limit=args.subsample)
        #is_subsample = True

        if not is_bad_chair and not is_remove_obstacles and not is_subsample:
            pass
        else:
            print("is_bad_chair", is_bad_chair, "is_remove_obstacles", is_remove_obstacles, "is_subsample", is_subsample)
            print("trial_name", trial_name)
    else:
        is_bad_chair = correct_bad_chair(phases_dict)
        # remove obstacles that are too big
        is_remove_obstacles = remove_large_obstacles(phases_dict)

        # downsample large object
        is_subsample = subsample_particles_on_large_objects(phases_dict, limit=args.subsample)
    print(phases_dict["n_particles"])
    # observation
    #collect info for gt
    pred_is_positive_trial = False
    start_timestep = 45#15
    start_id = 15 #5
    for current_fid, step in enumerate(timesteps[:start_id]):


        data_path = os.path.join(trial_name, str(step) + '.h5')
        data_nxt_path = os.path.join(trial_name, str(step + int(args.training_fpt)) + '.h5')

        if args.env == "TDWdominoes":

            data = load_data_dominoes(data_names, data_path, phases_dict)

            data_nxt = load_data_dominoes(data_names, data_nxt_path, phases_dict)

            data_prev_path = os.path.join(trial_name, str(max(0, step - int(args.training_fpt))) + '.h5')
            data_prev = load_data_dominoes(data_names, data_prev_path, phases_dict)

            _, data, data_nxt = recalculate_velocities([data_prev, data, data_nxt], dt, data_names)

        else:
            data = load_data(data_names, data_path)
            data_nxt = load_data(data_names, data_nxt_path)


        attr, state, rels, n_particles, n_shapes, instance_idx = \
                prepare_input(data, stat, args, phases_dict, args.verbose_data)

        Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
        gt_node_rs_idxs.append(np.stack([rels[0][0][0], rels[1][0][0]], axis=1))

        velocities_nxt = data_nxt[1]

        ### instance idx # for visualization
        #   instance_idx (n_instance + 1): start idx of instance
        if step == 0:
            if args.env == "TDWdominoes":
                positions, velocities = data
                clusters = phases_dict["clusters"]
                n_shapes = 0
            else:
                raise AssertionError("Unsupported env")

            count_nodes = positions.shape[0]
            n_particles = count_nodes - n_shapes
            print("n_particles", n_particles)
            print("n_shapes", n_shapes)

            p_gt = np.zeros((total_nframes, n_particles + n_shapes, args.position_dim))
            s_gt = np.zeros((total_nframes, n_shapes, args.shape_state_dim))
            v_nxt_gt = np.zeros((total_nframes, n_particles + n_shapes, args.position_dim))

            p_pred = np.zeros((total_nframes, n_particles + n_shapes, args.position_dim))

        p_gt[current_fid] = positions[:, -args.position_dim:]
        v_nxt_gt[current_fid] = velocities_nxt[:, -args.position_dim:]

        # print(step, np.sum(np.abs(v_nxt_gt[step, :args.n_particles])))

        import scipy
        spacing = 0.05
        positions = data[0]

        st, ed = instance_idx[red_id], instance_idx[red_id + 1]
        red_pts = positions[st:ed]

        st2, ed2 = instance_idx[yellow_id], instance_idx[yellow_id + 1]
        yellow_pts = positions[st2:ed2]

        sim_mat = scipy.spatial.distance_matrix(yellow_pts, red_pts, p=2)
        min_dist= np.min(sim_mat)

        if scenario == "Drape":
            thres = 0.1222556027835 * 0.05/0.035
        elif scenario == "Contain":
            thres = spacing * 1.0
        elif scenario == "Drop":
            thres = spacing * 1.0
        else:
            thres = spacing * 1.5
        pred_target_contacting_zone = min_dist < thres
        if pred_target_contacting_zone:
            pred_is_positive_trial = True
            break

        positions = positions + velocities_nxt * dt

    print("finish observation 15 frames, prediction:", pred_is_positive_trial)


    # model rollout
    start_timestep = 45#15
    start_id = 15 #5
    data_path = os.path.join(trial_name, f'{start_timestep}.h5')
    if args.env == "TDWdominoes":
        data = load_data_dominoes(data_names, data_path, phases_dict)
        data_path_prev = os.path.join(trial_name, f'{int(start_timestep - args.training_fpt)}.h5')
        data_prev = load_data_dominoes(data_names, data_path_prev, phases_dict)
        _, data = recalculate_velocities([data_prev, data], dt, data_names)

    else:
        data = load_data(data_names, data_path)

    #timesteps = timesteps[start_id:]
    #total_nframes = len(timesteps)

    #import ipdb; ipdb.set_trace()
    for current_fid in range(total_nframes - start_id):
        if pred_is_positive_trial:
            break
        if current_fid % 10 == 0:
            print("Step %d / %d" % (current_fid + start_id, total_nframes))

        p_pred[start_id + current_fid] = data[0]

        attr, state, rels, n_particles, n_shapes, instance_idx = \
                prepare_input(data, stat, args, phases_dict, args.verbose_data)



        Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]

        #node_rs_idxs.append(np.stack([rels[0][0][0], rels[1][0][0]], axis=1))


        Rr, Rs, Rr_idxs = [], [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
            Rr_idxs.append(Rr_idx)
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        buf = [attr, state, Rr, Rs, Ra, Rr_idxs]

        with torch.set_grad_enabled(False):
            if use_gpu:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t].cuda())
                    else:
                        buf[d] = Variable(buf[d].cuda())
            else:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t])
                    else:
                        buf[d] = Variable(buf[d])

            attr, state, Rr, Rs, Ra, Rr_idxs = buf
            # print('Time prepare input', time.time() - st_time)

            # st_time = time.time()
            vels = model(
                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                node_r_idx, node_s_idx, pstep, rels_types,
                instance_idx, phases_dict, args.verbose_model)
            # print('Time forward', time.time() - st_time)

            # print(vels)

            if args.debug:
                data_nxt_path = os.path.join(trial_name, str(step + args.training_fpt) + '.h5')
                data_nxt = normalize(load_data(data_names, data_nxt_path), stat)
                label = Variable(torch.FloatTensor(data_nxt[1][:n_particles]).cuda())
                # print(label)
                loss = np.sqrt(criterionMSE(vels, label).item())
                print(loss)

        vels = denormalize([vels.data.cpu().numpy()], [stat[1]])[0]

        if args.ransac_on_pred:
            positions_prev = data[0]
            predicted_positions = data[0] + vels * dt
            for obj_id in range(len(instance_idx) - 1):
                st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
                if phases_dict['material'][obj_id] == 'rigid':

                    pos_prev = positions_prev[st:ed]
                    pos_pred = predicted_positions[st:ed]

                    R, T = calc_rigid_transform(pos_prev, pos_pred)
                    refined_pos = (np.dot(R, pos_prev.T) + T).T

                    predicted_positions[st:ed, :] = refined_pos


            data[0] = predicted_positions
            data[1] = (predicted_positions - positions_prev)/dt


        else:
            data[0] = data[0] + vels * dt
            data[1][:, :args.position_dim] = vels

        if args.debug:
            data[0] = p_gt[current_fid + 1].copy()
            data[1][:, :args.position_dim] = v_nxt_gt[current_fid]

        import scipy
        spacing = 0.05
        positions = data[0]

        st, ed = instance_idx[red_id], instance_idx[red_id + 1]
        red_pts = positions[st:ed]

        st2, ed2 = instance_idx[yellow_id], instance_idx[yellow_id + 1]
        yellow_pts = positions[st2:ed2]

        sim_mat = scipy.spatial.distance_matrix(yellow_pts, red_pts, p=2)
        min_dist= np.min(sim_mat)

        if "Drape" in scenario:
            thres = 0.1222556027835 * 0.05/0.035
        elif "Contain" in scenario:
            thres = spacing * 1.0
        elif "Drop" in scenario:
            thres = spacing * 1.0
        else:
            thres = spacing * 1.5
        pred_target_contacting_zone = min_dist < thres
        if pred_target_contacting_zone:
            pred_is_positive_trial = True
            break


    acc = int(label_gt == pred_is_positive_trial)

    accs.append(acc)
    #predf.writeline(",".join([str(infos[idx]), gt_target_contacting_zone, pred_target_contacting_zone, min_dist2, min_dist1]))
    print(args.dataf)
    print(args.modelf, args.ransac_on_pred)
    print("gt vs pred:", label_gt, pred_is_positive_trial)
    print("accuracy:", np.mean(accs))

    predf.write(",".join([trial_name, str(acc), str(label_gt), str(pred_is_positive_trial)]) + "\n")

