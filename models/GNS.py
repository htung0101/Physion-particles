import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

import ipdb
_st=ipdb.set_trace
### Dynamic Particle Interaction Networks

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
        return self.model(x)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        # print(x.size())
        return self.model(x)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        '''
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        '''
        if self.residual:
            x = self.relu(self.linear(x) + res)
        else:
            x = self.relu(self.linear(x))

        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x)



class GNS(nn.Module):
    def __init__(self, args, stat, phases_dict, residual=False, use_gpu=False):

        super(GNS, self).__init__()

        self.args = args

        state_dim = args.state_dim
        attr_dim = args.attr_dim
        relation_dim = args.relation_dim
        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect

        self.nf_effect = args.nf_effect

        self.stat = stat
        self.use_gpu = use_gpu
        self.residual = residual
        self.quat_offset = torch.FloatTensor([1., 0., 0., 0.])
        if use_gpu:
            self.quat_offset = self.quat_offset.cuda()

        self.n_stages = args.n_stages
        self.n_stages_types = args.n_stages_types
        self.dt = args.dt * args.training_fpt

        if use_gpu:

            # make sure std_v and std_p is ok
            for item in range(2):
                for idx in range(3):
                    if stat[item][idx, 1] == 0:
                        stat[item][idx, 1] = 1


            self.pi = Variable(torch.FloatTensor([np.pi])).cuda()
            self.dt = Variable(torch.FloatTensor([self.dt])).cuda()
            self.mean_v = Variable(torch.FloatTensor(stat[1][:, 0])).cuda() #velocity
            self.std_v = Variable(torch.FloatTensor(stat[1][:, 1])).cuda() # velocity
            self.mean_p = Variable(torch.FloatTensor(stat[0][:3, 0])).cuda() #position
            self.std_p = Variable(torch.FloatTensor(stat[0][:3, 1])).cuda() #position
        else:
            self.pi = Variable(torch.FloatTensor([np.pi]))
            self.dt = Variable(torch.FloatTensor(self.dt))
            self.mean_v = Variable(torch.FloatTensor(stat[1][:, 0]))
            self.std_v = Variable(torch.FloatTensor(stat[1][:, 1]))
            self.mean_p = Variable(torch.FloatTensor(stat[0][:3, 0]))
            self.std_p = Variable(torch.FloatTensor(stat[0][:3, 1]))

        # (1) particle attr (2) state
        self.particle_encoder = ParticleEncoder(attr_dim + state_dim * 2, nf_particle, nf_effect)

        # (1) sender attr (2) receiver attr (3) state receiver (4) state_diff (5) relation attr
        self.relation_encoder = RelationEncoder(
                2 * attr_dim + 2 * state_dim + 4 + relation_dim,
                nf_relation, nf_relation)

        # (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(nf_relation + 2 * nf_effect, nf_effect)

        # (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(2 * nf_effect, nf_effect, self.residual)

        # (1) set particle effect
        self.particle_predictor = ParticlePredictor(nf_effect, nf_effect, args.position_dim)

    def save_local(self, args, path_name):
        def foo(args):
            return locals()
        output = foo(args)
        output["pi"] = self.pi.cpu().numpy()
        output["dt"] = self.dt.cpu().numpy()
        output["mean_v"] = self.mean_v.cpu().numpy()
        output["std_v"] = self.std_v.cpu().numpy()
        output["mean_p"] = self.mean_p.cpu().numpy()
        output["std_p"] = self.std_p.cpu().numpy()

        with open(path_name, "wb") as f:
            pickle.dump(output, f)

    def load_local(self, path_name):
        with open(path_name, "rb") as f:
            output = pickle.load(f)
        if self.use_gpu:
            self.pi = Variable(torch.FloatTensor(output["pi"])).cuda()
            self.dt = Variable(torch.FloatTensor(output["dt"])).cuda()
            self.mean_v = Variable(torch.FloatTensor(output["mean_v"])).cuda() #velocity
            self.std_v = Variable(torch.FloatTensor(output["std_v"])).cuda() # velocity
            self.mean_p = Variable(torch.FloatTensor(output["mean_p"])).cuda() #position
            self.std_p = Variable(torch.FloatTensor(output["std_p"])).cuda() #position
        else:
            self.pi = Variable(torch.FloatTensor(output["pi"]))
            self.dt = Variable(torch.FloatTensor(output["dt"]))
            self.mean_v = Variable(torch.FloatTensor(output["mean_v"])) #velocity
            self.std_v = Variable(torch.FloatTensor(output["std_v"])) # velocity
            self.mean_p = Variable(torch.FloatTensor(output["mean_p"])) #position
            self.std_p = Variable(torch.FloatTensor(output["std_p"])) #position
        return vars(output["args"])



    def rotation_matrix_from_quaternion(self, params):
        # params dim - 4: w, x, y, z

        if self.use_gpu:
            one = Variable(torch.ones(1, 1)).cuda()
            zero = Variable(torch.zeros(1, 1)).cuda()
        else:
            one = Variable(torch.ones(1, 1))
            zero = Variable(torch.zeros(1, 1))

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / torch.norm(params)
        w, x, y, z = params[0].view(1, 1), params[1].view(1, 1), params[2].view(1, 1), params[3].view(1, 1)

        rot = torch.cat((
            torch.cat((one-y*y*2-z*z*2, x*y*2+z*w*2, x*z*2-y*w*2), 1),
            torch.cat((x*y*2-z*w*2, one-x*x*2-z*z*2, y*z*2+x*w*2), 1),
            torch.cat((x*z*2+y*w*2, y*z*2-x*w*2, one-x*x*2-y*y*2), 1)), 0)

        return rot

    def forward(self, attr, state, Rr, Rs, Ra, Rr_idxs, n_particles, node_r_idx, node_s_idx, pstep, rels_types,
                instance_idx, phases_dict, verbose=1):
        """
        attr: #nodes x attr_dim
        state: #nodes x state_dim


        """
        # calculate particle encoding
        if self.use_gpu:
            particle_effect = Variable(torch.zeros((attr.size(0), self.nf_effect)).cuda())
            pos_mask = Variable(torch.ones((1, state.size(1))).cuda())
            pos_mask[0, :3] = 0
        else:
            particle_effect = Variable(torch.zeros((attr.size(0), self.nf_effect)))
            pos_mask = Variable(torch.ones((1, state.size(1))))
            pos_mask[0, :3] = 0

        # add offset to center-of-mass for rigids to attr
        if self.use_gpu:
            offset = Variable(torch.zeros((attr.size(0), state.size(1))).cuda())
        else:
            offset = Variable(torch.zeros((attr.size(0), state.size(1))))

        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            if phases_dict['material'][i] == 'rigid':
                c = torch.mean(state[st:ed], dim=0)
                offset[st:ed] = state[st:ed] - c
        attr = torch.cat([attr, offset], 1)

        n_stage = len(Rr)
        assert(len(rels_types) == n_stage)

        s = 0
        if verbose:
            print("=== Stage", s, ":", args.n_stages_types[s])
        Rrp = Rr[s].t()
        Rsp = Rs[s].t()

        # receiver_attr, sender_attr
        attr_r = attr[node_r_idx[s]]
        attr_s = attr[node_s_idx[s]]
        attr_r_rel = Rrp.mm(attr_r)
        attr_s_rel = Rsp.mm(attr_s)

        # receiver_state, sender_state
        state_r = state[node_r_idx[s]]
        state_s = state[node_s_idx[s]]
        state_r_rel = Rrp.mm(state_r)
        state_s_rel = Rsp.mm(state_s)
        state_diff = state_r_rel - state_s_rel


        # particle encode
        if verbose:
            print('attr_r', attr_r.shape, 'state_r', state_r.shape)
        particle_encode = self.particle_encoder(torch.cat([attr_r, state_r * pos_mask], 1))

        # calculate relation encoding
        state_r_s_rel = torch.cat([state_r_rel[:,:3] - state_s_rel[:,:3], torch.norm(state_r_rel[:,:3] - state_s_rel[:,:3], p=2, keepdim=True, dim=1)], dim=1)

        n_relations = attr_s_rel.shape[0]

        max_relations_on_gpu = 150000
        if n_relations > max_relations_on_gpu:
            print("large n relations", n_relations)
        #     n_splits = int((n_relations - 1) /max_relations_on_gpu) + 1

        #     #if n_splits > 2:
        #     print("#relations too large:", n_relations, ", split the computation into",  n_splits)

        #     relation_encode_list = []

        #     for split_id in range(n_splits):
        #         start = split_id * max_relations_on_gpu
        #         end = (split_id + 1) * max_relations_on_gpu
        #         relation_encode1 = self.relation_encoder(
        #             torch.cat([attr_r_rel[start:end], attr_s_rel[start:end], state_r_s_rel[start:end], Ra[s][start:end]], 1))
        #         relation_encode_list.append(relation_encode1)
        #     relation_encode = torch.cat(relation_encode_list)
        # else:

        relation_encode = self.relation_encoder(
            torch.cat([attr_r_rel, attr_s_rel, state_r_s_rel, Ra[s]], 1))


        if verbose:
            print("relation encode:", relation_encode.size())

        for i in range(pstep[s]):
            if verbose:
                print("pstep", i)
                print("Receiver index range", np.min(node_r_idx[s]), np.max(node_r_idx[s]))
                print("Sender index range", np.min(node_s_idx[s]), np.max(node_s_idx[s]))

            effect_p_r = particle_effect[node_r_idx[s]]
            effect_p_s = particle_effect[node_s_idx[s]]

            receiver_effect = Rrp.mm(effect_p_r)
            sender_effect = Rsp.mm(effect_p_s)

            # print(i, "n_relations", receiver_effect.shape[0])
            #import ipdb; ipdb.set_trace()

            # calculate relation effect
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, receiver_effect, sender_effect], 1))
            if verbose:
                print("relation effect:", effect_rel.size())

            # calculate particle effect by aggregating relation effect
            effect_p_r_agg = Rr[s].mm(effect_rel)

            # calculate particle effect
            effect_p = self.particle_propagator(
                torch.cat([particle_encode, effect_p_r_agg], 1),
                res=effect_p_r)
            if verbose:
                print("particle effect:", effect_p.size())

            particle_effect[node_r_idx[s]] = effect_p

        # ex. fliudFall instance_idx[0, 189] means there is only one object state[0:190]
        # ex. boxBath [0, 64, 1024], instance=["cube", "fluid"], material=["rigid", "fluid"]
        # particle effect: 1032 x 200
        # ex. FluidShake: [0, 570], fluid
        normalized_velocities = self.particle_predictor(particle_effect)[:n_particles]




        if verbose:
            print("pred:", normalized_velocitiess.size())

        return normalized_velocities


