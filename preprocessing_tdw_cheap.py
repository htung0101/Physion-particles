# preprocessing
import h5py
import copy
import os
import trimesh
import pickle
import numpy as np
import random
import argparse
import imageio
from scipy.spatial.transform import Rotation as R
from utils_geom import save_obj, mesh_to_particles
from utils import mkdir, get_query_dir
from data import store_data
import copy

import ipdb
st=ipdb.set_trace


training_data_dir = get_query_dir("training_data_dir")
testing_data_dir = get_query_dir("testing_data_dir")
output_prefix = get_query_dir("dpi_data_dir")


models_full_check_path = "./files/models_full_check_window.txt"

bad_meshes_ratio = dict()
with open(models_full_check_path, "r") as f:
    for line in f:
        obj_name, x_ratio, y_ratio, z_ratio = line.split(",")
        bad_meshes_ratio[obj_name.encode('UTF-8')] = [float(x_ratio), float(y_ratio), float(z_ratio)]

# output data:
"""
positions: n_nodes x 3
velocities: n_nodes x 3
clusters: clusters: [[[ array(#num_nodes_in_instance) ]]*n_root_level   ]*num_clusters


phases_dict["instance_idx"]:[0, 64, 1024]
                           -> 2 objects
phases_dict["root_num"]: [[8], []]
                         [instance1, instance2]



phases_dict["root_sib_radius"] : [[0.4], []]
phases_dict["root_des_radius"] : [[0.08], []]
phases_dict["root_pstep"]: [[2], []]

rels: (sender, reciever, type)
type=0 for leaf-root, root-root, root-leaf

"""


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--visualization', type=int, default=0)
parser.add_argument('--scenario', default='')
parser.add_argument('--is_human_test', type=int, default=0)
args = parser.parse_args()

visualization = args.visualization
train_ratio = 0.9
shuffle=False

mode = args.mode
if mode == "train":
    data_source_root = training_data_dir
elif mode == "test":
    data_source_root = testing_data_dir
else:
    raise ValueError

output_root = os.path.join(output_prefix, mode)
# make random folder
#randn = random.randint(0, 30)
randn = 0
tmp_path = f"tmp{randn}"
while os.path.exists(tmp_path):
   print(tmp_path, " exists, regenerate")
   randn = random.randint(0, 30)
   tmp_path = f"tmp{randn}"
os.mkdir(tmp_path)


# dominoes


scenario = args.scenario
scene_path = os.path.join(data_source_root, scenario)
trial_names = [file for file in os.listdir(scene_path) if file.endswith("hdf5")]

trial_names.sort()
trial_ndata = [os.path.join(scenario, trial_name) for trial_name in trial_names]
output_roots = [ os.path.join(output_root, path)[:-5]  for path in trial_ndata]

flex_engine = False
if "Drape" in scenario:
    flex_engine = True

color_base = np.array([[1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 0.5],
    [0, 1, 1, 1],
    [0, 0.5, 1, 1],
    [0, 1, 0.5, 1],
    [0.5, 1, 1, 1],
    [0.5, 0.5, 1, 1],
    [1, 1, 1, 1]])
spacing = 0.05 #0.05
dt = 0.01 #default time_step in tdw


for trial_id, trial_hdf5_name in enumerate(trial_ndata):
    output_root = output_roots[trial_id]
    mkdir(output_root)
    filename = os.path.join(data_source_root, trial_hdf5_name)

    # dummy node variance
    positions_stat = np.zeros((3,3))
    velocities_stat= np.zeros((3,3))
    store_data(["positions", "velocities"], [positions_stat, velocities_stat], os.path.join(output_root, 'stat.h5'))

    print("processing", filename)
    f = h5py.File(filename, "r")

    object_ids = f["static"]["object_ids"][:].tolist()
    scales = f["static"]["scale"][:]
    object_names = f["static"]["model_names"][:]

    print("model_names", f["static"]["model_names"][:])

    if b'ramp_with_platform_30' in object_names:
        # this is a bug in the tdw_physics data generation for the Roll scenario with ramp
        # the object name and scales are swaped
        if(object_names[2] == b'ramp_with_platform_30'):
            # correct the scales and object_names
            object_name1 = copy.deepcopy(object_names[1])
            object_names[1] = copy.deepcopy(object_names[2])
            object_names[2] = object_name1

            scales1 = copy.deepcopy(scales[1])
            scales[1] = copy.deepcopy(scales[2])
            scales[2] = scales1
        else:
            assert(object_names[1] == b'ramp_with_platform_30')
            assert(np.linalg.norm(scales[1] - np.array([0.2, 1, 0.5])) < 0.000001)

    nobjects = len(object_ids)

    nonexists = np.zeros((nobjects), dtype=bool)
    clusters = []
    root_num = []
    root_des_radius = []
    instance_idx = [0]
    count_nodes = 0
    instance = []
    material = []
    obj_points = []


    for idx, object_id in enumerate(object_ids):

        if flex_engine:
            points = f["frames"]["0000"]["particles"][str(object_id)][:]
            obj_points.append(points)

        else:
            #vertices, faces = self.object_meshes[object_id]
            vertices = f["static"]["mesh"][f"vertices_{idx}"][:]
            faces = f["static"]["mesh"][f"faces_{idx}"][:]
            obj_name = f["static"]["model_names"][:][idx]


            #checking empty mesh
            if vertices.shape[0] == 0:
                nonexists[idx] = 1
                if obj_name in  [b'cloth_square', b'buddah', b'bowl', b'cone', b'cube', b'cylinder', b'dumbbell', b'octahedron', b'pentagon', b'pipe', b'platonic', b'pyramid', b'sphere', b'torus', b'triangular_prism']:
                    print("critical object with empty mesh", obj_name)
                    print("there should a problem with a data. Please contact the authors to check this")
                    import ipdb; ipdb.set_trace()
                continue


            if obj_name in bad_meshes_ratio:
                print("current trial", filename)
                print(obj_name, scales[idx])
                # cork objects has weird bounding box from tdw like 1.0, 0, 0, so only use the first dimension
                if obj_name in [b"cork_plastic_black", b'tapered_cork_w_hole']:
                    scales[idx, :] *= bad_meshes_ratio[obj_name][0]
                else:
                    scales[idx, :] *= np.array(bad_meshes_ratio[obj_name])
                print("after", scales[idx])


            if max(scales[idx]) > 25:
                # stop if it is not the long bar
                # some heuristic for bad
                if not(abs(scales[idx,0] - 0.05) < 0.00001 and abs(scales[idx,1] - 0.05) < 0.00001 and abs(scales[idx,2] - 100) < 0.00001):
                    if not (obj_name == b"889242_mesh" and np.max(abs(scales[idx] - 37.092888)) < 0.00001):
                        if obj_name in [b"cork_plastic_black", b'tapered_cork_w_hole']:
                            #pass
                            scales[idx] = 25


            obj_filename = f"{tmp_path}/{idx}.obj"
            vertices[:,0] *= scales[idx, 0]
            vertices[:,1] *= scales[idx, 1]
            vertices[:,2] *= scales[idx, 2]

            if vertices.shape[0] == 0:
                nonexists[idx] = 1
                continue

            save_obj(vertices, faces, obj_filename)
            points = mesh_to_particles(obj_filename, spacing=spacing, visualization=False)

            if points.shape[0] == 0:
                nonexists[idx] = 1
                continue
            obj_points.append(points)
            os.remove(obj_filename)

        npts = points.shape[0]
        count_nodes += npts
        clusters.append([[np.array([0]* npts, dtype=np.int32)]])
        root_num.append([1])
        root_des_radius.append([spacing])
        instance_idx.append(count_nodes)

        obj_name = f["static"]["model_names"][:][idx]

        if obj_name == b'cloth_square':

            material.append("cloth")
        else:
            material.append("rigid")
        instance.append(obj_name)



    rollout_dir = output_root
    mkdir(rollout_dir)
    nsteps = len(f["frames"])
    n_objects = len(obj_points)
    n_particles = np.sum([obj_pts.shape[0] for obj_pts in obj_points])

    if visualization:
        colors = [np.ones((obj_pts.shape[0], 4)) * color_base[obj_id:obj_id + 1] for obj_id, obj_pts in enumerate(obj_points)]
        colors = np.concatenate(colors, axis=0)
        print("=================", n_particles, "======================")


    yellow_id = f["static"]["zone_id"][()]
    red_id = f["static"]["target_id"][()]

    yellow_id_order = [order_id for order_id, id_ in enumerate(object_ids) if id_ == yellow_id]
    assert(len(yellow_id_order) == 1)
    yellow_id_order = yellow_id_order[0]

    red_id_order = [order_id for order_id, id_ in enumerate(object_ids)  if id_ == red_id]
    assert(len(red_id_order) == 1)
    red_id_order = red_id_order[0]

    assert(yellow_id_order==0)

    safe_up_to_idx = max(yellow_id_order, red_id_order)
    for id_ in range(safe_up_to_idx):
        assert(not nonexists[id_])

    #import ipdb; ipdb.set_trace()
    # save global info for the trial
    phases_dict = dict()
    phases_dict["instance_idx"] = instance_idx
    phases_dict["root_des_radius"] = root_des_radius
    phases_dict["root_num"] = root_num
    phases_dict["clusters"] = clusters
    phases_dict["instance"] = instance
    phases_dict["material"] = material
    phases_dict["time_step"] = nsteps
    phases_dict["n_objects"] = n_objects
    phases_dict["n_particles"] = n_particles
    phases_dict["obj_points"] = obj_points
    phases_dict["dt"] = dt
    phases_dict["yellow_id"] = yellow_id_order
    phases_dict["red_id"] = red_id_order

    assert(phases_dict["n_objects"] == len(phases_dict["instance_idx"]) - 1)
    assert(phases_dict["n_objects"] == len(phases_dict["root_des_radius"]))
    assert(phases_dict["n_objects"] == len(phases_dict["root_num"]))
    assert(phases_dict["n_objects"] == len(phases_dict["clusters"]))
    assert(phases_dict["n_objects"] == len(phases_dict["instance"]))
    assert(phases_dict["n_objects"] == len(phases_dict["material"]))
    assert(phases_dict["n_objects"] == len(phases_dict["obj_points"]))
    for obj_pts in obj_points:# check not empty mesh
        assert(obj_pts.shape[0] > 0)


    if visualization:
        import vispy.scene
        from vispy import app
        from vispy.visuals import transforms
        c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = c.central_widget.add_view()

        if "Drape" in scenario:
            distance = 4.0
        else:
            distance = 8.0
        view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=100, elevation=10, distance=distance, up='+y')

        p1 = vispy.scene.visuals.Markers()
        p1.antialias = 0  # remove white edge
        view.add(p1)
        vispy_imgf = []
        gt_imgs = []



    with open(os.path.join(rollout_dir, 'phases_dict.pkl'), "wb") as pf:
        pickle.dump(phases_dict, pf)


    #positions_T = np.zeros((nsteps, n_objects, 3), dtype=np.float32)
    #velocities_T = np.zeros((nsteps, n_objects, 3), dtype=np.float32)

    for step in range(nsteps):
        if flex_engine:

            positions = []
            velocities = []
            for idx, object_id in enumerate(object_ids):

                pos = f["frames"][f"{step:04}"]["particles"][str(object_id)][:][:,:3]
                vel = f["frames"][f"{step:04}"]["velocities"][str(object_id)][:][:,:3]

                positions.append(pos)
                velocities.append(vel)
            positions = np.concatenate(positions, axis=0)
            velocities= np.concatenate(velocities, axis=0)
            data_names = ['particle_positions', "particle_velocities"]
            data = [positions, velocities]

            store_data(data_names, data, os.path.join(rollout_dir, str(step) + '.h5'))

        else:
            #print(filename, step)
            positions = f["frames"][f"{step:04}"]["objects"]["positions"][:]
            rotations = f["frames"][f"{step:04}"]["objects"]["rotations"][:] #x,y,z,w


            if np.sum(nonexists) > 0:
                #remove the missing guy
                pos_ = []
                ros_ = []
                for idx, object_id in enumerate(object_ids):
                    if not nonexists[idx]:
                        pos_.append(positions[idx])
                        ros_.append(rotations[idx])
                positions = np.stack(pos_, axis=0)
                rotations = np.stack(ros_, axis=0)

                assert(positions.shape[0] == phases_dict["n_objects"])
                assert(rotations.shape[0] == phases_dict["n_objects"])

            data_names = ['obj_positions', 'obj_rotations']
            data = [positions, rotations]
            store_data(data_names, data, os.path.join(rollout_dir, str(step) + '.h5'))

            if step== 0:
                assert(n_objects == positions.shape[0]), f"nobjects does not match number of positions: {n_objects} vs {positions.shape[0]}"


        if visualization and step%1 == 0:
            #visualize generated particle scenes and original video
            mkdir("vispy")
            tmp = f["frames"][f"{step:04}"]["images"]["_img"][:]

            import PIL.Image as Image
            from PIL import ImageOps
            import io
            image = Image.open(io.BytesIO(tmp))
            image = ImageOps.mirror(image)
            gt_imgs.append(image)
            image.save("tmp.png")
            #with open("tmp.png", "wb") as pngf:
            #    pngf.write(tmp)
            #rotations = np.concatenate([rotations[:, 3:4], rotations[:, :3]], axis=1)
            if flex_engine:
                transformed_obj_pts = []
                for idx, object_id in enumerate(object_ids):

                    positions = f["frames"][f"{step:04}"]["particles"][str(object_id)][:][:,:3]
                    transformed_obj_pts.append(positions)
            else:

                transformed_obj_pts = []
                # compute object rotation and positions
                for obj_id in range(n_objects):
                    #if nonexists[obj_id] == 1:
                    #    continue
                    rot = R.from_quat(rotations[obj_id]).as_matrix()

                    trans = positions[obj_id]

                    transformed_pts = np.matmul(rot, obj_points[obj_id].T).T + np.expand_dims(trans, axis=0)
                    transformed_obj_pts.append(transformed_pts)

            #imageio.imwrite("tmp.png", np.flip(videodata[step], axis=1))
            transformed_pcd = np.concatenate(transformed_obj_pts, axis=0)

            #import pyrender


            p1.set_data(transformed_pcd, edge_color='black', face_color=colors)


            img = c.render()
            #c.show()
            #
            vispy.io.write_png(f"vispy/render_{step}.png", img)
            vispy_imgf.append(f"vispy/render_{step}.png")

    if visualization:
        import imageio
        imgs = []
        for imgf in vispy_imgf:
            imgs.append(imageio.imread(imgf))

        out = imageio.mimsave(
                os.path.join("vispy", 'vid_%d_vispy.gif' % (trial_id)),
                imgs, fps = 20)
        out = imageio.mimsave(
                os.path.join("vispy", 'vid_gt_%d_vispy.gif' % (trial_id)),
                gt_imgs, fps = 20)

        [os.remove(img_path) for img_path in vispy_imgf]

        #app.run()
        import ipdb; ipdb.set_trace()

os.rmdir(tmp_path)
    #print(np.mean(np.linalg.norm(velocities_T, axis=2), axis=1))



