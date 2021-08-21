# preprocessing
import h5py
import copy
import os
import trimesh
import pickle
import argparse
import numpy as np
import random
from tqdm import tqdm
import imageio
from scipy.spatial.transform import Rotation as R
from utils_geom import save_obj, mesh_to_particles
from utils import mkdir
from data import store_data
import copy
from utils import mkdir, get_query_dir
import ipdb
st=ipdb.set_trace

#TODO: change this to test dir
training_data_dir  = get_query_dir("training_data_dir")
testing_data_dir  = get_query_dir("testing_data_dir")
output_prefix = get_query_dir("dpi_data_dir")

models_full_check_path = "./files/models_full_check_window.txt"

parser = argparse.ArgumentParser()
parser.add_argument('--visualization', type=int, default=0)
parser.add_argument('--is_not_parallel', type=int, default=1)
parser.add_argument('--scenario', default='')
parser.add_argument('--mode', default='')
parser.add_argument('--label_filename', default='labels')
args = parser.parse_args()


if args.mode == "train":
    data_source_root = training_data_dir
elif args.mode == "test":
    data_source_root = testing_data_dir
else:
    raise ValueError

output_root = os.path.join(output_prefix, args.mode)
visualization = False
build_label_txt = True
train_ratio = 0.9
shuffle=False


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

scenarios = [args.scenario]


for scenario in scenarios:

    if build_label_txt:
        mkdir(os.path.join(output_root, args.label_filename), is_assert=args.is_not_parallel)

        label_file_name = os.path.join(output_root, args.label_filename, f"{scenario}.txt")
        label_file = open(label_file_name, "w")


    scene_path = os.path.join(data_source_root, scenario)
    trial_names = [file for file in os.listdir(scene_path) if file.endswith("hdf5")]
    trial_id = 0
    for trial_name in tqdm(trial_names, desc=scenario):

        spacing = 0.05 #0.05

        dt = 0.01 #default time_step in tdw

        #print("processing", filename)
        f = h5py.File(os.path.join(scene_path, trial_name), "r")
        object_ids = f["static"]["object_ids"][:].tolist()
        scales = f["static"]["scale"][:]
        object_names = f["static"]["model_names"][:]

        images = []
        nsteps = len(f["frames"])
        is_hit = False
        for step in range(nsteps):
            target_contacting_zone = f["frames"][f"{step:04}"]["labels"]["target_contacting_zone"][()]
            if target_contacting_zone:
                is_hit = True
                if not visualization:
                    break

            #print(step, is_hit, target_contacting_zone)
            if visualization:
                import PIL.Image as Image
                from PIL import ImageOps
                import io
                tmp = f["frames"][f"{step:04}"]["images"]["_img"][:]
                image = Image.open(io.BytesIO(tmp))
                image_np = np.array(image)

                if is_hit:
                    image_np[:50, :, :] = np.expand_dims(np.expand_dims(np.array([255,0,0]), axis=0), axis=0)
                else:
                    image_np[:50, :, :] = np.expand_dims(np.expand_dims(np.array([255,255,255]), axis=0), axis=0)

                #image = ImageOps.mirror(image)
                images.append(image_np)
        if build_label_txt:
            label_file.write(",".join([trial_name, str(is_hit)]) + "\n")

        if visualization:
            import imageio
            out = imageio.mimsave(
                    os.path.join("vispy", 'rgb_gt_%s_%s.gif' % (scenario, trial_name)),
                    images, fps = 20)

        trial_id += 1
    if build_label_txt:
        label_file.close()


# os.rmdir(tmp_path)