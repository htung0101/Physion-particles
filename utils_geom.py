import numpy as np
import os
import trimesh
import copy
import matplotlib.pyplot as plt
import binvox_rw as binvox_rw

import ipdb
st = ipdb.set_trace

def calc_rigid_transform(XX, YY):
    X = XX.copy().T
    Y = YY.copy().T

    mean_X = np.mean(X, 1, keepdims=True)
    mean_Y = np.mean(Y, 1, keepdims=True)
    X = X - mean_X
    Y = Y - mean_Y
    C = np.dot(X, Y.T)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    R = np.dot(Vt.T, np.dot(D, U.T))
    T = mean_Y - np.dot(R, mean_X)

    '''
    YY_fitted = (np.dot(R, XX.T) + T).T
    print("MSE fit", np.mean(np.square(YY_fitted - YY)))
    '''

    return R, T


def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))
        for face in faces:
            f.write("f")
            for vertex in face:
                f.write(" %d" % (vertex + 1))
            f.write("\n")


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


#filename = "/home/htung/Documents/2021/example_meshes/0000_obj1_1.binvox"




def mesh_to_particles(mesh_filename, spacing=0.2, visualization=False, remove_outofbbox_pts=True):
    """
    mesh_filename:"/home/htung/Documents/2021/example_meshes/Rubber_duck.obj"
    spacing: the size of the voxel grid in real-world scale # what is the distance between 2 particles
    """

    # the output path used by binvox
    output_binvox_filename = mesh_filename.replace(".obj", ".binvox")
    # check if output file exists
    assert not os.path.isfile(output_binvox_filename), f"binvox file {output_binvox_filename} exists, please delete it first"

    #load mesh
    mesh = as_mesh(trimesh.load_mesh(mesh_filename, process=True))
    # make the mesh transparent
    mesh_ori = copy.deepcopy(mesh) # for visualization
    mesh_ori.visual.face_colors[:,3] = 120



    edges = mesh.bounding_box.extents
    maxEdge = max(edges)
    meshLower0 = mesh.bounds[0,:]
    meshUpper0 = mesh.bounds[1,:]

    # shift the mesh to it is in some bounding box [0, +x], [0, +y], [0, +z]
    #mesh.vertices -= meshLower0


    edges = mesh.bounding_box.extents
    maxEdge = max(edges)
    meshLower = mesh.bounds[0,:]
    meshUpper = mesh.bounds[1,:]
    #  tweak spacing to avoid edge cases for particles laying on the boundary
    # just covers the case where an edge is a whole multiple of the spacing.
    spacingEps = spacing*(1.0 - 1e-4)
    spacingEps_p = 0 #(9e-4) if spacing >= 0.001 else 0


    # naming is confusing, dx denotes the number of voxels in each dimension
    # make sure to have at least one particle in each dimension
    dx = 1 if spacing > edges[0] else int(edges[0]/spacingEps)
    dy = 1 if spacing > edges[1] else int(edges[1]/spacingEps)
    dz = 1 if spacing > edges[2] else int(edges[2]/spacingEps)

    maxDim = max(max(dx, dy), dz);

    #expand border by two voxels to ensure adequate sampling at edges
    # extending by a small offset to avoid point sitting exactly on the boundary
    meshLower_spaced = meshLower - 2.0 * spacing - spacingEps_p
    meshUpper_spaced = meshUpper +  2.0 * spacing + spacingEps_p

    maxDim_spaced = maxDim + 4

    voxelsize_limit = 512
    if maxDim_spaced > voxelsize_limit :
        print(meshLower_spaced)
        print(meshUpper_spaced)
        print("=====")
        for dim in range(3):
            if edges[dim] < (voxelsize_limit - 4) * spacing:
                continue # short edge, no need to chunk
            else:
                amount_to_cut = edges[dim] - (voxelsize_limit - 4) * spacing
                meshLower_spaced[dim] += amount_to_cut * 0.5
                meshUpper_spaced[dim] -= amount_to_cut * 0.5
        maxDim_spaced = voxelsize_limit

    # we shift the voxelization bounds so that the voxel centers
    # lie symmetrically to the center of the object. this reduces the
    # chance of missing features, and also better aligns the particles
    # with the mesh
    # ex. |1|1|1|0.3| --> |0.15|1|1|0.15|
    meshOffset = np.zeros((3))
    meshOffset[0] = 0.5 * (spacing - (edges[0] - (dx-1)*spacing))
    meshOffset[1] = 0.5 * (spacing - (edges[1] - (dy-1)*spacing))
    meshOffset[2] = 0.5 * (spacing - (edges[2] - (dz-1)*spacing))
    meshLower_spaced -= meshOffset;

    # original space
    #meshLower_spaced += meshLower0
    meshUpper_spaced = meshLower_spaced + maxDim_spaced * spacing + 2 * spacingEps_p

    #print(meshLower_spaced, meshUpper_spaced)
    #print(f'binvox -aw -dc -d {maxDim_spaced} -pb -bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} {meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} -t binvox {mesh_filename}')

    # voxelsize_limit = 512
    # if maxDim_spaced > voxelsize_limit :
    #     import ipdb; ipdb.set_trace()
    #     cutting_space = spacing * (maxDim_spaced -voxelsize_limit ) * 0.5
    #     maxDim_spaced = voxelsize_limit
    #     meshLower_spaced += cutting_space
    #     meshUpper_spaced -= cutting_space

    #     import ipdb; ipdb.set_trace()


    os.system(f'binvox -aw -dc -d {maxDim_spaced} -pb -bb {meshLower_spaced[0]} {meshLower_spaced[1]} {meshLower_spaced[2]} {meshUpper_spaced[0]} {meshUpper_spaced[1]} {meshUpper_spaced[2]} -t binvox {mesh_filename}')
    #print(meshLower_spaced, meshUpper_spaced)

    # binvox -aw -dc -d 5 -pb -bb -0.9 -0.4 -0.9 0.9 1.4 0.9 -t binvox {mesh_filename}
    #os.system(f"binvox -aw -dc -d 5 -pb -bb -0.9 -0.4 -0.9 0.9 1.4 0.9 -t binvox {mesh_filename}")

    # convert voxel into points

    with open(output_binvox_filename, 'rb') as f:
         m1 = binvox_rw.read_as_3d_array(f)



    adjusted_spacing = (maxDim_spaced * spacing + 2 * spacingEps_p)/maxDim_spaced
    x, y, z = np.nonzero(m1.data)
    points = np.expand_dims(meshLower_spaced, 0) + np.stack([(x + 0.5)*adjusted_spacing, (y + 0.5)*adjusted_spacing, (z + 0.5)*adjusted_spacing], axis=1)
    os.remove(output_binvox_filename)


    if remove_outofbbox_pts:

        bbox = mesh_ori.bounds
        lower_bound = bbox[0, :]
        upper_bound = bbox[1, :]

        idx = (points[:, 0] - upper_bound[0] <= 0) * (points[:, 0] - lower_bound[0] >= 0)
        idy = (points[:, 1] - upper_bound[1] <= 0) * (points[:, 1] - lower_bound[1] >= 0)
        idz = (points[:, 2] - upper_bound[2] <= 0) * (points[:, 2] - lower_bound[2] >= 0)

        points = points[idx*idy*idz]

    if visualization:
        # for visualization
        axis = trimesh.creation.axis(axis_length=1)
        pcd = trimesh.PointCloud(points)
        (axis + mesh_ori).show()
        (trimesh.Scene(pcd) + axis).show()

    return points