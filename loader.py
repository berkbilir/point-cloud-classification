
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import os.path
import numpy as np
import torch
import open3d as o3d
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors


object_names=["/chair/","/toilet/","/table/","/airplane/","/dresser/","/bed/","/sofa/","/desk/"]

def normalize_pc(pc_np):
    ### Complete for task 1
    # Normalize the cloud to unit cube
    # input numpy ndarray -> output numpy ndarray

    pc_np_norm = pc_np - np.mean(pc_np, axis=0) 
    pc_np_norm /= np.max(np.linalg.norm(pc_np_norm, axis=1))

    return pc_np_norm

def spectify(points):
    ### Complete for task 3
    # Find the spectral embedding of the cloud
    # input numpy ndarray -> output numpy ndarray

    embedding = SpectralEmbedding(n_components=3)
    emb = embedding.fit_transform(points)

    return emb

def group_knn(points,seed_points,group_size):
    # Builds a list of point groups around seed_points, can optionally be used
    neigh = NearestNeighbors(n_neighbors=group_size)
    neigh.fit(points)
    groups=neigh.kneighbors( seed_points, return_distance=False)
    groups_list=[points[groups[indexes,:],:] for indexes in range(seed_points.shape[0])]
    return groups_list

def vis_voxel_o3d(voxel_o3d):
    # Visualizes the open3d voxel object
    o3d.visualization.draw_geometries([voxel_o3d],mesh_show_wireframe=True)

def vis_points(points):
    # Visualizes the open3d point cloud object or a numpy containing point coordinates
    if type(points)==o3d.open3d_pybind.geometry.PointCloud:
        o3d.visualization.draw_geometries([points],mesh_show_wireframe=True)
    elif type(points)==np.ndarray:
        points_o3d=o3d.geometry.PointCloud()
        points_o3d.points=o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([points_o3d],mesh_show_wireframe=True)


def mesh_parser(mesh,parse_to):
    # Parses the query mesh into different representations
    if parse_to=='point':
        pc_o3d=o3d.geometry.PointCloud()
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)
        pc_np=np.asarray(pc_o3d.points)
        #vis_points(pc_np)
        pc_np=normalize_pc(pc_np)
        #vis_points(pc_np)
        return torch.from_numpy(pc_np).to(dtype=torch.float)
    elif parse_to=='voxel':
        
        pc_o3d = o3d.geometry.PointCloud()
        ### Complete for task 2
        
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)

        pc_o3d.scale(1 / np.max(pc_o3d.get_max_bound() - pc_o3d.get_min_bound()),
        center=pc_o3d.get_center())

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc_o3d,
                                                            voxel_size=1/31)

        #vis_voxel_o3d(voxel_grid)

        voxel_list = voxel_grid.get_voxels()

        vox_np = np.empty((len(voxel_list),3))

        for i in range(len(voxel_list)):
            vox_np[i] = voxel_list[i].grid_index


        grid = np.zeros((32,32,32))


        for i in vox_np:
            grid[i[0].astype(int)][i[1].astype(int)][i[2].astype(int)] = 1
            #print(i)

       

        grid = np.expand_dims(grid, axis=0)
        #print(grid.shape)


        return torch.from_numpy(grid).to(dtype=torch.float)

    elif parse_to=='spectral':
        pc_o3d=o3d.geometry.PointCloud()
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)
        ### Complete for task 3
        pc_np=np.asarray(pc_o3d.points)
        pc_np=normalize_pc(pc_np)
    
        spect_np=spectify(pc_np)

        x = np.column_stack((pc_np,spect_np))
        #vis_points(spect_np)


        return torch.from_numpy(x).to(dtype=torch.float)

    


class ChallengeDataset(Dataset):

    def __init__(self, folderPath,train_test,rep_type,processed_root='processed/'):
        self.root=folderPath
        self.mode=train_test
        self.rep_type=rep_type
        self.file_list=[]
        self.label_list=[]
        self.paths=[folderPath+obj+train_test for obj in object_names]
        self.rep_type=rep_type
        self.num_classes=2
        self.processed_path=processed_root+self.rep_type+'_'+train_test
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        
        self.read_raw=False if os.path.exists(self.processed_path+'/0.pt') else True

        
        for label_id,path in enumerate(self.paths):
            for (dirpath, dirnames, filenames) in os.walk(path):
                self.label_list.extend([label_id for f in filenames])
                self.file_list.extend([dirpath+'/'+f for f in filenames])

    def __getitem__(self, index):
        if self.read_raw:
            mesh_o3d=o3d.io.read_triangle_mesh(self.file_list[index])
            input_tensor=mesh_parser(mesh_o3d,self.rep_type)
            target=torch.tensor(self.label_list[index],dtype=torch.int64)
            torch.save({"input": input_tensor, "target":target},self.processed_path+'/%d.pt'%index)
            return input_tensor,target
        else:
            data=torch.load(self.processed_path+'/%d.pt'%index)
            return data["input"],data["target"] 


    def __len__(self):
        return len(self.file_list)


