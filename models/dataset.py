import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    

    #use opencv + c2w format
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose() #transpose in advance, in case of avoiding the transpose in the training process.
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose




class Dataset:
    # def original_init(self, conf):
    #     super(Dataset, self).__init__()
    #     print('Load data: Begin')
    #     self.device = torch.device('cuda')
    #     self.conf = conf

    #     self.data_dir = conf.get_string('data_dir')
    #     self.render_cameras_name = conf.get_string('render_cameras_name')
    #     self.object_cameras_name = conf.get_string('object_cameras_name')

    #     self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
    #     self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

    #     camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
    #     self.camera_dict = camera_dict
    #     self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.jpg')))
    #     self.n_images = len(self.images_lis)
    #     self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
    #     self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
    #     self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

    #     # world_mat is a projection matrix from world to image
    #     self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

    #     self.scale_mats_np = []

    #     # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    #     self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

    #     self.intrinsics_all = []
    #     self.pose_all = []

    #     for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
    #         P = world_mat @ scale_mat
    #         P = P[:3, :4]
    #         intrinsics, pose = load_K_Rt_from_P(None, P)
    #         self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
    #         self.pose_all.append(torch.from_numpy(pose).float())

    #     self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
    #     self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
    #     self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
    #     self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
    #     self.focal = self.intrinsics_all[0][0, 0]
    #     self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
    #     self.H, self.W = self.images.shape[1], self.images.shape[2]
    #     self.image_pixels = self.H * self.W

    #     object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
    #     object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
    #     # Object scale mat: region of interest to **extract mesh**
    #     object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
    #     object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
    #     object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
    #     self.object_bbox_min = object_bbox_min[:3, 0]
    #     self.object_bbox_max = object_bbox_max[:3, 0]
    #     print('Load data: End')

    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.jpg')))
        self.n_images = len(self.images_lis)
        #self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
        #self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        #self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        #self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        
        img_0 = cv.imread(self.images_lis[0])
        self.H, self.W = img_0.shape[0], img_0.shape[1]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate full rays of a img at world space from one camera with consideration of downsample resolution
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size,ray_idx):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).to(self.device)
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).to(self.device)
        img_idx = torch.tensor(img_idx).to(self.device)
        #color = self.images[img_idx].to(self.device)[(pixels_y, pixels_x)]    # batch_size, 3
        #mask = self.masks[img_idx].to(self.device)[(pixels_y, pixels_x)]      # batch_size, 3

        color = torch.from_numpy((cv.imread(self.images_lis[img_idx])/256.0).astype(np.float32)).to(self.device)[(pixels_y, pixels_x)]    # batch_size, 3

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        #ray_idx = torch.ones(batch_size,1).to(self.device)*ray_idx
        #return torch.cat([rays_o.cuda(), rays_v.cuda(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10
        return torch.cat([rays_o.cuda(), rays_v.cuda(), color], dim=-1).cuda()    # batch_size, 10
    def gen_random_rays_in_images(self,idx_of_images,rays_per_image):
        """
        Generate random rays at world space from one camera.
        """
        rays_o_all = []
        rays_v_all = []
        color_all = []
        #ray_idx_all = []
        for i in range(len(idx_of_images)):
            img_idx = idx_of_images[i]
            data = self.gen_random_rays_at(img_idx, rays_per_image,i)
            rays_o, rays_d, true_rgb = data[:, :3], data[:, 3: 6], data[:, 6: 9]
            rays_o_all.append(rays_o)
            rays_v_all.append(rays_d)
            color_all.append(true_rgb)
        rays_o_all = torch.cat(rays_o_all, dim=0)
        rays_v_all = torch.cat(rays_v_all, dim=0)
        color_all = torch.cat(color_all, dim=0)
        #ray_idx_all = torch.cat(ray_idx_all, dim=0)
        return rays_o_all,rays_v_all,color_all,None

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

class PG_Dataset(Dataset):
    def __init__(self,conf):
        #super(PG_Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')# data_dir = ./public_data/CASE_NAME/
        self.camera_pose_json_dir = conf.get_string('cameras_pose')
        # self.render_cameras_name = conf.get_string('render_cameras_name') # render_cameras_name = cameras_sphere.npz  the input camera pose equivalent to the render camera pose
        # self.object_cameras_name = conf.get_string('object_cameras_name') # object_cameras_name = cameras_sphere.npz.. %what is that means?

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=False) # camera_outside_sphere = True
        #self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1) # scale_mat_scale = 1.1

        json_dir = os.path.join(self.data_dir,self.camera_pose_json_dir)
        self.camera_json = json.load(open(json_dir,encoding="UTF-8"))
        self.n_images = len(self.camera_json["frames"])
        #self.camera_dict = camera_dict
        #self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.images_np =[]
        self.pose_list = []
        self.intrinsics_all = []
        self.images_lis = []
        for i, frame in enumerate(self.camera_json["frames"]):
            #load images
            this_image_filename = os.path.join( self.data_dir, frame["rgb_path"])
            self.images_lis.append(this_image_filename)
            self.images_np.append(cv.imread(this_image_filename))

            #load poses
            this_pose = np.array(frame["camtoworld"])
            this_pose[2,:]  *= -1
            this_pose = this_pose[np.array([1, 0, 2, 3]), :]
            this_pose[0:3, 1:3] *= -1
            this_pose = np.linalg.inv(this_pose)
            
            # c2w = np.linalg.inv(w2c)
            # # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            # c2w[0:3, 1:3] *= -1
            # c2w = c2w[np.array([1, 0, 2, 3]), :]
            # c2w[2, :] *= -1
            self.pose_list.append(this_pose)
            
            #load intrinsics
            this_intrinsics = np.array(frame["intrinsics"])
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = this_intrinsics
            self.intrinsics_all.append(intrinsics)
        
        self.pose_all = np.array(self.pose_list)
        #self.pose_all[:,0:3,1:3] *= -1 #transform the coordinate between openGL and openCV
        self.images_np = np.array(self.images_np)
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.intrinsics_all = np.array(self.intrinsics_all)

        #Automatically scale the poses to fit in +/- 1 bounding box.
        self.scale_factor = 1.0
        self.scale_factor /= 1.3*(np.max(np.abs(self.pose_all[:, :3, 3])))
        self.pose_all[:, :3, 3] *= self.scale_factor
        
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis])
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu() #useless

        
        print(f"self,images.shape = {self.images.shape}")
        print(f"self.masks.shape:{self.masks.shape}")
        # self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.from_numpy(self.intrinsics_all.astype(np.float32)).to(self.device)   # [n_images, 4, 4]

        #TODO it's that necessary to inverse the intrinsic Mat?
        #x = K[R|T]X
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        

        
        self.pose_all = torch.from_numpy(self.pose_all.astype(np.float32)).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        # object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        # object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        # object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]