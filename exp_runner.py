import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset,PG_Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer

from nerfacc.estimators.occ_grid import OccGridEstimator
import nerfacc
class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        #self.dataset = PG_Dataset(self.conf['dataset'])# load dataset
        self.dataset = Dataset(self.conf['dataset'])# load dataset
        self.render_step_size = 2.0 / (self.conf['model.neus_renderer.n_samples']+ \
                                       self.conf['model.neus_renderer.n_importance']* self.conf['model.neus_renderer.up_sample_steps'])
        self.selcted_img_num_each_batch = 2
        self.selcted_rays_num_each_img = 50

        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()


        #utilize nerfacc
        self.scene_aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=self.device)
        self.estimator = OccGridEstimator(
        roi_aabb=self.scene_aabb, resolution=64, levels=1
        ).to(self.device)
    

    #应该不需要这个函数了，参照instant-nsr-pl中对nerfacc在neus上的使用，
    #alpha_fn直接设置为None,因为alpha的计算在neus中需要梯度，而nerfacc的sampling函数中如果开启梯度，显存消耗是原来的2倍以上。
    #alpha_fn设置为None之后，应该是只使用了occpancy grid来减少采样点个数。
    def alpha_fn_with_gradientCal(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]#shape = (n_samples,) 代表每个采样点所在光线的原点
                t_dirs = rays_d[ray_indices]#shape = (n_samples,) 代表每个采样点所在光线的方向
                pts = t_origins + t_dirs * (t_ends-t_starts)[:, None] /2.0 #shape = (n_samples,3) 代表每个采样点的位置,取的是每个采样区间的中点
                sample_interval_dist = (t_ends - t_starts)[...,None]
                cos_anneal_ratio = self.get_cos_anneal_ratio()

                sdf_nn_output = self.sdf_network(pts)
                sdf = sdf_nn_output[:, :1]
                #feature_vector = sdf_nn_output[:, 1:]
                gradients = self.sdf_network.gradient(pts).squeeze()
                
                #sampled_color_without_bkgdColor = self.color_network(pts, gradients, t_dirs, feature_vector).reshape(pts.shape[0], 3)#shape = (n_samples,3)

                inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
                inv_s = inv_s.expand(pts.shape[0], 1)
                true_cos = (t_dirs * gradients).sum(-1, keepdim=True)

                iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                            F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
                
                estimated_next_sdf = sdf + iter_cos * sample_interval_dist * 0.5
                estimated_prev_sdf = sdf - iter_cos * sample_interval_dist * 0.5
                
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

                p = prev_cdf - next_cdf
                c = prev_cdf
                # print(f"p.shape = {p.shape}")
                # print(f"c.shape = {c.shape}")
                alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0) #shape = (n_samples,1)
                # print(f"alpha.shape = {alpha.shape}")
                return alpha.squeeze()
    
    
    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.get_cos_anneal_ratio()) +
                     F.relu(-true_cos) * self.get_cos_anneal_ratio())  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha


    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        self.sdf_network.train()
        self.color_network.train()
        self.deviation_network.train()
        
        # images_idx = self.get_random_sequal_images_idx(2)
        # #print(f"this_images_batch = {images_idx}")
        # rays_o, rays_d, true_rgb, mask = self.dataset.gen_random_rays_in_images(images_idx, 400)
        
        for iter_i in tqdm(range(res_step)):
            #每隔几个iter就重新采样一次图片
            if self.iter_step % 10==0:
                images_idx = self.get_random_sequal_images_idx(self.selcted_img_num_each_batch)
            #每隔几个iter重新采样一次光线
            if self.iter_step % 5==0:
                rays_o, rays_d, true_rgb, mask = self.dataset.gen_random_rays_in_images(images_idx,self.selcted_rays_num_each_img)
            
            # near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            # print(f"near = {near}, far = {far}")



            #NeRFacc的occpancy grid更新。这里参考了instant-nsr-pl的代码，直接估计alpha值来更新occpancy grid
            def occ_eval_fn(x):
                sdf = self.sdf_network(x)[:, :1]
                inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
                #step_size =  1.732 * 2 * 1.0/ 1024 #参考instant-nsr-pl的,这里应该只是一个估计，所以不需要算得很准确，故直接采用固定值
                estimated_next_sdf = sdf - self.render_step_size * 0.5 #其实sdf的求导不是在光线方向，而是sdf的梯度方向，即法线方向，故直接对sdf加减即可
                estimated_prev_sdf = sdf + self.render_step_size * 0.5
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                p = prev_cdf - next_cdf
                c = prev_cdf
                alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                return alpha
            # update occupancy grid
            with torch.no_grad():
                self.estimator.update_every_n_steps(
                    step=self.iter_step,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=0.02,
                    warmup_steps =3000,
                )
            
            
            
            def rgb_alpha_fn(t_starts, t_ends, ray_indices):
                t_origins = rays_o[ray_indices]#shape = (n_samples,) 代表每个采样点所在光线的原点
                t_dirs = rays_d[ray_indices]#shape = (n_samples,) 代表每个采样点所在光线的方向
                position = t_origins + t_dirs * (t_ends-t_starts)[:, None]/2.0 #shape = (n_samples,3) 代表每个采样点的位置,取的是每个采样区间的中点
                dir = t_dirs
                sample_interval_dist = (t_ends - t_starts)[...,None] #shape = (n_samples,) 代表每个采样区间的长度
                #z_val是没有考虑ray_o和ray_dir时，光线总长为1，其上每个采样区间的左边界
                #pos 是每个采样区间的中点
                alpha,sample_pts_color,extras = self.renderer.nerfacc_forward(
                    position, 
                    dir,
                    sample_interval_dist,
                    self.sdf_network,
                    self.deviation_network,
                    self.color_network,
                    None,None,None,
                    self.get_cos_anneal_ratio())
                return sample_pts_color,alpha.squeeze(),extras

            #ray_indices.shape = (n_samples), 代表每个采样点所在光线的索引

            #try 0.5.3 nerfacc
            with torch.no_grad():
                ray_indices, t_starts, t_ends = self.estimator.sampling(
                    rays_o, rays_d, alpha_fn=None, near_plane=0.01, far_plane=5,render_step_size=self.render_step_size, early_stop_eps=1e-2, alpha_thre=0) #试试0.3.5的nerfacc？
            
            
            if ray_indices.shape == torch.Size([0]):
                print(ray_indices.shape)
                continue #如果没有采样到点，就跳过这个batch


            infer_color, infer_opacity, infer_depth, extras = nerfacc.rendering(
                    t_starts, t_ends, ray_indices, n_rays=int(rays_o.shape[0]), rgb_alpha_fn=rgb_alpha_fn)
            
            
            #try nerfacc 0.3.5
            # print("begin nerfacc")
            # with torch.no_grad():
            #     ray_indices, t_starts, t_ends = nerfacc.ray_marching(
            #         rays_o, 
            #         rays_d,
            #         scene_aabb=self.scene_aabb,
            #         grid=None, #self.occupancy_grid if self.config.grid_prune else None,
            #         alpha_fn=None,
            #         near_plane=None, far_plane=None,
            #         render_step_size=self.render_step_size,
            #         stratified=False,
            #         cone_angle=0.0,
            #         alpha_thre=0.0
            #         )
            # print("finish_ray_marching")
            # ray_indices = ray_indices.long()
            # t_origins = rays_o[ray_indices]
            # t_dirs = rays_d[ray_indices]
            # midpoints = (t_starts + t_ends)[:, None] / 2.
            # positions = t_origins + t_dirs * midpoints
            # dists = (t_ends - t_starts)[...,None]
            # print("begin nerfacc_forward,samples points num = ",positions.shape[0])
            # alpha,sample_pts_color,extras = self.renderer.nerfacc_forward(
            #         positions, 
            #         t_dirs,
            #         dists,
            #         self.sdf_network,
            #         self.deviation_network,
            #         self.color_network,
            #         None,None,None,
            #         self.get_cos_anneal_ratio())
            # print(f"alpha.shape = {alpha.shape}")
            # print(f"alpha size = {alpha.size()}")
            # print("finish nerfacc_forward")
            # weights = nerfacc.render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=self.selcted_img_num_each_batch*self.selcted_rays_num_each_img)
            # #opacity = nerfacc.accumulate_along_rays(weights, ray_indices, values=None, n_rays=self.selcted_img_num_each_batch*self.selcted_rays_num_each_img)
            # #depth = nerfacc.accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=self.selcted_img_num_each_batch*self.selcted_rays_num_each_img)
            # comp_rgb = nerfacc.accumulate_along_rays(weights, ray_indices, values=sample_pts_color, n_rays=self.selcted_img_num_each_batch*self.selcted_rays_num_each_img)
            # print("finish nerfacc")
            # print(extras)



            gradient_error = extras['network_output']["gradient_error"]
            s_val= extras['network_output']['s_val']
            
            color_fine_loss =  F.l1_loss(infer_color, true_rgb,reduction='sum') / len(ray_indices)
            eikonal_loss = gradient_error
            loss = color_fine_loss + eikonal_loss * self.igr_weight
            psnr = 20.0 * torch.log10(1.0 / (((infer_color - true_rgb)**2).sum() / (rays_o.shape[0])).sqrt())



            
            #原版NeuS的渲染
            #background
            # background_rgb = None
            # if self.use_white_bkgd:
            #     background_rgb = torch.ones([1, 3])

            # if self.mask_weight > 0.0:
            #     mask = (mask > 0.5).float()
            # else:
            #     mask = torch.ones_like(true_rgb)

            # mask_sum = mask.sum() + 1e-5
            # # render_out = self.renderer.render(rays_o, rays_d, near, far,
            # #                                   background_rgb=background_rgb,
            # #                                   cos_anneal_ratio=self.get_cos_anneal_ratio())

            # color_fine = render_out['color_fine']
            # s_val = render_out['s_val']
            # #cdf_fine = render_out['cdf_fine']
            # gradient_error = render_out['gradient_error']
            # #weight_max = render_out['weight_max']
            # #weight_sum = render_out['weight_sum']

            # # Loss
            # color_error = (color_fine - true_rgb) * mask
            # color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            # psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            # eikonal_loss = gradient_error

            # # if mask != None:
            # #     mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            # #     loss = color_fine_loss +\
            # #        eikonal_loss * self.igr_weight +\
            # #        mask_loss * self.mask_weight
            # # else:
            # loss = color_fine_loss +\
            #     eikonal_loss * self.igr_weight
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            #self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            #self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={} ,psnr = {}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr'],psnr))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)
        #将0~n-1（包括0和n-1）随机打乱后获得的数字序列 torch.randperm(10)===> tensor([2, 3, 6, 7, 8, 9, 1, 5, 0, 4])
    def get_images_idx_all(self):
        return range(self.dataset.n_images-1)
    
    def get_random_sequal_images_idx(self, num_sequal_images=5):
        band = int(num_sequal_images/2)
        random_idx = np.random.randint(band,self.dataset.n_images-band)
        return list(range(random_idx-band,random_idx+band))

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end]) #0->1

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha 

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello huangziyang~')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=1536, threshold=args.mcube_threshold)
    elif args.mode == 'validate_image':
        runner.validate_image(idx=-1, resolution_level=1)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
