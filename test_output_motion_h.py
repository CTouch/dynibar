import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio.v2 as imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image_mono
from ibrnet.model import DynibarMono
from ibrnet.projection import Projector
from ibrnet.data_loaders.data_utils import get_nearest_pose_ids
from ibrnet.data_loaders.data_utils import get_interval_pose_ids
from ibrnet.data_loaders.llff_data_utils import load_mono_data
from ibrnet.data_loaders.llff_data_utils import batch_parse_llff_poses 
from ibrnet.data_loaders.llff_data_utils import batch_parse_vv_poses
import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from render_monocular_bt import DynamicVideoDataset

from ibrnet.render_ray import compute_traj_pts


def parse_camera(camera):
    # camera [34]
    camera = camera.squeeze(0)
    h, w = camera[:2]
    pose = camera[-16:].reshape(4, 4)   # [4, 4]
    intrin = camera[2:18].reshape(4, 4)  # [4, 4]
    
    return pose, intrin, h, w

def proj_points_to_view(pt, pose, intrin, h, w):
    # pt [3]
    # pose [4, 4]
    # intrin [4, 4]
    # h, w
    pt = torch.cat([pt, torch.ones(1).cuda()], dim=0)
    pixel_loc = torch.matmul(intrin, torch.matmul(torch.inverse(pose), pt))
    pixel_loc = pixel_loc[:2] / torch.clamp(pixel_loc[2], min=1e-8)
    return pixel_loc
    


if __name__ == '__main__':
    
    # camera_pose_path = '/data3/Touch/dynibar/dynamic-test/kid-running-test_mr-42_w-disp-0.100_w-flow-0.010_anneal_cycle-0.1-0.1-w_mode-0/-2_0.1/kid-running_450000/videos/camera_pose/camera_pose_0.npy'
    # camera_pose = np.load(camera_pose_path)
    # print(camera_pose)
    # print(camera_pose.shape)
    
    
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    test_dataset = DynamicVideoDataset(args, scenes=args.eval_scenes)
    args.num_frames = test_dataset.num_frames

    # Create ibrnet model
    model = DynibarMono(args)
    
    # data_dir = "/data3/Touch/dynibar/dynamic-test/parkour_test_mr-42_w-disp-0.100_w-flow-0.010_anneal_cycle-0.1-0.1-w_mode-0/-2_0.1/parkour_400000/videos"
    data_dir = "/data3/Touch/dynibar/dynamic-test/kid-running-test_mr-42_w-disp-0.100_w-flow-0.010_anneal_cycle-0.1-0.1-w_mode-0/-2_0.1/kid-running_450000/videos"
    
    raw_coeff_dir = os.path.join(data_dir, 'raw_coeff')
    alpha_dy_dir = os.path.join(data_dir, 'alpha_dy')
    pts_ref_dir = os.path.join(data_dir, 'pts_ref')
    rgb_out_dir = os.path.join(data_dir, 'rgb_out')
    
    
    camera_pose_dir = os.path.join(data_dir, 'camera_pose')
    
    # load alpha_dy_0
    alpha_dy_0_path = os.path.join(alpha_dy_dir, 'alpha_dy_0.npy')
    alpha_dy_0 = np.load(alpha_dy_0_path)
    alpha_dy_0 = alpha_dy_0.reshape([288,512,64,1])
    # alpha_dy_0[:,:,-10,:] = 0
    # find index of the max value of alpha_dy_0
    max_index = np.argmax(alpha_dy_0)     # regard as the point on the foreground, focus it 
    print('max_index: ', max_index)
    print('alpha_dy_0.max', alpha_dy_0.max())
    # load pts_ref_0
    pts_ref_0_path = os.path.join(pts_ref_dir, 'pts_ref_0.npy')
    pts_ref_0 = np.load(pts_ref_0_path).reshape([-1, 3])
    print('alpha_dy_0.shape', alpha_dy_0.shape)
    print('pts_ref_0.shape', pts_ref_0.shape)
    # find the point on the foreground
    pt_xyz = pts_ref_0[max_index, :]
    print('pt: ', pt_xyz)
    
    # # output model.trajectory_basis as a numpy array
    # trajectory_basis = model.trajectory_basis.detach().cpu().numpy()
    trajectory_basis = model.trajectory_basis
    print(trajectory_basis.shape)

    # print('model.trajectory_basis[3]', model.trajectory_basis[3])
    t_offset = 3
    with torch.no_grad():
        pts_ref_0 = torch.from_numpy(pts_ref_0).float().cuda()
        zeros = torch.ones_like(pts_ref_0[..., :1]) * (t_offset / args.num_frames)
        pts_ref_0 = torch.cat([pts_ref_0, zeros], dim=-1)
        
        raw_coeff = model.motion_mlp(pts_ref_0)
        raw_coeff_x = raw_coeff[...,0:6]
        raw_coeff_y = raw_coeff[...,6:12]
        raw_coeff_z = raw_coeff[...,12:18]
        # compute next frame pt
        rel_pt_curr = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, trajectory_basis[0 + t_offset])
        rel_pt_next = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, trajectory_basis[0 + t_offset + 1])
        delta_pos = rel_pt_next - rel_pt_curr
        delta_len = torch.norm(delta_pos, dim=-1)
        longest_idx = torch.argmax(delta_len)
        print('delta_len[longest]', delta_len[longest_idx])
        
    print("======================================================")
    hphix = []
    hphiy = []
    hphiz = []
    pt_xyzt = pts_ref_0[longest_idx]
    pts_xyzt = []
    with torch.no_grad():
        for i in range(trajectory_basis.shape[0] - 6):
            camera_pose_path = os.path.join(camera_pose_dir, 'camera_pose_' + str(i) + '.npy')
            camera = torch.from_numpy(np.load(camera_pose_path)).float().cuda()
            pose, intrin, h, w = parse_camera(camera)
            pixel_loc = proj_points_to_view(pt_xyzt[:3], pose, intrin, h, w)
            print('pixel_loc: ', pixel_loc)
            
            rgb_out_path = os.path.join(rgb_out_dir, str(i) + '.png')
            rgb_out = cv2.imread(rgb_out_path)
            cv2.circle(rgb_out, (int(pixel_loc[0]), int(pixel_loc[1])), 5, (0, 0, 255), -1)
            cv2.imwrite(f'focus_{i}.png', rgb_out)
            
            pts_xyzt.append(pt_xyzt)
            pt_xyz = pt_xyzt[0:3]
            raw_coeff = model.motion_mlp(pt_xyzt.reshape([1, 4]))
            raw_coeff_x = raw_coeff[...,0:6]
            raw_coeff_y = raw_coeff[...,6:12]
            raw_coeff_z = raw_coeff[...,12:18]
            hphix.append(raw_coeff_x * trajectory_basis[i + t_offset,:])
            hphiy.append(raw_coeff_y * trajectory_basis[i + t_offset,:])
            hphiz.append(raw_coeff_z * trajectory_basis[i + t_offset,:])
            
            # compute next frame pt
            rel_pt_curr = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, trajectory_basis[i + t_offset])
            rel_pt_next = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, trajectory_basis[i + t_offset + 1])
            # print('rel_pt_curr: ', rel_pt_curr)
            # print('rel_pt_next: ', rel_pt_next)
            pt_xyz = pt_xyz + rel_pt_next[0] - rel_pt_curr[0]
            # add time t to pt_xyz
            pt_xyzt = torch.cat([pt_xyz, torch.Tensor([(i + t_offset + 1) / args.num_frames]).float().cuda()], dim=0)
            # print('pt_xyzt', pt_xyzt)
        print('pts_xyzt[0]',pts_xyzt[0])
        print('pts_xyzt[1]',pts_xyzt[1])
        
        print('delta pts_xyzt: ', pts_xyzt[-1] - pts_xyzt[0])
        print('delta pts_xyzt len: ', torch.norm(pts_xyzt[-1][:3] - pts_xyzt[0][:3]))
    # hphix = np.array(hphix)
    # hphiy = np.array(hphiy)
    # hphiz = np.array(hphiz)
    
    output_dir = 'traj_output'
    # for i in range(trajectory_basis.shape[1]):
    #     plt.figure()
    #     plt.plot(range(trajectory_basis.shape[0]), trajectory_basis[:,i])
    #     plt.plot(range(3, 3 + hphix.shape[0]), hphix[:, i])
    #     plt.legend(['trajectory_basis', 'hphix'])
    #     output_path = os.path.join(output_dir, 'hphix_and_basis' + str(i) + '.png')
    #     plt.savefig(output_path)
    #     plt.close()
        
    #     plt.figure()
    #     plt.plot(range(trajectory_basis.shape[0]), trajectory_basis[:,i])
    #     plt.plot(range(3, 3 + hphiy.shape[0]), hphiy[:, i])
    #     plt.legend(['trajectory_basis', 'hphiy'])
    #     output_path = os.path.join(output_dir, 'hphiy_and_basis' + str(i) + '.png')
    #     plt.savefig(output_path)
    #     plt.close()
        
    #     plt.figure()
    #     plt.plot(range(trajectory_basis.shape[0]), trajectory_basis[:,i])
    #     plt.plot(range(3, 3 + hphiz.shape[0]), hphiz[:, i])
    #     plt.legend(['trajectory_basis', 'hphiz'])
    #     output_path = os.path.join(output_dir, 'hphiz_and_basis' + str(i) + '.png')
    #     plt.savefig(output_path)
    #     plt.close()
    
    trajectory_basis = trajectory_basis.detach().cpu()
    plt.figure()    
    for i in range(trajectory_basis.shape[1]):
        # add distance between two subplots
        # plt.subplots_adjust(hspace=0.5, wspace=0.5)        
        # plt.subplot(2,2,1)
        plt.plot(range(trajectory_basis.shape[0]), trajectory_basis[:,i])
        
        # plt.subplot(2,2,2)
        # plt.plot(range(3, 3 + hphix.shape[0]), hphix[:, i])
        
        # plt.subplot(2,2,3)
        # plt.plot(range(3, 3 + hphiy.shape[0]), hphiy[:, i])
        
        # plt.subplot(2,2,4)
        # plt.plot(range(3, 3 + hphiz.shape[0]), hphiz[:, i])
    plt.legend(range(trajectory_basis.shape[1]))
    # title = ['trajectory_basis', 'hphix', 'hphiy', 'hphiz']
    # for i in range(4):
    #     plt.subplot(2,2,i+1)
    #     # legend set outside the plot
    #     plt.legend(range(trajectory_basis.shape[1]), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #     # plt.legend(range(trajectory_basis.shape[1]))
    #     plt.title(title[i])
    output_path = os.path.join(output_dir, 'trajectory_basis.png')
    plt.savefig(output_path)