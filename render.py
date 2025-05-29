import torch
import numpy as np 
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))
                
def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()         # Loads cameras, dynamic masks, etc
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)  
            visualizer.visualize(result, camera)

        visualizer.summarize()



def make_extrinsics(old_rot, old_trans, yaw, i):
    c2w = np.array([[0., 0., 1., ],  # change (u,v,z) to (x,y,z) : (right, down, forward) to (forward, left, up)
                    [-1., 0., 0., ],
                    [0., -1., 0., ]])

    matrix = np.eye(4)
    old_yaw, old_pitch, old_roll = R.from_matrix(old_rot @ inv(c2w)).as_euler('zyx', degrees=True)
    # print("i", i, "old yaw", old_yaw)
    rot = R.from_euler('zyx', [yaw + old_yaw, 0, 0], degrees=True).as_matrix()
    matrix[:3,:3] = rot @ c2w
    matrix[:3, 3] = old_trans
    return matrix



def render_lidar():

    # INTRINSICS
    K = np.zeros((3,3))
    K[0, 0] = 554.26  # Focal length in x direction
    K[1, 1] = 554.26  # Focal length in y direction
    K[0, 2] = 1920/2  # Principal point x
    K[1, 2] = 1920/2  # Principal point y
    K[2, 2] = 1       # Homogeneous coordinate


    cfg.render.save_image = True
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory_lidar', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))
        # print("CAMERAS", cameras)
        old_rot, old_trans = None, None

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            # print("idx", idx, "id", camera.id)
            if camera.id % 3 == 0:
                old_ext = camera.get_extrinsic()
                old_rot = old_ext[:3, :3]
                old_trans = old_ext[:3, 3]
            yaw = 0 if camera.id % 3 == 0 else 120 if camera.id % 3 == 1 else 240
            # The order here is important
            camera.image_height = 1920
            camera.image_width = 1920
            camera.set_intrinsic(K)
            camera.set_extrinsic(make_extrinsics(old_rot, old_trans, yaw, camera.id))
            camera.FoVx = 554.26
            camera.FoVy = 554.26
            result = renderer.render_all(camera, gaussians)
            visualizer.visualize(result, camera)

        visualizer.summarize()
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    elif cfg.mode == 'lidar':
        render_lidar()
    else:
        raise NotImplementedError()
