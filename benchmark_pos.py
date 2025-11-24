

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from load_obj import load_obj
#from Poisson_Depth_Recovery.main import PoissonOperator

sys.path.append("Poisson-Depth-Recovery") #can't rename this folder, but python cannot do hyphens
poisson_module = __import__("main")
PoissonOperator = poisson_module.PoissonOperator

from photometric import visualize_normals

from util import load_dataset

def compute_camera_matrix(fov=50, image_res=512, origin_vec=[0, 0, 3], target_vec=[0,0,0], up_vec=[0,1,0]):
  '''
  intrinsic related:
    fov
    image_res
  extrinsic related 
    origin vec (where camera is positioned)
    target vec (where we are looking)
    up vec (z or y dir, usually)
  
  Return:
    K (intrinsic matrix, 3x3)
    E (extrinsic matrix, 4x4)
    P (full projecction matrix, from P= KE or P = K[R | t])
  '''

  origin = np.array(origin_vec, dtype=float)
  target = np.array(target_vec, dtype=float)
  up = np.array(up_vec, dtype=float)

  fov_rad = np.radians(fov)
  f = image_res / (2 * np.tan(fov_rad/2))

  K = np.array([
    [f, 0, image_res/2],
    [0, f, image_res/2],
    [0, 0, 1]
  ])

  forward = origin - target
  forward /= np.linalg.norm(forward)

  right = np.cross(up, forward)
  right /= np.linalg.norm(right)

  actual_up = np.cross(forward, right)

  R = np.array([right, actual_up, forward])
  t = -R @ origin

  E = np.eye(4)
  E[:3,:3] = R
  E[:3, 3] = t

  P = K @ E[:3,:]


  return K, E, P



if __name__ == "__main__":

  #load mesh file



  PROJECT_DIR = Path.cwd()
  DATASET_DIR = PROJECT_DIR / "dataset" / "bunny"

  # MESH_NAME = "bunny.obj"
  # MESH_PATH = PROJECT_DIR / "meshes" / MESH_NAME
  # MESH_PATH = str(MESH_PATH) 
  # print(f"MESH PATH is: {MESH_PATH}")

  # v, f, vn = load_obj(MESH_PATH, load_normals=True)

  n = np.load("/mnt/c/Users/Rusla/OneDrive/Desktop/10701/10701_light_estimation/results/bunny/calibrated_ff/normals.npy")
  n = np.nan_to_num(n, nan=0.0)
  print("Loading dataset...")
  dataset = load_dataset(str(DATASET_DIR))

  depth_maps = dataset.depth()
  depth = depth_maps[0]
 
  print("vertex normals shape is", n.shape)

  #using mitsuba setup code as default params...
  cam_int, cam_ext, cam_matrix = compute_camera_matrix()

  print(f"INT:\n{cam_int} \n EXT:\n{cam_ext} \n MATRIX:\n {cam_matrix}")
  cam_matrix = cam_int

  fx = cam_matrix[0, 0]
  fy = cam_matrix[1, 1]
  px = cam_matrix[0, 2]
  py = cam_matrix[1, 2]

  #questions: wtf is depth d 

  image_res = 512

  print("depth dim", depth.shape)
  print("normals dim", n.shape)

  x, y = np.meshgrid(np.arange(image_res), np.arange(image_res))
  u = x - px
  v = np.flipud(y) - py

  p = -n[..., 0] / (u * n[..., 0] + v * n[..., 1] + fx * n[..., 2])
  q = -n[..., 1] / (u * n[..., 0] + v * n[..., 1] + fy * n[..., 2])
  # d[~d_mask] = 1
  # d = np.log(d)

  #rembmer this is from an output image so lets do this porperly




  n_mask = ~(np.all(np.abs(n) == np.array([0, 0, 1]), axis=-1))
  # print("normals (gt)", n)
  # Before passing to PoissonOperator
  d_mask = (depth > 0) & n_mask  # Valid depth pixels
  depth_log = depth.copy()
  depth_log[depth_log <= 0] = 1  # Avoid log(0)
  depth_log = np.log(depth_log)
  depth_log[~d_mask] = 0  # Zero out invalid regions
  
  
  batch = PoissonOperator(np.dstack([p, q]), n_mask.astype(np.int8), depth_log, 0.1)
  d_est = np.exp(batch.run())

  d_est[~n_mask] = 0
  depth[~n_mask] = 0

  fig, axes = plt.subplots(1, 4, figsize=(15, 5))
  fig.suptitle('Normal Maps Comparison', fontsize=16)

  # Ground truth
  im = axes[0].imshow(d_est, cmap='viridis')
  cbar = plt.colorbar(im)
  axes[0].set_title('poisson integrated')
  axes[0].axis('off')
  
  im2 = axes[1].imshow(depth, cmap='viridis')
  cbar = plt.colorbar(im2)
  axes[1].set_title('Ground Truth Depth')
  axes[1].axis('off')
  
  im2 = axes[2].imshow((n+1.0)/2.0)
  axes[2].set_title('Ground Truth Normals')
  axes[2].axis('off')

  im2 = axes[3].imshow(np.abs(d_est - depth))
  axes[3].set_title('Error')
  axes[3].axis('off')
  plt.show()




  
  
  




  #grab normals

  #integrate with poisson depth recovery and visualize

  #integrate with conjugate gradient descent? And visualize. I am worried about the convention but it should still form a matrix