
import copy
import json
import os
import pickle as pk

import cv2
import numpy as np
import torch.utils.data as data
from hybrik.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from hybrik.utils.pose_utils import (cam2pixel, pixel2cam, cam2pixel_matrix, pixel2cam_matrix, reconstruction_error)
from hybrik.utils.presets import (SimpleTransform3DSMPL,
                                  SimpleTransform3DSMPLCam)



class NIA3D(data.Dataset):
    CLASSES = ['person']
    num_joints = 14
    num_thetas = 24
    # H36M 기준 joints
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    
    joints_name_29 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'           # 28
    )
        
    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/nia',
                 img_folder = 'imgs',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):
        self._cfg = cfg

        self._img_folder = img_folder
        self._ann_file = os.path.join(
            root, f'annotations', ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._img_folder = img_folder
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        
        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = getattr(cfg.MODEL.EXTRA, 'DEPTH_DIM', None)

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        # joint 총 갯수 확인해야할듯
        self.num_joints = 28 if self._train else 17

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.augment = cfg.MODEL.EXTRA.AUGMENT
        self.dz_factor = cfg.MODEL.EXTRA.get('FACTOR', None)

        self._loss_type = cfg.LOSS['TYPE']


        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.kinematic = cfg.MODEL.EXTRA.get('KINEMATIC', False)
        self.classfier = cfg.MODEL.EXTRA.get('WITHCLASSFIER', False)
        
        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.root_idx_smpl = self.joints_name_29.index('pelvis')
        self.lshoulder_idx_29 = self.joints_name_29.index('left_shoulder')
        self.rshoulder_idx_29 = self.joints_name_29.index('right_shoulder')

        self._items, self._labels = self._load_jsons()
        

        if cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d':
            self.transformation = SimpleTransform3DSMPL(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, scale_mult=1,) # two_d=True)

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img_id = 0
        
        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)


    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)

        for ann in database['annotations']:
            
            image_id = ann['image']
            abs_path = os.path.join(self._root, self._img_folder, image_id)
            image_id = 0
            
            width, height = float(ann['width']), float(ann['height'])
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann['bbox']), width, height)

            f, c = np.array(ann['f'], dtype=np.float32), np.array(ann['c'], dtype=np.float32)

            keypoints_3d = ann['keypoints_3d']
            for idx in [self.joints_name_17.index('Torso'), self.joints_name_17.index('Nose'), self.joints_name_17.index('Head')]:
                keypoints_3d.insert(idx*3, float(0))
                keypoints_3d.insert(idx*3+1, float(0))
                keypoints_3d.insert(idx*3+2, float(0))
            
            joint_cam_17 = np.array(keypoints_3d).reshape(17, 3)
            joint_img_17 = cam2pixel(joint_cam_17, f, c)
            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]

            joint_cam = np.array(ann['smpl_joints']) 
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)

            joint_img_29 = cam2pixel(joint_cam_29, f, c)
            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_cam_29[self.root_idx_smpl, 2]

            joint_vis_17 = np.ones((17, 3))
            joint_vis_29 = np.ones((29, 3))
        
            if 'angle_twist' in ann.keys():
                twist = ann['angle_twist']
                angle = np.array(twist['angle'])
                cos = np.array(twist['cos'])
                sin = np.array(twist['sin'])
                assert (np.cos(angle) - cos < 1e-6).all(), np.cos(angle) - cos
                assert (np.sin(angle) - sin < 1e-6).all(), np.sin(angle) - sin
                phi = np.stack((cos, sin), axis=1)
                # phi_weight = np.ones_like(phi)
                phi_weight = (angle > -10) * 1.0 # invalid angles are set to be -999
                phi_weight = np.stack([phi_weight, phi_weight], axis=1)
            else:
                phi = np.zeros((23, 2))
                phi_weight = np.zeros_like(phi)

            if 'beta' in ann.keys():
                beta = ann['beta']
            else:
                beta = np.zeros((10))
                
            if 'theta' in ann.keys():
                theta = np.array(ann['thetas']).reshape(self.num_thetas, 3)
            else:
                theta = np.zeros((self.num_thetas, 3))

            if 'root_coord' in ann.keys():
                root_cam = np.array(ann['root_coord'])
            else:
                root_cam = np.zeros(3)
            labels.append({'bbox' : (xmin, ymin, xmax, ymax),
                           'img_id': image_id,
                           'img_path': abs_path,
                           'width': width,
                           'height': height,
                           'joint_img_17': joint_img_17,
                           'joint_vis_17': joint_vis_17,
                           'joint_cam_17': joint_cam_17,
                           'joint_relative_17': joint_relative_17,
                           'joint_img_29': joint_img_29,
                           'joint_vis_29': joint_vis_29,
                           'joint_cam_29': joint_cam_29,
                           'twist_phi': phi,
                           'twist_weight': phi_weight,
                           'beta': beta,
                           'theta': theta,
                           'root_cam': root_cam,
                           'f': f,
                           'c': c})            
            
            items.append(abs_path)
            
        return items, labels

    @property
    def joint_pairs_17(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

    @property
    def joint_pairs_24(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

    @property
    def joint_pairs_29(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))

    

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
