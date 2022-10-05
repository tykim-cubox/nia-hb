import copy
import os
import pickle as pk
import json 

import hashlib
import numpy as np
# import scipy.misc
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO

from hybrik.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from hybrik.utils.presets import SimpleTransform, SimpleTransform3DSMPLCam, SimpleTransformCam


class NIA(data.Dataset):
    CLASSES = ['person']
    num_joints = 17
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                    'left_shoulder', 'right_shoulder',                           # 6
                    'left_elbow', 'right_elbow',                                 # 8
                    'left_wrist', 'right_wrist',                                 # 10
                    'left_hip', 'right_hip',                                     # 12
                    'left_knee', 'right_knee',                                   # 14
                    'left_ankle', 'right_ankle')                                 # 16
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
        self._ann_file = os.path.join(root, 'annotations', ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._img_folder = img_folder
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

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
        
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.augment = cfg.MODEL.EXTRA.AUGMENT
        self.dz_factor = cfg.MODEL.EXTRA.get('FACTOR', None)
        
        self._loss_type = cfg.LOSS['TYPE']

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        
        if cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d':
                self.transformation = SimpleTransform(
                    self, scale_factor=self._scale_factor,
                    color_factor=self._color_factor,
                    occlusion=self._occlusion,
                    input_size=self._input_size,
                    output_size=self._output_size,
                    rot=self._rot, sigma=self._sigma,
                    train=self._train, add_dpg=self._dpg,
                    loss_type=self._loss_type, dict_output=True)

        elif cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d_cam':
            self.transformation = SimpleTransformCam(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, dict_output=True, 
                bbox_3d_shape=self.bbox_3d_shape)

        self._items, self._labels = self._load_jsons()

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img_id = 0
        
        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        # img = scipy.misc.imread(img_path, mode='RGB')
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
        labels = []
        items  = []
        
        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)
            
            
        for ann_annotation in database['annotations']:
            image = ann_annotation['image']
            width, height = float(ann_annotation['width']), float(ann_annotation['height'])
            
            # bbox = np.asarray(ann_annotation['bbox'][0], dtype=np.float32)
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                        bbox_xywh_to_xyxy(ann_annotation['bbox']), width, height)


            if xmax <= xmin or ymax <= ymin:
                continue
            
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = ann_annotation['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = ann_annotation['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, ann_annotation['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                
            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue
            # 2D 데이터셋에선 필요없음    
            # f = np.array(ann_annotations['f'], dtype=np.float32)
            # c = np.array(ann_annotations['c'], dtype=np.float32)
            
            labels.append({'bbox' : (xmin, ymin, xmax, ymax),
                        'width': width,
                        'height': height,
                        'joints_3d': joints_3d,
                        'keypoints': ann_annotation['keypoints']})
            
            abs_path = os.path.join(self._root, self._img_folder, image)
            items.append(abs_path)
        return items, labels
    
    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]


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