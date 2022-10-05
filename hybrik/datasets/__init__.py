from .h36m_smpl import H36mSMPL
from .hp3d import HP3D
from .pw3d import PW3D
from .mscoco import Mscoco
from .nia2d import NIA
from .nia3d import NIA3D

from .mix_dataset import MixDataset
from .mix_dataset_cam import MixDatasetCam
from .mix_dataset2_cam import MixDataset2Cam
from .h36m_nia_mix_dataset import H36NIADataset
from .h36m_nia_mix_dataset_cam import H36NIADatasetCam
from .h36m_coco_mix_dataset import H36MCoCoDataset
from .h36m_dataset import H36MDataset
from .h36m_nia3d_coco_mix_dataset import NIA3DMixDataset

from .single_dataset import SingleH36MDataset

__all__ = ['H36mSMPL', 'HP3D', 'PW3D', 'NIA', 'NIA3D', 'MixDataset', 'MixDatasetCam', 'MixDataset2Cam', 'H36NIADataset', 'H36NIADatasetCam', 'H36MCoCoDataset', 'H36MDataset', 'Mscoco', 'NIA3DMixDataset', 'SingleH36MDataset']
