### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union, Tuple, Callable
import time
import logging ## avoid weird wsidicom logspam: WARNING:root:Orientation [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0] is not orthogonal with equal lengths with column rotated 90 deg from row
logging.getLogger().setLevel(logging.ERROR)
### External Imports ###
import numpy as np
import torch as tc
from wsidicom import WsiDicom
import skimage
import PIL
### Internal Imports ###
from BPTorch.datasets import get_bg_rm_tool
########################

class WsiDicomDataset():
    """
    Parent class (to overload) responsible for managing the whole slide images. It handles masking the WSI content and splits the image into patches during loading.
    """
    def __init__(self,
        wsi_path : Union[str, Path],
        target_mpp : float = 0.5,
        patch_size : tuple = (224, 224),
        patch_stride: tuple = (224, 224),
        calculate_mask : bool = True,
        calculate_mask_params : str = 'dilated-otsu',
        transforms : Callable = None,
        half_precision : bool = True,
        verbose:bool = True,
        precomputed = False,
        metadata = None,
        ):
        """
        
        """
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.calculate_mask = calculate_mask
        self.calculate_mask_params = calculate_mask_params
        self.transforms = transforms
        self.half_precision = half_precision
        self.gs_threshold = None
        self.image_slide = None
        self.verbose = verbose
        self.meta = metadata
        self.target_mpp = target_mpp

        if not precomputed:
            image_slide = WsiDicom.open(self.wsi_path)
            try: self.resolution_level = self._infer_level(image_slide, target_mpp); self.resample=False
            except Exception as e: 
                self.resample = True
                self.resolution_level=None
                if self.verbose: print(e)
            
            mask_slide = get_bg_rm_tool(self.calculate_mask_params)(image_slide, verbose=self.verbose) if self.calculate_mask else None
            
            if not self.resample: 
                self.mpp = (image_slide.levels[self.resolution_level].mpp.width, image_slide.levels[self.resolution_level].mpp.height)
                self.upper_left_corners, self.coordinates = self._calculate_upper_left_corners(image_slide, mask_slide)
                self.resolution = (image_slide.levels[self.resolution_level].size.height, image_slide.levels[self.resolution_level].size.width)
            else: 
                self.mpp = (target_mpp, target_mpp)
                self.upper_left_corners, self.coordinates = self._calculate_upper_left_corners_resampled(image_slide, mask_slide) ## self.resolution is set inside this func

            image_slide.close()
            
    
    @staticmethod      
    def get_default_kwargs():
        kwargs = {
            "target_mpp": 0.5,
            "patch_size": (224, 224),
            "patch_stride": (224, 224),
            "calculate_mask": True,
            "calculate_mask_params": "dilated-otsu",
            "transforms": None,
            "half_precision": True,
            "precomputed": False,
            "metadata": None,
        }
        return kwargs
        
    def _calculate_upper_left_corners(self, image_slide : WsiDicom, mask_slide : WsiDicom=None):
        """Calculates the coordinates of the upper left corners of every patch in the slide given the patch stride and patch size arguments
        Args:
            image_slide (WsiDicom): _description_
            mask_slide (WsiDicom, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        height = image_slide.levels[self.resolution_level].size.height 
        width = image_slide.levels[self.resolution_level].size.width 
        height_multiplier = 1 if self.resolution_level == 0 else int(height / image_slide.levels[self.resolution_level].size.height)
        width_multiplier = 1 if self.resolution_level == 0 else int(width / image_slide.levels[self.resolution_level].size.width)
        step_height = self.patch_stride[0] * height_multiplier
        step_width = self.patch_stride[1] * width_multiplier
        corners = []
        coordinates = []
        self.number_of_rows = len(list(range(0, height - step_height, step_height)))
        self.number_of_cols = len(list(range(0, width - step_width, step_width)))
        for y in range(0, height - step_height, step_height):
            for x in range(0, width - step_width, step_width):
                coordinate = np.array([int(x), int(y)])
                coordinate_given_level = np.array([int(x / width_multiplier), int(y / height_multiplier)])
                
                if mask_slide is not None:
                    mask_patch = mask_slide.read_region((coordinate[0], coordinate[1]), self.resolution_level, self.patch_size)
                    if np.any(mask_patch):
                        corners.append(coordinate)
                        coordinates.append(coordinate_given_level)
                        
                else:
                    corners.append(coordinate)
                    coordinates.append(coordinate_given_level)
        corners = np.array(corners)
        if self.verbose: print(f"=== Obtained {corners.shape[0]} patches for image")
        coordinates = np.array(coordinates)
        return corners, coordinates
    
    def _calculate_upper_left_corners_resampled(self, image_slide : WsiDicom, mask_slide : WsiDicom=None, mpp=0.5):
        """Calculates the coordinates of the upper left corners of every patch in the slide given the patch stride and patch size arguments
        Args:
            image_slide (WsiDicom): _description_
            mask_slide (WsiDicom, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        um2mm = 0.001 ## um * um2mm = mm (u = mu) converter
        self.mm_p_px = mpp*um2mm
        
        ## convert image size to mm coordinates
        height = image_slide.levels[0].size.height 
        width = image_slide.levels[0].size.width 
        if self.verbose: print(f"=== Original resolution at level 0 = {image_slide.levels[0].mpp.height}mpp is {(height, width)}")
        height_mpp = height*image_slide.levels[0].mpp.height
        width_mpp = width*image_slide.levels[0].mpp.width
        height_mm = height_mpp*um2mm
        width_mm = width_mpp*um2mm
        
        ## write resolution at mpp
        self.resolution=(round(height_mpp/mpp), round(width_mpp/mpp))
        if self.verbose: print(f"=== Resolution at {mpp} is {self.resolution}")
        
        ## convert strides to mm
        step_height_px = self.patch_stride[0]
        step_width_px = self.patch_stride[1]
        step_height_mm = self.patch_stride[0]*mpp*um2mm
        step_width_mm = self.patch_stride[1]*mpp*um2mm
        
        ## convert patch size to mm
        patch_height_mm = self.patch_size[0]*mpp*um2mm
        patch_width_mm = self.patch_size[1]*mpp*um2mm
        
        ## overwrite stirdes and sizes
        self.patch_size = (patch_height_mm,patch_width_mm)
        self.patch_stride = (step_height_mm,step_width_mm)
        
        ## generate patches
        corners = []
        coordinates = []
        self.number_of_rows = len(list(range(0, round(self.resolution[0] - step_height_px), step_height_px)))
        self.number_of_cols = len(list(range(0, round(self.resolution[1] - step_width_px), step_width_px)))

        for y in range(0, round(self.resolution[0] - step_height_px), step_height_px):
            for x in range(0, round(self.resolution[1] - step_width_px), step_width_px):
                coordinate = np.array([float(x*mpp*um2mm), float(y*mpp*um2mm)]) ## width, height
                coordinate_given_level = np.array([int(x), int(y)])
                ## mask case
                if mask_slide is not None:
                    mask_patch = mask_slide.read_region_mm((coordinate[0], coordinate[1]), self.mm_p_px, (patch_width_mm, patch_height_mm))
                    if np.any(mask_patch):
                        corners.append(coordinate) ## the coord for load
                        coordinates.append(coordinate_given_level) ## the coord for the model
                        
                ## full case 
                else:
                    corners.append(coordinate)
                    coordinates.append(coordinate_given_level)
                    
        corners = np.array(corners)
        if self.verbose: print(f"=== Obtained {corners.shape[0]} patches for image")
        coordinates = np.array(coordinates)
        return corners, coordinates

    def _infer_level(self, wsi, mpp=0.5, tolerance=0.06):
        mpp_per_level = np.zeros(len(wsi.levels))
        for i in range(len(wsi.levels)):
            mpp_per_level[i] = wsi.levels[i].mpp.width
        level = np.argmin(np.abs(mpp_per_level-mpp))
        if self.verbose: print(f"=== Selecting level {level} with {mpp_per_level[level]} as its closest to {mpp}")
        if abs(mpp_per_level[level]-mpp)>tolerance:
            raise ValueError(f"===!! The discrepancy between the closest level and the desired level is too high.")
        return level
    
    def _ensure_image_is_open(self):
        """Ensures that an instance of the image is opened inside the worker to avoid errors from multiple workers accessing the same instance at the same time leading to errors
        """
        if self.image_slide is None:
            self.image_slide = WsiDicom.open(self.wsi_path)
            
    def _load_patch_at(self, coordinates):
        self._ensure_image_is_open()
        try: self.resolution_level = self._infer_level(self.image_slide, self.target_mpp); self.resample=False
        except: self.resample = True
        if not self.resample: patch = self.image_slide.read_region(coordinates, self.resolution_level, self.patch_size).convert('RGB')
        else: patch = self.image_slide.read_region_mm(coordinates, self.mm_p_px, self.patch_size).convert('RGB')
        patch = np.array(patch)
        return patch

    def _get_coordinates(self, idx):
        return self.upper_left_corners[idx], self.coordinates[idx]
    
    def get_resolution(self):
        return self.resolution

    def get_number_of_rows(self):
        return self.number_of_rows

    def get_number_of_cols(self):
        return self.number_of_cols

    def __len__(self) -> int:
        return len(self.upper_left_corners)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.image_slide is not None: self.image_slide.close()
        
    def __del__(self):
        if self.image_slide is not None: self.image_slide.close()
    
    def __getitem__(self, idx : Union[int, tuple]) -> Tuple[tc.Tensor]:
        corners, coordinates = self._get_coordinates(idx) if type(idx)==int else idx
        patch = self._load_patch_at(corners).astype(np.float16 if self.half_precision else np.float32)
        coordinates = tc.tensor(coordinates, dtype=int)
        if self.transforms is not None:
            patch = self.transforms(patch)
        to_return = {'image' : patch, 'coordinates' : coordinates, 'metadata':self.meta}
        return to_return