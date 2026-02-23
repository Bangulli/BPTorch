### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import logging ## avoid weird wsidicom logspam: WARNING:root:Orientation [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0] is not orthogonal with equal lengths with column rotated 90 deg from row
logging.getLogger().setLevel(logging.ERROR)
from typing import Any
### External Imports ###
import numpy as np
import skimage
import PIL
import cv2
########################

# TODO:
#   - Make optim optional with a param in the constructor
#   - Implement slow read-region-mm

class Otsu:
    """Full resolution background removal using raw otsu.
    Used to load binarized patches from an image.
    Not optimized as Raw otsu does not rely on expensive mathematical morphology operations.
    """
    def __init__(self, wsi, verbose=False):
        self.wsi = wsi
        self.verbose = verbose
        thumbnail = self._get_thumbnail()
        w, h = thumbnail.size
        thumbnail = thumbnail.crop((w*0.1, h*0.1, w*0.9, h*0.9)) ## Crop edges to avoid bad thresholding
        self.gs_threshold = skimage.filters.threshold_otsu(np.array(thumbnail)) 
        if self.verbose: print(f"==== Otsu threshold is {self.gs_threshold}")
        
    def _get_thumbnail(self, size=512):
        return self.wsi.read_thumbnail((size, size)).convert("L")
    
    def read_region(self, coords, level, patch_size):
        mask_patch = self.wsi.read_region(coords, level, patch_size).convert("L")
        mask_patch = np.array(mask_patch)<self.gs_threshold # binarize: bg is bright, fg is dark
        return mask_patch
    
    def read_region_mm(self, coords_mm, mm_pp, patch_size_mm):
        mask_patch = self.wsi.read_region_mm(coords_mm, mm_pp, patch_size_mm).convert("L")
        mask_patch = np.array(mask_patch)<self.gs_threshold # binarize: bg is bright, fg is dark
        return mask_patch
    
    def close(self): # NOTE: usually doesnt need to be called, as the underlying wsi is closed at the end of dataset.__init__ already
        self.wsi.close()
        
    def get_segmented_thumbnail_PIL(self):
        thmbnl = self._get_thumbnail()
        thmbnl = np.array(thmbnl)<self.gs_threshold
        thmbnl = PIL.Image.fromarray(thmbnl)
        return thmbnl

class DilatedOtsu:
    """Scaled background removal using otsu+dilation
    Used to load binarized scaled patches of an image. 
    The image is scaled such that the longes side is 5000px usually equating to ~5x magnification.
    then this large thumbnail is binarized using the method described abouve and returned patches are scaled to this thumbnail, not original size
    """
    def __init__(self, wsi, se_rad_microns=64, verbose=False):
        self.wsi = wsi
        self.verbose = verbose
        thumbnail = self._get_thumbnail()
        w, h = thumbnail.size
        thumbnail = thumbnail.crop((w*0.1, h*0.1, w*0.9, h*0.9)) ## Crop edges to avoid bad thresholding
        self.gs_threshold = skimage.filters.threshold_otsu(np.array(thumbnail)) 
        if self.verbose: (f"==== Otsu threshold is {self.gs_threshold}")
        self.se_rad_microns = se_rad_microns
        
        # None inits
        self.seg_thmbnl = None
        self.multiplier = None
        
    def _get_thumbnail(self, size=512):
        return self.wsi.read_thumbnail((size, size)).convert("L")
            
    def _get_level_mpp(self, level):
        mpp = self.wsi.levels[level].mpp.height
        return mpp
    
    ## Full scale implementation with per-patch binarization, very slow hence disabled.
    # def read_region_slow(self, coords, level, patch_size):
    #     if self.se is None: self.se = self._get_se(level)
    #     mask_patch = self.wsi.read_region(coords, level, patch_size).convert("L")
    #     mask_patch = np.array(mask_patch)<self.gs_threshold # binarize: bg is bright, fg is dark
    #     mask_patch = cv2.dilate(mask_patch.astype(np.uint8), self.se, iterations=1)#skimage.morphology.dilation(mask_patch, self.se)
    #     return mask_patch
    
    def read_region(self, coords, level, patch_size): ## passed coords are w, h
        if self.seg_thmbnl is None: self.seg_thmbnl = self.get_segmented_thumbnail(level)
        if self.multiplier is None: 
            self.multiplier = np.array(self._get_region_to_thmbnl_converter(level))
            if self.verbose: print(f"=== Patch size scaled to {np.round(patch_size*self.multiplier).astype(int)} in thumbnail of size {self.seg_thmbnl.shape}")
        coords_in_thmbnl = np.round(coords*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.round(patch_size*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.maximum(patch_size_in_thmbnl, [1, 1])
        return self.seg_thmbnl[coords_in_thmbnl[1]:coords_in_thmbnl[1]+patch_size_in_thmbnl[1], coords_in_thmbnl[0]:coords_in_thmbnl[0]+patch_size_in_thmbnl[0]] ## NOTE flipped coords, because numpy expects height, width
    
    def _get_region_to_thmbnl_converter(self, level): # returns width, height
        thmbnl_shape = self.seg_thmbnl.shape ## height, width
        wsi_at_level_shape = self._get_resolution(level)
        return thmbnl_shape[1]/wsi_at_level_shape[0], thmbnl_shape[0]/wsi_at_level_shape[1]
    
    def read_region_mm(self, coords_mm, mm_per_pixel, patch_size_mm):
        if self.seg_thmbnl is None: self.seg_thmbnl = self.get_segmented_thumbnail(0)
        if self.multiplier is None: 
            self.multiplier = np.array(self._get_mm_to_thmbnl_coords_converter(mm_per_pixel))
            if self.verbose: print(f"=== Patch size scaled to {np.round(np.asarray(patch_size_mm)*self.multiplier).astype(int)} in thumbnail of size {self.seg_thmbnl.shape}")
        coords_in_thmbnl = np.round((np.asarray(coords_mm)*self.multiplier)).astype(int)
        patch_size_in_thmbnl = np.round(np.asarray(patch_size_mm)*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.maximum(patch_size_in_thmbnl, [1, 1])
        return self.seg_thmbnl[coords_in_thmbnl[1]:coords_in_thmbnl[1]+patch_size_in_thmbnl[1], coords_in_thmbnl[0]:coords_in_thmbnl[0]+patch_size_in_thmbnl[0]] ## NOTE flipped coords, because numpy expects height, width
       
    def _get_mm_to_thmbnl_coords_converter(self, mmpp):
        thmbnl_shape = self.seg_thmbnl.shape ## height, width
        wsi_at_level_shape = self._get_resolution(0)
        mpp_at_level = self._get_level_mpp(0)
        level_to_thmbnl = wsi_at_level_shape[1]/thmbnl_shape[0] #h/h
        mpp_at_thmbnl = mpp_at_level*level_to_thmbnl
        return 1/(mpp_at_thmbnl/1000) ### returns um per pixel in thumbnail
        
    def _get_resolution(self, level): # returns width, height
        return (self.wsi.levels[level].size.width, self.wsi.levels[level].size.height)
        
    def close(self): # NOTE: usually doesnt need to be called, as the underlying wsi is closed at the end of dataset.__init__ already
        self.wsi.close()
        
    def get_segmented_thumbnail(self, level=0):
        mpp = self._get_level_mpp(level)
        dims = self._get_resolution(level) # w, h
        size = max(dims) / (2/mpp) ## get thumbnail at 2mpp i.e. 5x
        thmbnl = self._get_thumbnail(size=size)
        thmbnl = np.array(thmbnl)<self.gs_threshold
        thmbnl_dims = thmbnl.shape # h, w
        x = dims[0]/thmbnl_dims[1]
        thmbnl_se = skimage.morphology.disk(round(self.se_rad_microns/(mpp*x)))
        #thmbnl = skimage.morphology.dilation(thmbnl, thmbnl_se)
        thmbnl = cv2.dilate(thmbnl.astype(np.uint8), thmbnl_se, iterations=1).astype(bool)
        return thmbnl
    
    def get_segmented_thumbnail_PIL(self, level=0, size=5000):
        return PIL.Image.fromarray(self.get_segmented_thumbnail(level, size))
    
class CleanedOtsu:
    """Scaled background removal using otsu+closing+opening
    Used to load binarized scaled patches of an image. 
    The image is scaled such that the longes side is 5000px usually equating to ~5x magnification.
    then this large thumbnail is binarized using the method described abouve and returned patches are scaled to this thumbnail, not original size
    """
    def __init__(self, wsi, se_rad_microns=64, verbose=False):
        self.wsi = wsi
        self.verbose
        thumbnail = self._get_thumbnail()
        w, h = thumbnail.size
        thumbnail = thumbnail.crop((w*0.1, h*0.1, w*0.9, h*0.9)) ## Crop edges to avoid bad thresholding
        self.gs_threshold = skimage.filters.threshold_otsu(np.array(thumbnail)) 
        if self.verbose: print(f"==== Otsu threshold is {self.gs_threshold}")
        self.se_rad_microns = se_rad_microns
        
        # None inits
        self.seg_thmbnl = None
        self.multiplier = None
        
    def _get_thumbnail(self, size=512):
        return self.wsi.read_thumbnail((size, size)).convert("L")
         
    def _get_level_mpp(self, level):
        mpp = self.wsi.levels[level].mpp.height
        return mpp
    
    ## Full scale implementation with per-patch binarization, very slow hence disabled.
    # def read_region_slow(self, coords, level, patch_size):
    #     if self.se is None: self.se = self._get_se(level)
    #     mask_patch = self.wsi.read_region(coords, level, patch_size).convert("L")
    #     mask_patch = np.array(mask_patch)<self.gs_threshold # binarize: bg is bright, fg is dark
    #     mask_patch = cv2.dilate(mask_patch.astype(np.uint8), self.se, iterations=1)#skimage.morphology.dilation(mask_patch, self.se)
    #     return mask_patch
    
    def read_region(self, coords, level, patch_size): ## passed coords are w, h
        if self.seg_thmbnl is None: self.seg_thmbnl = self.get_segmented_thumbnail(level)
        if self.multiplier is None: 
            self.multiplier = np.array(self._get_region_to_thmbnl_converter(level))
            if self.verbose: print(f"=== Patch size scaled to {np.round(patch_size*self.multiplier).astype(int)} in thumbnail of size {self.seg_thmbnl.shape}")
        coords_in_thmbnl = np.round(coords*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.round(patch_size*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.maximum(patch_size_in_thmbnl, [1, 1])
        return self.seg_thmbnl[coords_in_thmbnl[1]:coords_in_thmbnl[1]+patch_size_in_thmbnl[1], coords_in_thmbnl[0]:coords_in_thmbnl[0]+patch_size_in_thmbnl[0]] ## NOTE flipped coords, because numpy expects height, width
    
    def _get_region_to_thmbnl_converter(self, level): # returns width, height
        thmbnl_shape = self.seg_thmbnl.shape ## height, width
        wsi_at_level_shape = self._get_resolution(level)
        return thmbnl_shape[1]/wsi_at_level_shape[0], thmbnl_shape[0]/wsi_at_level_shape[1]
    
    def read_region_mm(self, coords_mm, mm_per_pixel, patch_size_mm):
        if self.seg_thmbnl is None: self.seg_thmbnl = self.get_segmented_thumbnail(0)
        if self.multiplier is None: 
            self.multiplier = np.array(self._get_mm_to_thmbnl_coords_converter(mm_per_pixel))
            if self.verbose: print(f"=== Patch size scaled to {np.round(np.asarray(patch_size_mm)*self.multiplier).astype(int)} in thumbnail of size {self.seg_thmbnl.shape}")
        coords_in_thmbnl = np.round((np.asarray(coords_mm)*self.multiplier)).astype(int)
        patch_size_in_thmbnl = np.round(np.asarray(patch_size_mm)*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.maximum(patch_size_in_thmbnl, [1, 1])
        return self.seg_thmbnl[coords_in_thmbnl[1]:coords_in_thmbnl[1]+patch_size_in_thmbnl[1], coords_in_thmbnl[0]:coords_in_thmbnl[0]+patch_size_in_thmbnl[0]] ## NOTE flipped coords, because numpy expects height, width
       
    def _get_mm_to_thmbnl_coords_converter(self, mmpp):
        thmbnl_shape = self.seg_thmbnl.shape ## height, width
        wsi_at_level_shape = self._get_resolution(0)
        mpp_at_level = self._get_level_mpp(0)
        level_to_thmbnl = wsi_at_level_shape[1]/thmbnl_shape[0] #h/h
        mpp_at_thmbnl = mpp_at_level*level_to_thmbnl
        return 1/(mpp_at_thmbnl/1000) ### returns um per pixel in thumbnail
        
    def _get_resolution(self, level): # returns width, height
        return (self.wsi.levels[level].size.width, self.wsi.levels[level].size.height)
        
    def close(self): # NOTE: usually doesnt need to be called, as the underlying wsi is closed at the end of dataset.__init__ already
        self.wsi.close()
        
    def get_segmented_thumbnail(self, level=0):
        mpp = self._get_level_mpp(level)
        dims = self._get_resolution(level) # w, h
        size = max(dims) / (2/mpp) ## get thumbnail at 2mpp i.e. 5x
        thmbnl = self._get_thumbnail(size=size)
        thmbnl = np.array(thmbnl)<self.gs_threshold
        thmbnl_dims = thmbnl.shape # h, w
        x = dims[0]/thmbnl_dims[1]
        thmbnl_se = skimage.morphology.disk(round(self.se_rad_microns/(mpp*x)))
        thmbnl = cv2.morphologyEx(thmbnl.astype(np.uint8), cv2.MORPH_CLOSE, thmbnl_se).astype(bool)
        thmbnl = cv2.morphologyEx(thmbnl.astype(np.uint8), cv2.MORPH_OPEN, thmbnl_se).astype(bool)
        return thmbnl
    
    def get_segmented_thumbnail_PIL(self, level=0, size=5000):
        return PIL.Image.fromarray(self.get_segmented_thumbnail(level, size))
    
def get_bg_rm_tool(method): # NOTE: to extend
    if method.lower() == 'otsu':
        return Otsu
    elif method.lower()  in ['dilated-otsu', 'dilated_otsu']:
        return DilatedOtsu
    elif method.lower()  in ['cleaned-otsu', 'cleaned_otsu']:
        return CleanedOtsu
    else: raise ValueError(f"{method} is not a supported BGRM tool")
