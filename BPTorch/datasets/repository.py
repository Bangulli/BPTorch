### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union, Tuple, Callable
import time, glob, tqdm, re, pprint, random, warnings, json, copy
import logging ## avoid weird wsidicom logspam: WARNING:root:Orientation [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0] is not orthogonal with equal lengths with column rotated 90 deg from row
logging.getLogger().setLevel(logging.ERROR)
### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from wsidicom import WsiDicom
### Internal Imports ###
from BPTorch.utils.metadata import BPMeta
from BPTorch.datasets.wsi import WsiDicomDataset
########################

class BigPictureRepository(tc.utils.data.Dataset):
    """
    A slide level dataset encapsulating all WSIs present in a directory
    """
    def __init__(self, path: Union[str, Path], return_type='patch', images: list=None, verbose=False, load=False, wsidicomdataset_kwargs=None):
        """Constructor.

        Args:
            path (Union[str, Path]): The path to the locally stored bigpicture data or the path to the .json file describing a BigPictureRepository object
            return_type (str, optional): What data to return. If 'wsi' = WsiDicomDataset objects, if 'patch' = Patches per image provided by WsiDicomDataset according to kwargs, if 'path' = path to images. Defaults to 'patch'.
            images (list, optional): A list of image paths from which to build the BigPictureRepository object. If None will infer them from path. Defaults to None.
            verbose (bool, optional): Verbosity. Defaults to True.
            load (bool, optional): Whether to treat path as a path to a saved BigPictureRepository object or not. Defaults to False.

        Returns:
            BigPictureRepository: the instance
        """
        if not load:
            self.kwargs = wsidicomdataset_kwargs
            self.kwargs['verbose'] = verbose
            assert return_type.lower() in ['wsi', 'patch', 'path'], 'Invalid return type requested, must be [path, patch, wsi]'
            self.return_type = return_type.lower()
            self.stats = None
            self.verbose = verbose
            self.root = path if type(path) is Path else Path(path)
            self.meta = {}
            self._BP_datasets = self._find_all_BP_datasets()
            self.patches = {}
            self.patch_idx = []
            self.imgs = images if images is not None else self._find_all_images()
            self.nomenclature = {
                'any': [],
                'short2str': {},
                'str2short': {}
            }
            self.metadata_fields = BPMeta.get_supported_fields()
            self.unusable_images = []
            self.patches_prepared = False
        else: 
            self.verbose = verbose
            self._load(path)
    
    def __len__(self):
        if self.return_type in ['wsi', 'path']:return len(self.imgs)
        elif self.return_type=='patch': return len(self.patch_idx)
    
    def __getitem__(self, idx):
        if self.return_type=='wsi':
            modkwargs = copy.deepcopy(self.kwargs)
            if 'metadata' in modkwargs.keys(): del modkwargs['metadata']
            return WsiDicomDataset(self.imgs[idx], metadata=self.meta[self.imgs[idx].parent.parent.name][self.imgs[idx].name], **modkwargs)
        
        elif self.return_type=='path': return self.imgs[idx]
        
        elif self.return_type=='patch':
            assert self.patches_prepared, "Return type is patch but data was attempted to be accessed before the patches were prepared."
            modkwargs = copy.deepcopy(self.kwargs)
            if 'metadata' in modkwargs.keys(): del modkwargs['metadata']
            modkwargs['precomputed'] = True
            key, p_idx, i_idx = self.patch_idx[idx]
            corners, coords = self.patches[key]['corners'][p_idx], self.patches[key]['coords'][p_idx]
            wsi = WsiDicomDataset(self.imgs[i_idx], metadata=self.meta[self.imgs[idx].parent.parent.name][self.imgs[idx].name], **modkwargs)
            return wsi[(corners, coords)]
            
    def prepare_patches(self):
        if self.patches_prepared: return
        assert self.return_type=='patch', f'It is not required to prepare patches, if the return type is {self.return_type}'
        iterator = self.imgs if self.verbose else tqdm.tqdm(self.imgs, desc=f'Preparing patches for {len(self.imgs)} WSIs')
        start_idx = 0
        for i, img in enumerate(iterator):
            try:
                wsi = WsiDicomDataset(img, **self.kwargs)
                n_patches = len(wsi)
                ### convert to json serializable
                coords = []
                for i in range(wsi.coordinates.shape[0]):
                    coords.append((int(wsi.coordinates[i, 0]), int(wsi.coordinates[i, 1])))
                corners = []
                for i in range(wsi.upper_left_corners.shape[0]):
                    corners.append((int(wsi.upper_left_corners[i, 0]), int(wsi.upper_left_corners[i, 1])))
                
                ## make dict
                cur_data = {
                    'lower': start_idx,
                    'upper': start_idx+n_patches,
                    'coords': coords,
                    'corners': corners
                }
                self.patches[start_idx]=cur_data
                for j in range(n_patches):
                    self.patch_idx.append((start_idx, j, i))
                start_idx+=n_patches
            except:
                if self.verbose: print(f"Image {img} cannot be patched and will be removed from the dataset")
                self.imgs.remove(img)
                self.unusable_images.append(img)
            
        self.patches_prepared = True
        
    def get_stats(self):
        """Function to generate a dictionary of the metadata distribution in the repo
        """
        if self.stats is None:
            self.stats, _, _ = self._find_all_beings()
        else: return self.stats
    
    def get_stats_plot(self, path: Union[str, Path]):
        """Function to generate plots of the metadata distribution in the repo
        Args:
            path (Union[str, Path]): Where to store the plot(s)
        """
        _ = self.get_stats()
        if not os.path.exists(path): os.mkdir(path)
        for field in self.metadata_fields:
            plt.pie(self.stats[field].values(), labels=self.stats[field].keys())
            plt.savefig(Path(path)/f"{field}.png")
            plt.close()
            plt.clf()
    
    def split(self, folds='0.9-0.1', stratify=None, eval_strategy = 'average', random_seed=42, tol=0.05, fail='raise', max_iter=100) -> list:
        """Data splitting function. Supports random and stratified splitting according to the fields provided by BPMeta.get_supported_fields().

        Args:
            folds (str, optional): The split configuration, must be a string of floats separated by '-'. Each float represents one fold and the fraction of data in the fold, the sum of all folds must be 1. Defaults to '0.9-0.1'.
            stratify (list, optional): The stratification configuration. Must be a None for no stratification or a list of fields  BPMeta.get_supported_fields() representing the fields to stratify for. Defaults to None.
            eval_strategy (str, optional): The evaluation strategy if 'average' check for the tolerance in the average of the deviations of the stratification fields, if 'strict' will check for each field individually. Defaults to 'average'.
            random_seed (int, optional): The random seed. Defaults to 42.
            tol (float, optional): The toleranche criterion to uphold. The splits have to uphold the metadata distribution for within this margin. Defaults to 0.05.
            fail (str, optional): Behaviour if stratification fails to uphold the tolerance after may_iter is reached. 'raise' = raise error, 'ignore' = quietly return best split, 'warn' = warn and return best split. Defaults to 'raise'.
            max_iter (int, optional): Maximum number of iterations. This function stratifies by iteratively permuting the beings in the splits to uphold the criteria. Defaults to 100.

        Returns:
            list: A list of BigPictureRepository object, the folds.
        """
        assert eval_strategy in ['average', 'strict'], f"{eval_strategy} is not a valid evaluation strategy."
        assert fail in ['ignore', 'raise', 'warn'], f"{fail} is not a valid failure behaviour."
        if any(stratify): 
            assert type(stratify) == list, "stratify needs to be a list if not none"
            assert all([f in self.metadata_fields for f in stratify]), f"Cannot stratify for metadata fields that are not supported: {[f for f in stratify if not f in self.metadata_fields]}."
        folds = [float(f) for f in folds.split('-')]
        if len(folds)==1: return [self] #return self if only one fold is requested
        assert sum(folds)==1, "The sum of all folds must be 1"
        
        ## get list of beings
        # abs coutns in set, abs in being, being2image map
        self.stats, beings, being2images = self._find_all_beings()
        
        ## randomize being order
        being_ids = sorted(list(beings.keys())) # sorted for consistency
        if random_seed is not None: random.seed(random_seed)
        random.shuffle(being_ids)
        
        ## init split storage
        fold_assignments = {}
        for i in range(len(folds)):
            fold_assignments[i] = []
        
        ## init split assignments
        fold_ids = list(fold_assignments.keys())
        if random_seed is not None: random.seed(random_seed)
        for b in being_ids:
            to_fold = random.choices(fold_ids, folds, k=1)[0]
            fold_assignments[to_fold].append(b)
        
        ## Return if no stratification is required
        if not stratify:
            splits = []
            for k, v in fold_assignments.items():
                fold_images = []
                for b in v:
                    fold_images += being2images[b]
                splits.append(BigPictureRepository(path=self.root, images=fold_images, verbose=self.verbose, return_type=self.return_type, wsidicomdataset_kwargs=self.kwargs))
            return splits
        
        best_split = copy.deepcopy(fold_assignments)
        best_split_score = None
        
        ### STRATIFICATION LOOP
        for it in tqdm.tqdm(range(max_iter), desc='Optimizing'):
            fold_stats, image_counts = self._get_fold_stats(fold_assignments, beings, being2images) ## absolute
            
            ## evaluation
            evaluator, disc = self._check_split_strat_tol(fold_stats, self.stats, tol, image_counts, stratify)
            
            ## best split updating
            split_score = sum([v for k,v in disc.items()])/len(disc)
            if best_split_score is None: best_split_score=split_score
            elif split_score < best_split_score:
                best_split_score = split_score
                best_split = copy.deepcopy(fold_assignments)
                
            ## exit
            exit = (sum([v for k,v in disc.items()])/len(disc) < tol) if eval_strategy=='average' else all([v for k, v in evaluator.items() if k in stratify])
            if exit:
                splits = []
                for k, v in best_split.items():
                    fold_images = []
                    for b in v:
                        fold_images += being2images[b]
                    splits.append(BigPictureRepository(path=self.root, images=fold_images, verbose=self.verbose, return_type=self.return_type, wsidicomdataset_kwargs=self.kwargs))
                return splits 
            
            ## attempt optimization
            # 1st Find which variables need to shift in which direction. As absolute image counts
            fold_discrepancies = []
            for i, fold in enumerate(fold_stats):
                discrepancies = {}
                for field in stratify:
                    discrepancies[field] = {}
                    for k, v in self.stats[field].items():
                        disc = round(fold[field][k] - (v * (image_counts[i]/len(self.imgs)))) ## the ref count scaled into the fold count
                        discrepancies[field][k] = disc 
                fold_discrepancies.append(discrepancies)
                    
            # 2nd Find the most suitable permutation for each fold. Per iteration one permutation per fold is performed.
            permutations = {}
            involved_beings = []
            for i1, fold in fold_assignments.items():                
                # 2.1 Find least suitable beings in folds
                cur_worst_being=None
                cur_worst_score=np.inf
                for b1 in fold:
                    if b1 in involved_beings: continue
                    score = 0
                    for field in stratify:
                        for k, v in beings[b1][field].items():
                            score += abs(fold_discrepancies[i1][field][k]-v)
                    if cur_worst_being is None: cur_worst_being = b1; cur_worst_score = score; continue
                    if cur_worst_score > score: cur_worst_being = b1; cur_worst_score = score
                    
                involved_beings.append(cur_worst_being)
                    
                # 2.2 Find most suitable being in other folds
                cur_best_fold=None
                cur_best_being=None
                cur_best_score=np.inf
                for i2, fold2 in fold_assignments.items():
                    if i1==i2: continue
                    for b2 in fold2:
                        if b2 in involved_beings: continue
                        score = 0
                        for field in stratify:
                            for k, v in beings[b2][field].items():
                                score += abs(fold_discrepancies[i1][field][k]+v) ## normalize by n_images
                        if cur_best_being is None:cur_best_fold = i2; cur_best_being = b2; cur_best_score = score; continue
                        if cur_best_score > score:cur_best_fold = i2; cur_best_being = b2; cur_best_score = score
                        
                involved_beings.append(cur_best_being) 
                
                # 2.3 Store
                permutations[i1] = (cur_worst_being, cur_best_fold, cur_best_being)
                
                ## debug observation
                # printout = {}
                # for field in stratify:
                #     printout[field] = {}
                #     for k in fold_discrepancies[i1][field].keys():
                #         printout[field][k] = (fold_discrepancies[i1][field][k], beings[cur_worst_being][field][k], beings[cur_best_being][field][k])
                # pprint.pprint(printout)

            # 2.4 apply
            for i, (to_rm, to_fold, to_replace) in permutations.items():
                fold_assignments[i].remove(to_rm)
                fold_assignments[i].append(to_replace)
                fold_assignments[to_fold].remove(to_replace)
                fold_assignments[to_fold].append(to_rm)
            
        ## handle failure behaviour and exit
        if fail == 'raise': 
            printout = {}
            for i,fd in enumerate(fold_discrepancies):
                for field in stratify:
                    if i == 0: printout[field]={}
                    for k, v in fd[field].items():
                        if i == 0: printout[field][k] = [v]
                        else: printout[field][k].append(v)   
            pprint.pprint(printout)  
            raise RuntimeError(f"Failed to split data into {folds} while maintaining tolerance {tol*100}%. Consider loosening the tolerance criterion. The best split found in the process achieved a mean deviation of {best_split_score}.")
        elif fail == 'warn':
            warnings.warn(f"Failed to split data into {folds} while maintaining tolerance {tol*100}%. Returning the best split with a mean deviation of {best_split_score}.")
        
        ## exit
        splits = []
        for k, v in best_split.items():
            fold_images = []
            for b in v:
                fold_images += being2images[b]
            splits.append(BigPictureRepository(path=self.root, images=fold_images, verbose=self.verbose, return_type=self.return_type, wsidicomdataset_kwargs=self.kwargs))
        return splits
       
    def save(self, path):
        full_dict = {
            "kwargs":self.kwargs,
            "return_type":self.return_type,
            "root": str(self.root),
            "_BP_datasets": [str(ds) for ds in self._BP_datasets],
            "patches": self.patches,
            "patch_idx": self.patch_idx,
            "imgs": [str(i) for i in self.imgs],
            "nomenclature": self.nomenclature,
            "stats": self.stats,
            "unusable_images": [str(i) for i in self.unusable_images],
            "patches_prepared": self.patches_prepared
        }
        with open(path, "w") as f:
            json.dump(full_dict, f, indent=4)
    
    def _load(self, path):
        with open(path, "r") as f:
            full_dict = json.load(f)
        self.kwargs = full_dict['kwargs']
        self.kwargs['verbose'] = self.verbose
        self.return_type = full_dict['return_type']
        self.root = Path(full_dict['root'])
        self._BP_datasets = [Path(ds) for ds in full_dict['_BP_datasets']]
        self.meta = {ds.name: BPMeta(ds) for ds in self._BP_datasets}
        self.patches =  {int(k):v for k , v in full_dict['patches'].items()}
        self.patch_idx = full_dict['patch_idx']
        self.imgs = [Path(i) for i in full_dict['imgs']]
        self.nomenclature = full_dict['nomenclature']
        self.stats = full_dict['stats']
        self.unusable_images = [Path(i) for i in full_dict['unusable_images']]
        self.patches_prepared = full_dict['patches_prepared']
         
    def _check_split_strat_tol(self, fold_stats, ref_stat, tol, image_counts, stratify): ## func to check if the split fulfill the stratification within tolerance
        folds = {'staining':True, 'diagnosis':True, 'organ':True, 'species':True}
        disc = {'staining':0, 'diagnosis':0, 'organ':0, 'species':0}
        for cur_stats, cur_count in zip(fold_stats, image_counts):
            for field in stratify:
                keys = list(ref_stat[field].keys())
                for key in keys:                
                    disc[field] += abs(cur_stats[field][key]/cur_count - ref_stat[field][key]/len(self.imgs))
                disc[field]/=len(keys) ## average discrepancy within field
                folds[field] = disc[field]<tol
        # printout = {}
        # for k, v in folds.items():
        #     printout[k] = (disc[k], v)
        # pprint.pprint(printout)
        return folds, disc
    
    def _get_fold_stats(self, folds, beings, being2img):
        fold_stats = []
        image_counts = []
        for _, fold in folds.items():
            cur_fold = None
            n_images_in_fold = sum(len(being2img[b]) for b in fold)
            for f in fold:
                if cur_fold is None: cur_fold=copy.deepcopy(beings[f]); continue
                for field in self.metadata_fields:
                    for k, v in beings[f][field].items():
                        if k not in list(cur_fold[field].keys()):
                            cur_fold[field][k] = v
                        else: cur_fold[field][k] += v
            fold_stats.append({k: {kk:vv for kk, vv in v.items()} for k, v in cur_fold.items()})
            image_counts.append(n_images_in_fold)
        return fold_stats, image_counts
            
    def _find_all_BP_datasets(self) -> list:
        """Find all BigPicture datasets in the repository

        Returns:
            list: A list of all present BP datasets
        """
        hits = []
        iterator = self.root.rglob("IMAGES") if self.verbose else tqdm.tqdm(self.root.rglob("IMAGES"), desc='Finding datasets')
        for images_dir in iterator:
            parent = images_dir.parent
            try:
                meta = BPMeta(parent)
                hits.append(parent)
                self.meta[parent.name]=meta
                if self.verbose: print(f"@BPR Found valid dataset {parent.name}")
            except: pass ## skip dataset if parsing fails
        return hits
    
    def _find_all_images(self) -> list:
        """Generate a list of all images in the root directory

        Returns:
            list: list of image paths (pathling.Path objects)
        """
        available_images = []
        iterator = self._BP_datasets if self.verbose else tqdm.tqdm(self._BP_datasets, desc='Finding images')
        for ds in iterator:
            for img in (ds/'IMAGES').iterdir():
                available_images.append(img)
        if self.verbose: print(f"@BPR BigPicture Repository at location {self.root} contains {len(available_images)} images.")
        return available_images
    
    def _find_all_beings(self) -> tuple:
        beings = {}
        total_stats = {
            'staining':{},
            'diagnosis':{},
            'organ':{},
            'species':{}
        }        
        being_images = {}
        ds = {}
        
        ## iter images
        for img in self.imgs:
            cur_ds = img.parent.parent.name
            id = img.name
            meta = self.meta[cur_ds][id]
            if meta['being'] not in list(being_images.keys()):
                being_images[meta['being']]=[img]
                ds[meta['being']]=cur_ds
            else:
                being_images[meta['being']].append(img)
            
        ## iter beings
        for b, img in being_images.items():
            cur_vars = {
                'staining':{},
                'diagnosis':{},
                'organ':{},
                'species':{}
            }
            
            ## iter images
            for i in img:
                cur_meta = self.meta[ds[b]][i.name]
                
                ## iter meta
                for var in self.metadata_fields:
                    val = cur_meta[var]
                    val, _ = self._nomenclature_interface(val)
                    if val not in list(total_stats[var].keys()):
                        total_stats[var][val] = 1
                    else: total_stats[var][val] += 1
                    
                    if val not in list(cur_vars[var].keys()):
                        cur_vars[var][val] = 1
                    else: cur_vars[var][val] += 1
            beings[b] = cur_vars
                
        ## add non existing samples to the beings
        for b in list(beings.keys()):
            for var in self.metadata_fields:
                ref = list(beings[b][var].keys())
                for total in list(total_stats[var].keys()):
                    if total not in ref:
                        beings[b][var][total]=0
                    else:
                        beings[b][var][total]=beings[b][var][total]
                
        return total_stats, beings, being_images
    
    def _list2str(self, lst):
        """Convert a list to a name

        Args:
            lst (any): The list

        Returns:
            str: the name
        """
        try:
            string = str(re.sub(r"\s*\([^)]*\)", "", lst[0]).replace('(', '').replace(')', ''))
            for i in range(1, len(lst)):
                string+=f"+{re.sub(r"\s*\([^)]*\)", "", str(lst[i])).replace('(', '').replace(')', '')}"
            return string
        except: return ''
        
    def _shorten(self, nme):
        """Convert a name into a shorthand

        Args:
            nme (str): The name

        Returns:
            str: The shorthand
        """
        try:
            shorthand = ''
            for section in nme.split('+'):
                for subword in section.strip().split(' '):
                    shorthand += subword[0]
                shorthand += '+'
            if not shorthand.endswith('+'): return shorthand.replace('(', '').replace(')', '')
            else: return shorthand.removesuffix('+').replace('(', '').replace(')', '')
        except: return ''   
    
    def _nomenclature_interface(self, query):
        """Nomenclature interface. Give any string or list and get the string and shorhand for it

        Args:
            key (any): the query

        Returns:
            tuple (str, str): The short and long name of the query
        """
        if not type(query)==str:
            string = self._list2str(query)
            short = self._shorten(string)
        else:
            try: 
                string = self.nomenclature['short2str'][query]
                short = query
            except: 
                string = query
                short = self._shorten(string)
                
        if not string in self.nomenclature['any']:
            self.nomenclature['any'].append(string)
            self.nomenclature['any'].append(short)
            self.nomenclature['short2str'][short]=string
            self.nomenclature['str2short'][string]=short
                  
        try: return string, self.nomenclature['short2str'][string]
        except: return self.nomenclature['str2short'][string], string
        
