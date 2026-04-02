from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
import torchvision.transforms as T
from datetime import datetime
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"
if __name__ == '__main__':
    ## CREATE, SPLIT AND SAVE A REPOSITORY
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets', verbose=False, return_type='patch', wsidicomdataset_kwargs=WsiDicomDataset.get_default_kwargs())
    ds.get_stats_plot("/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/base")
    splits=ds.split('0.1-0.1-0.8', stratify=['species', 'organ', 'staining', 'diagnosis'], max_iter=5, fail="raise", tol=0.025)
    print("Obtained splits.")
    for i, s in enumerate(splits):
        s.get_stats_plot(f"/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_{i}")
        s.prepare_patches()
        s.save(f"/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_{i}/BPR.json")
    
    ## SETUP ARGS FOR DATALOADER
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = T.Resize(224) # your transfroms here
    
    # ## LOAD AND EXTRACT PATCHES FROM A REPOSITORY AND THEN POINT THE REPO TO TAKE THE EXTRACTED PATCHES AS SOURCE
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    ds.save_patches_as_images('subset', 'jpeg', randomize_subset=500)
    ds.source_precomputed_patches_from('subset')
    
    # ## USE A DATALOADER
    dl = DataLoader(ds, 256, collate_fn=bptorch_collate, shuffle=True) ## NOTE: Shuffling breaks the Repo's caching logic, leading to much slower data access as the random patches need to be accessed individually without a fallback on pre-loaded WsiDicom objects.
    start = datetime.now()
    for batch in dl:
        print(f"Loading batch took {datetime.now()-start}s")
        start = datetime.now()
        ## OPTIONAL: pprint(batch)