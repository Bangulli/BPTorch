from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from torch.utils.data import DataLoader
from BPTorch.utils import bptorch_collate
from pprint import pprint
import torchvision.transforms as T
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"
if __name__ == '__main__':
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets', verbose=False, return_type='patch', wsidicomdataset_kwargs=WsiDicomDataset.get_default_kwargs())
    ds.get_stats_plot("/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/base")
    splits=ds.split('0.1-0.1-0.8', stratify=['species', 'organ', 'staining', 'diagnosis'], max_iter=5, fail="raise", tol=0.025)
    print("Obtained splits.")
    for i, s in enumerate(splits):
        s.get_stats_plot(f"/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_{i}")
        s.prepare_patches()
        s.save(f"/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_{i}/BPR.json")
    kwargs = WsiDicomDataset.get_default_kwargs()
    kwargs['transforms'] = T.Resize(224)
    kwargs['precomputed'] = True
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/fold_0/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False)
    print(f"Dataset contains {len(ds)} foreground patches")
    patch = ds[213]
    pprint(patch)
    
    dl = DataLoader(ds, 4, collate_fn=bptorch_collate, shuffle=True)
    for batch in dl:
        pprint(batch)
   