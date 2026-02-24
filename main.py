from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from BPTorch.utils import BPMeta
# pip install "BPTorch @ git+https://github.com/Bangulli/BPTorch"
if __name__ == '__main__':
    ds = BigPictureRepository('/mnt/nas6/data/BigPicture_CBIR/datasets', verbose=False, return_type='patch', **WsiDicomDataset.get_default_kwargs())
    ds.get_stats_plot("base")
    splits=ds.split('0.8-0.1-0.1', stratify=['species', 'organ', 'staining', 'diagnosis'], max_iter=5, fail="raise", tol=0.025)
    print("Obtained splits.")
    # for i, s in enumerate(splits):
    #     s.get_stats_plot(f"fold_{i}")
    #     s.prepare_patches()
    #     s.save(f"fold_{i}/BPR.json")