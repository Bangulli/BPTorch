import torch
def bptorch_collate(batch):
    images = []
    coordinates = []
    meta = []
    for item in batch:
       images.append(item['image'])
       coordinates.append(item['coordinates'])
       meta.append(item['metadata']) 
    res = {
        'image': torch.stack(images),
        'coordiantes': torch.stack(coordinates),
        'metadata': meta
    }
    return res