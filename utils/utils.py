import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb
import io 
from PIL import Image

class Struct:
    '''The recursive class for building and representing objects with.'''
    def __init__(self, obj):
        self.obj_dict = obj
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)
    
    def __getitem__(self, val):
        return self.__dict__[val]
    
    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))


def masked_image(img, mask):
    mask = mask > 0
    for i in range(3):
        img[:,:,i] = img[:,:,i] * mask
    return img

def plot_image_with_mask(epoch, cnt, idx_in_dataset, dataset, mask_name):
    idx_in_dataset = idx_in_dataset.cpu().numpy().astype(int)
    fig, axs = plt.subplots(1, 3, figsize=(20,20))  
    
    for i in range(3):
        sample = dataset.idx_to_data_dict[idx_in_dataset[i]]
        rgb_all = mpimg.imread(sample['rgb_all_path'])
        mask = mpimg.imread(sample['mask_path'])
        valid = np.sum(mask > 0)
        masked_img = masked_image(rgb_all, mask)
        axs[i].imshow(masked_img)
        axs[i].set_title('{}_{}'.format(sample['obj_cat'], sample['obj_id']))
    
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    final_img = Image.open(buf)
    log_key = 'triplet_pairs/{}_{}_{}_{}-{}-{}'.format(mask_name, epoch, cnt, idx_in_dataset[0], idx_in_dataset[1], idx_in_dataset[2])
    wandb.log({log_key: wandb.Image(final_img)}, step=cnt)
    plt.close()







