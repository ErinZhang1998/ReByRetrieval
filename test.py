import torch
from torch import linalg as LA
import loss.triplet_loss as triploss
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image
from torchvision.transforms import ToTensor
import copy

def eval_dataset(cnt, loss_args, model, device, test_loader, writer, test_args):
    loss_cat = []
    loss_obj = []
    loss_scale = []
    loss_pixel = []
    
    
    with torch.no_grad():
        
        for batch_idx, (image, scale_info, pixel_info, cat_info, id_info) in enumerate(test_loader):
            model = model.to(device)
            image = image.to(device)
            # scale_info = scale_info.to(device)
            # pixel_info = pixel_info.to(device)
            # cat_info = cat_info.to(device)
            # id_info = id_info.to(device)
            img_embed, pose_pred = model(image)
            img_embed = img_embed.cpu()
            pose_pred = pose_pred.cpu()
            pose_pred = pose_pred.float().detach()

            scale_pred = pose_pred[:,:1]
            pixel_pred = pose_pred[:,1:]
            img_embed -= img_embed.min(1, keepdim=True)[0]
            img_embed /= img_embed.max(1, keepdim=True)[0]

            c_loss = triploss.batch_all_triplet_loss(labels=cat_info, embeddings=img_embed, margin=loss_args.margin, squared=False) #.cpu()
            o_loss = triploss.batch_all_triplet_loss(labels=id_info, embeddings=img_embed, margin=loss_args.margin, squared=False) #.cpu()
            s_loss = torch.square(LA.norm(scale_pred - scale_info)).item() #.cpu().item()
            p_loss = torch.square(LA.norm(pixel_pred - pixel_info, axis=1)).numpy() #.cpu().numpy()

            if batch_idx % test_args.plot_gt_image_every == 0 and batch_idx != len(test_loader)-1:
                image, j = plot_image(pixel_pred, batch_idx, test_loader)
                writer.add_image('image/test_image', image, cnt)
                writer.add_text('text/test_scale', 'gt: {}, pred: {}'.format(scale_info[j].item(), scale_pred[j].item()), cnt)

            loss_cat.append(c_loss)
            loss_obj.append(o_loss)
            loss_scale.append(s_loss)
            loss_pixel.append(p_loss)
            torch.cuda.empty_cache()
    
    total_samples = len(test_loader.dataset)
    loss_scale = np.sum(loss_scale) / total_samples
    loss_pixel = np.sum(np.hstack(loss_pixel)) / total_samples
    return np.sum(loss_cat) / len(test_loader), np.sum(loss_obj) / len(test_loader), loss_scale, loss_pixel


def plot_image(pixel_pred, batch_idx, test_loader):
    batch_row = test_loader.batch_indices[batch_idx]
    j_idx = np.random.choice(len(batch_row),1)[0]
    j =  batch_row[j_idx]
    sample = test_loader.dataset.idx_to_data_dict[j]
    img = mpimg.imread(sample['rgb_all_path'])

    center = copy.deepcopy(sample['object_center'].reshape(-1,))
    center[0] = img.shape[1] - center[0]

    center_pred0 = pixel_pred[0] * test_loader.dataset.img_w
    center_pred1 = pixel_pred[1] * test_loader.dataset.img_h

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(111)
    ax1.imshow(img)
    ax1.scatter(center[0], center[1],   marker=".", c='b', s=30)
    ax1.scatter(center_pred0, center_pred1,   marker=".", c='r', s=30)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image) #.unsqueeze(0)
    return image, j_idx

