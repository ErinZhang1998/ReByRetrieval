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
import wandb


def eval_dataset(cnt, model, device, test_loader, loss_args, test_args):
    np.random.seed(129)
    loss_cat = []
    loss_obj = []

    scale_pred_list = []
    scale_info_list = []
    pixel_pred_list = []
    pixel_info_list = []
    
    with torch.no_grad():
        
        for batch_idx, data in enumerate(test_loader):
            image = data[0]
            scale_info = data[1]
            pixel_info = data[2]
            cat_info = data[3]
            id_info = data[4]

            model = model.to(device)
            image = image.to(device)

            img_embed, pose_pred = model(image)
            img_embed = img_embed.cpu()
            pose_pred = pose_pred.cpu()
            pose_pred = pose_pred.float().detach()
            scale_pred = pose_pred[:,:1]
            pixel_pred = pose_pred[:,1:]

            scale_pred_list.append(scale_pred)
            scale_info_list.append(scale_info)
            pixel_pred_list.append(pixel_pred)
            pixel_info_list.append(pixel_info)

            # Normalize the image embedding
            img_embed -= img_embed.min(1, keepdim=True)[0]
            img_embed /= img_embed.max(1, keepdim=True)[0]

            _,c_loss = triploss.batch_all_triplet_loss(labels=cat_info, embeddings=img_embed, margin=loss_args.margin, squared=False) #.cpu()
            _,o_loss = triploss.batch_all_triplet_loss(labels=id_info, embeddings=img_embed, margin=loss_args.margin, squared=False) #.cpu()

            if batch_idx % test_args.plot_gt_image_every == 0 and batch_idx != len(test_loader)-1:
                plot_image(pixel_pred, batch_idx, test_loader, scale_pred, scale_info)

            loss_cat.append(c_loss.item())
            loss_obj.append(o_loss.item())

            torch.cuda.empty_cache()
    
    scale_pred = torch.cat(scale_pred_list, dim=0)
    scale_info = torch.cat(scale_info_list, dim=0)
    pixel_pred = torch.cat(pixel_pred_list, dim=0)
    pixel_info = torch.cat(pixel_info_list, dim=0)
    
    # total_samples = len(test_loader.dataset)
    final_loss_cat = np.mean(loss_cat) * loss_args.lambda_cat
    final_loss_obj = np.mean(loss_obj) * loss_args.lambda_obj

    final_loss_s = torch.nn.MSELoss(reduction='mean')(scale_pred, scale_info).item() * loss_args.lambda_scale
    final_loss_p = (torch.nn.MSELoss(reduction='sum')(pixel_pred, pixel_info).item() / len(pixel_pred)) * loss_args.lambda_pixel

    
    return final_loss_cat, final_loss_obj, final_loss_s, final_loss_p


def plot_image(pixel_pred, batch_idx, test_loader,scale_pred, scale_info):
    batch_row = test_loader.batch_indices[batch_idx]
    j_idx = np.random.choice(len(batch_row),1)[0]
    j =  batch_row[j_idx]
    sample = test_loader.dataset.idx_to_data_dict[j]
    img = mpimg.imread(sample['rgb_all_path'])

    center = copy.deepcopy(sample['object_center'].reshape(-1,))
    center[0] = img.shape[1] - center[0]

    pixel_pred = pixel_pred[j_idx]
    center_pred0 = pixel_pred[0] * test_loader.dataset.img_w
    center_pred1 = pixel_pred[1] * test_loader.dataset.img_h

    fig = plt.figure(figsize=(10, 4))
    # ax1 = fig.add_subplot(111)
    plt.imshow(img)
    plt.scatter(center[0], center[1],   marker=".", c='b', s=30)
    plt.scatter(center_pred0, center_pred1,   marker=".", c='r', s=30)
    plt.title('gt: {}, pred: {}'.format(scale_info[j_idx].item(), scale_pred[j_idx].item()))
    # print(pixel_pred, center_pred0, center_pred1)
    
    wandb.log({'image/test_image_{}'.format(sample['sample_id']): plt})
    plt.close()
    # wandb.log({"image/test_image": [wandb.Image(image, caption=rgb_all_path)]})
    # wandb.log({"test/test_scale_pred": scale_pred[j].item(), 'test/test_scale_gt': scale_info[j].item()})
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    # image = ToTensor()(image) #.unsqueeze(0)
    # return image, j_idx, sample['rgb_all_path']

