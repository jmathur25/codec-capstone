#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


from src.models.DCVC_net import DCVC_net
import torch
from torchvision import transforms
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
import wandb
import tqdm
import torchnet.meter as meter


# In[3]:


LAMBDA = 2048
BATCH_SIZE = 4
#TODO: Patrick
DATA_DIR1 = pathlib.Path('/home/pzy2/vimeo_septuplet_train/sequences')
DATA_DIR2 = pathlib.Path('/home/pzy2/vimeo_septuplet_test/sequences')
#TODO: Patrick
OPT_FLOW_DIR = pathlib.Path('/data2/jatin/vimeo_septuplet/sequences/opt_flow.pth')
DEVICE = torch.device('cuda')
DEVICE


# In[4]:


video_net = DCVC_net()


# In[ ]:


exp_name = f'dcvc-reproduce_lamba={LAMBDA}-train-procedure'


# In[1]:


# load the good weights
video_net.opticFlow = torch.load(OPT_FLOW_DIR)
video_net = video_net.to(DEVICE)
optimizer = torch.optim.AdamW(video_net.parameters(), lr=1e-4)


# In[6]:


wandb.init(
    project="DCVC-B-frames", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=exp_name, 
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "DCVC",
    "dataset": "Vimeo-90k",
    "epochs": 20,
    "resume": True
})


# In[8]:


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir1, data_dir2, crop_size=256, make_b_cut=True, deterministic=False):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.crop_size = crop_size
        self.make_b_cut = make_b_cut
        self.deterministic = deterministic
        self.all_paths = []
        for seq in os.listdir(self.data_dir1):
            subseq = os.listdir(self.data_dir1 / seq)
            for s in subseq:
                self.all_paths.append(self.data_dir1 / seq / s)
        for seq in os.listdir(self.data_dir2):
            subseq = os.listdir(self.data_dir2 / seq)
            for s in subseq:
                self.all_paths.append(self.data_dir2 / seq / s)
        assert len(self.all_paths) == 91701
        
        self.transforms = torch.nn.Sequential(
            transforms.RandomCrop(crop_size)
        )
       
    def __getitem__(self, i):
        path = self.all_paths[i]
        imgs = []
        '''
        if self.make_b_cut:
            # load two reference frames and the B-frame in the middle
            #TODO: implement making this deterministic
            interval = np.random.randint(1, 4) # can be 1, 2, or 3
            ref1 = plt.imread(path / f'im{1}.png')
            ref2 = plt.imread(path / f'im{1 + interval*2}.png')
            # this is the B-frame, in the middle
            im = plt.imread(path / f'im{1 + interval}.png')
            imgs = [ref1, ref2, im]
        else:
        '''
        '''
        # load full sequence
        for i in range(1, 8):
            # should be between [0, 1]
            img = plt.imread(path / f'im{i}.png')
            imgs.append(img)
        '''
        idx = np.random.randint(1, 7)
        ref = plt.imread(path / f'im{idx}.png')
        im = plt.imread(path /	f'im{idx + 1}.png')
        imgs = [ref, im]        

        # plt.imread should make inputs in [0, 1] for us
        imgs = np.stack(imgs, axis=0)
        # bring RGB channels in front
        imgs = imgs.transpose(0, 3, 1, 2)
        return self.transforms(torch.FloatTensor(imgs))

    def __len__(self):
        return len(self.all_paths)

ds = VideoDataset(DATA_DIR1, DATA_DIR2)
dl = torch.utils.data.DataLoader(
    ds,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=6,
    prefetch_factor=5,
)


# In[9]:


mse_criterion = torch.nn.MSELoss()


# In[10]:


def count_parameters(model):
    """Return number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(video_net)} trainable parameters')


# In[11]:
def mse_to_psnr(mse):
    return -10.0*np.log10(mse)
    

def train_epoch(model, epoch, dl, optimizer, train_type):
    mse_criterion = torch.nn.MSELoss()
    model.train()
    
    if train_type == 'memc':
        mse_meter = meter.AverageValueMeter()
    elif train_type == 'memc_bpp':
        mse_meter = meter.AverageValueMeter()
        bpp_meter = meter.AverageValueMeter()
    elif train_type == 'recon':
        mse_meter = meter.AverageValueMeter()
    else:
        mse_meter = meter.AverageValueMeter()
        bpp_meter = meter.AverageValueMeter()
        bpp_mv_y_meter = meter.AverageValueMeter()
        bpp_mv_z_meter = meter.AverageValueMeter()
        bpp_y_meter = meter.AverageValueMeter()
        bpp_z_meter = meter.AverageValueMeter()
    
    optimizer.zero_grad()
    pbar = tqdm.tqdm(dl)
    for i, x in enumerate(pbar):
        x = x.to(DEVICE)
        ref = x[:,0]
        im = x[:,1]
        preds_dict = model(ref, im, compress_type='train_compress', train_type=train_type)
        if train_type == 'memc':
            mse = mse_criterion(preds_dict['pred'], im)
            mse.backward()
            
            # metrics
            mse_meter.add(mse.item())
            wandb_export = {
                'train/psnr': mse_to_psnr(mse_meter.value()[0]),
            }
        elif train_type == 'memc_bpp':
            mse = mse_criterion(preds_dict['pred'], im)
            bpp = preds_dict['mv_z_bpp'] + preds_dict['mv_y_bpp']
            loss = mse * LAMBDA + bpp
            loss.backward()
            
            # metrics
            mse_meter.add(mse.item())
            bpp_meter.add(bpp.item())
            wandb_export = {
                'train/psnr': mse_to_psnr(mse_meter.value()[0]),
                'train/bpp': bpp_meter.value()[0]
            }
        elif train_type == 'recon':
            mse = mse_criterion(preds_dict['recon_image'], im)
            mse.backward()
            
            # metrics
            mse_meter.add(mse.item())
            wandb_export = {
                'train/psnr_recon': mse_to_psnr(mse_meter.value()[0]),
            }
        else:
            mse = mse_criterion(preds_dict['recon_image'], im)
            loss = mse * LAMBDA + preds_dict['bpp']
            loss.backward()
            
            # metrics
            mse_meter.add(mse.item())
            bpp_meter.add(preds_dict['bpp'].item())
            bpp_mv_y_meter.add(preds_dict['bpp_mv_y'].item())
            bpp_mv_z_meter.add(preds_dict['bpp_mv_z'].item())
            bpp_y_meter.add(preds_dict['bpp_y'].item())
            bpp_z_meter.add(preds_dict['bpp_z'].item())
            
            wandb_export = {
                'train/psnr_recon': mse_to_psnr(mse_meter.value()[0]),
                'train/bpp': bpp_meter.value()[0],
                'train/bpp_mv_y': bpp_mv_y_meter.value()[0],
                'train/bpp_mv_z': bpp_mv_z_meter.value()[0],
                'train/bpp_y': bpp_y_meter.value()[0],
                'train/bpp_z': bpp_z_meter.value()[0],
            }
            

        optimizer.step()
        optimizer.zero_grad()
        
        if i % 1 == 0:
            wandb_export['train/epoch'] = epoch
            wandb_export['train/train_type'] = train_type
            wandb.log(wandb_export)
        # save every 
        if i % 4000 == 3999:
            print('Saving model')
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                SAVE_FOLDER / f"dcvc_epoch={epoch}_batch_{i}.pt",
            )
            


# In[12]:


SAVE_FOLDER = pathlib.Path(exp_name)
os.makedirs(SAVE_FOLDER, exist_ok=True)


# In[ ]:


def freeze_layer(l):
    for p in l.parameters():
        p.requires_grad = False

def unfreeze_layer(l):
    for p in l.parameters():
        p.requires_grad = True


# In[13]:


freeze_layer(video_net.opticFlow)

train_type = 'memc'
train_epoch(video_net, 1, dl, optimizer, train_type=train_type)
torch.save(
    {'model': video_net.state_dict(), 'optimizer': optimizer.state_dict()},
    SAVE_FOLDER / f"dcvc_{train_type}_epoch={i}.pt",
)

train_type = 'memc_bpp'
for i in range(1, 4):
    train_epoch(video_net, i, dl, optimizer, train_type=train_type)
    torch.save(
        {'model': video_net.state_dict(), 'optimizer': optimizer.state_dict()},
        SAVE_FOLDER / f"dcvc_{train_type}_epoch={i}.pt",
    )

# freeze mv layers
mv_layers = [
    video_net.bitEstimator_z_mv,
    video_net.mvpriorEncoder,
    video_net.mvpriorDecoder,
    video_net.mvDecoder_part1,
    video_net.mvDecoder_part2,
    video_net.auto_regressive_mv,
    video_net.entropy_parameters_mv,
]
print('Freezing MV layers')
for l in mv_layers:
    freeze_layer(l)

train_type = 'recon'
for i in range(4, 10):
    train_epoch(video_net, i, dl, optimizer, train_type=train_type)
    torch.save(
        {'model': video_net.state_dict(), 'optimizer': optimizer.state_dict()},
        SAVE_FOLDER / f"dcvc_{train_type}_epoch={i}.pt",
    )

train_type = 'full'
for i in range(10, 13):
    train_epoch(video_net, i, dl, optimizer, train_type=train_type)
    torch.save(
        {'model': video_net.state_dict(), 'optimizer': optimizer.state_dict()},
        SAVE_FOLDER / f"dcvc_{train_type}_epoch={i}.pt",
    )
    
# now unfreeze
print('Unfreezing MV layers and optical flow for end-to-end training')
for l in [video_net.opticFlow] + mv_layers:
    unfreeze_layer(l)
    
for i in range(13, 19):
    train_epoch(video_net, i, dl, optimizer, train_type=train_type)
    torch.save(
        {'model': video_net.state_dict(), 'optimizer': optimizer.state_dict()},
        SAVE_FOLDER / f"dcvc_{train_type}_epoch={i}.pt",
    )
    


# In[ ]:


wandb.finish()


# In[ ]:





# In[ ]:




