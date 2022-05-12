from src.models.DCVC_net_quantize import DCVC_net
import torch
from torchvision import transforms
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
import wandb
import tqdm
import torchnet.meter as meter

LAMBDA = 2048
BATCH_SIZE = 4
DATA_DIR1 = pathlib.Path('/home/pzy2/vimeo_septuplet_train/sequences')  #pathlib.Path('PATH')
DATA_DIR2 = pathlib.Path('/home/pzy2/vimeo_septuplet_test/sequences')
DEVICE = torch.device('cuda')


video_net = DCVC_net()

chpt = torch.load('/home/pzy2/model_dcvc_quality_3_psnr.pth')
video_net.load_state_dict(chpt, strict=False)
video_net = video_net.to(DEVICE)

expirement_name = f"experiment-variable-quantization-{LAMBDA}"

wandb.init(
    project="DCVC-Variable-Quantization", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=expirement_name, 
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "DCVC",
    "dataset": "Vimeo-90k",
    "epochs": 20,
    "resume": True
})

optimizer = torch.optim.Adam(video_net.parameters(), lr=wandb.config.learning_rate)
del chpt

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
        
        ref = plt.imread(path / f'im1.png')
        im = plt.imread(path /	f'im2.png')
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
    prefetch_factor=5
)


criterion = torch.nn.MSELoss()

def count_parameters(model):
    """Return number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(video_net)} trainable parameters')


def train_epoch(model, epoch, dl, optimizer, criterion, use_lambda=True):
    model.train()

    loss_meter = meter.AverageValueMeter()
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
        ref1 = x[:,0]
        #ref2 = x[:,1]
        im = x[:,1]
        preds_dict = model(ref1, im, compress_type='train_compress')
        preds = preds_dict['recon_image']
        bpp = preds_dict['bpp']
        mse_loss = criterion(preds, im)
        mse_ls = mse_loss.item()
        avg_mse = mse_meter.value()[0]
        if not np.isnan(avg_mse) and avg_mse < 0.001 and mse_ls > 0.1:
            # waits until average loss is pretty low
            print('REALLY BAD LOSS', mse_ls)
            wandb.alert(
                title="High Loss", 
                text=f"Loss {mse_ls} is way above the threshold"
            )
            return None, True, x
        if use_lambda:
            loss = mse_loss * LAMBDA + bpp
        else:
            loss = mse_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        ls = loss.item()
        loss_meter.add(ls)
        mse_meter.add(mse_ls)
        bpp_meter.add(bpp.item())
        bpp_mv_y_meter.add(preds_dict['bpp_mv_y'].item())
        bpp_mv_z_meter.add(preds_dict['bpp_mv_z'].item())
        bpp_y_meter.add(preds_dict['bpp_y'].item())
        bpp_z_meter.add(preds_dict['bpp_z'].item())
        if i % 1 == 0:
            avg_psnr = -10.0*np.log10(mse_meter.value()[0])
            wandb.log(
                {
                    'train/epoch': epoch,
                    'train/batch_loss': ls,
                    'train/avg_loss': loss_meter.value()[0],
                    'train/avg_mse_loss': mse_meter.value()[0],
                    'train/avg_bpp': bpp_meter.value()[0],
                    'train/avg_bpp_mv_y': bpp_mv_y_meter.value()[0],
                    'train/avg_bpp_mv_z': bpp_mv_z_meter.value()[0],
                    'train/avg_bpp_y': bpp_y_meter.value()[0],
                    'train/avg_bpp_z': bpp_z_meter.value()[0],
                    'train/avg_psnr': avg_psnr,
                }
            )
            ls = round(ls, 6)
            avg_bitrate = round(bpp_meter.value()[0], 6)
            avg_psnr = round(avg_psnr, 6)
            pbar.set_description(f'Avg PSNR/Bitrate, Batch Loss: {avg_psnr, avg_bitrate, ls}')
        # save every 
        if i % 4000 == 3999:
            print('Saving model with avg psnr', avg_psnr)
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                SAVE_FOLDER / f"dcvc_epoch={epoch}_batch_{i}_avg_psnr={avg_psnr}.pt",
            )

    return loss_meter.value()[0], False, None



SAVE_FOLDER = pathlib.Path(expirement_name)
os.makedirs(SAVE_FOLDER, exist_ok=True)

USE_LAMBDA = True
print("USE LAMBDA", USE_LAMBDA)
for i in range(1, wandb.config.epochs + 1):
    avg_loss, had_err, err_x = train_epoch(video_net, i, dl, optimizer, criterion, use_lambda=USE_LAMBDA)
    if had_err:
        print('Breaking out of train loop for debugging...')
        break
    torch.save(
        {'model': video_net.state_dict(), 'optimizer': optimizer.state_dict()},
        SAVE_FOLDER / f"dcvc_epoch={i}_avg_loss={avg_loss}.pt",
    )
    
wandb.finish()
