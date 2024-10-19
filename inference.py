from main.imports import *
from main.models.vit.base import VitBackBone
from main.models.vit.block import Block
from main.models.vit.embeddings import SimplePatchify3D
from main.models.vit.video import ViTVideo

torch.autograd.set_detect_anomaly(True)
#import matplotlib.pyplot as plt
import copy

import cv2
import dlib
import numpy as np
from imutils import face_utils

ddetector = dlib.get_frontal_face_detector()
# dlib  library path
dpredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



import os
import random
from collections import defaultdict


def get_paths():


    videos=[]
    
    videos_path = os.path.join('path_to_videos')
    if os.path.exists(videos_path):
        for file_name in os.listdir(videos_path):
                videos.append(os.path.join(videos_path, file_name))

  
    test_paths = videos
    test_labels =  [0]  * len(videos) ###   The labels of videos are 0 for real and 1 for fake

    
    return (test_paths, test_labels)



import logging
import random

from main.preprocessing import *
from main.preprocessing.audio import *
from main.preprocessing.video import *
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, filename='dataset_loader_log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class ProcessingPipeLine():
    def __init__(self,device):
        self.device = device
        print(f"Using device: {self.device}")
        self.loader = Loader(device,debug=False)

        self.cropper = FaceCropper(device,crop_shape=(224,224),normalised_input=False)
        self.sampler = SimpleSampler('uniform',num_frames=16)
        self.vnorm = VNorm()
        self.stacker = Stacker()

    @torch.no_grad()
    def process(self,links):

        logger.info(f"Processing {len(links)} videos")

        data = self.loader.process({'video_links': links})
        data = self.sampler.process(data)
        data = self.cropper.process(data)
        data = self.vnorm.process(data)
        data = self.stacker.process(data)
        return data




class MyDataset(Dataset):
    def __init__(self,video_paths, labels, pipeline):
        self.video_paths, self.labels = video_paths, labels
        self.preprocessing_pipeline = pipeline
        self.shuffle()
    
    def __len__(self):
        return len(self.video_paths)
    
    def shuffle(self):
        c = list(zip(self.video_paths, self.labels))
        random.shuffle(c)
        self.video_paths, self.labels = zip(*c)
    
    def __getitem__(self, idx):
        path, label = self.video_paths[idx], self.labels[idx]
        data = self.preprocessing_pipeline.process([path])
        data['labels'] = torch.tensor(label).to('cpu')
        logger.info(f"Loaded video {path} with label {label}")

        return data

    
    def _slice(self,start,end):
        self_copy = copy.deepcopy(self)
        self_copy.video_paths = self.video_paths[start:end]
        self_copy.labels = self.labels[start:end]
        return self_copy
    
(test_paths, test_labels) = get_paths()
processing_pipeline = ProcessingPipeLine(device='cuda')
import joblib


val_dataset = MyDataset(test_paths, test_labels, processing_pipeline)
joblib.dump({
            'val': val_dataset},
                'dataset.joblib')

def collate_fn(batch):
    new_data = {'video': [], 'labels': []}
    for data in batch:
        new_data['video'].append(data['video'])
        new_data['labels'].append(data['labels'])
    new_data['video'] = torch.concatenate(new_data['video'],axis=0)
    new_data['labels'] = torch.stack(new_data['labels'])
    return new_data




val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                 collate_fn=collate_fn)






# from pytorch_lightning import Trainer, LightningModule
# from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import f1_score

# API key
os.environ['WANDB_API_KEY'] = "Wandb API"
# os.environ['WANDB_MODE'] = "disabled"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
wandb.init(project='DEEPFAKE_testing')



    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class Model(nn.Module):
    def __init__(self,norm_layer=nn.LayerNorm, embed_dim=768, use_mean_pooling=True, fc_drop_rate=0.0):
        super().__init__()
        self.video_encoder = nn.Sequential(ViTVideo(), nn.Linear(768, 768*2))
   

        self.video_encoder = ViTVideo()
        self.patcher = SimplePatchify3D(16, 224, 224, 16, 16, 2)
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.clf_head = nn.Sequential( nn.Linear(768,2))

    def forward(self, videos):
        # videos: B * 3 * 16 * 224 * 224
        video_encodings = self.video_encoder(videos) 
        x = self.norm(video_encodings)


        x =  self.fc_norm(x.mean(1))
        logits = self.clf_head(self.fc_dropout(x))

        return logits
    
    def calculate_clf_loss(self, logits,gt):
    
        gt = gt.long()
        focal_loss = FocalLoss(alpha=1, gamma=2)
        return focal_loss(logits, gt)

    
    def calculate_au_loss(self, au_maps_pred, au_maps):
        bce = F.binary_cross_entropy(au_maps_pred, au_maps) 
        hb = 10*F.huber_loss(au_maps_pred, au_maps)
        dice = self.dice_loss(au_maps_pred, au_maps)
        return bce + hb + dice, {'bce': bce, 'hb': hb, 'dice': dice}
    
    def training_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['labels']
        #au_maps = process_image_batch(videos)

        logits= self(videos)
        clf_loss = self.calculate_clf_loss(logits, labels)
    
        loss = clf_loss
        to_log = {'train_loss': loss, 'train_clf_loss': clf_loss}
        return loss, to_log
    
 
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['labels']
        #au_maps = process_image_batch(videos)

        with torch.no_grad():
            logits = self(videos)
        

        clf_loss = self.calculate_clf_loss(logits, labels)
        loss = clf_loss
        to_log = {'val_loss': loss, 'val_clf_loss': clf_loss}

        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels) / len(labels)
        f1 = f1_score(labels.cpu(), preds.cpu())
        to_log['val_acc'] = acc
        to_log['val_f1'] = f1
        to_log['TP'] = torch.sum((preds == 1) & (labels == 1)).float()
        to_log['TN'] = torch.sum((preds == 0) & (labels == 0)).float()
        to_log['FP'] = torch.sum((preds == 1) & (labels == 0)).float()
        to_log['FN'] = torch.sum((preds == 0) & (labels == 1)).float()
        return loss, to_log



    def log_images(self, videos, au_maps_pred, au_maps):
        # videos: B * 3 * 16 * 224 * 224
        # au_maps_pred: B * 16 * 16 * 224 * 224
        # au_maps: B * 16 * 16 * 224 * 224
        frame = videos[0, :, 0, :, :].detach().cpu().permute(1, 2, 0)
        au_maps = au_maps[0, :, 0, :, :]
        au_maps_pred = au_maps_pred[0, :, 0, :, :]
        images = [frame] 
        captions = ['frame']
        for i in range(5):
            images.append(au_maps[i])
            images.append(au_maps_pred[i])
            captions.append(f'au_map_{i}')
            captions.append(f'au_map_pred_{i}')
        images = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]
        wandb.log({"sample_images_video": [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]})
    
    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=1e-5)

model = Model()

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm

wandb.watch(model)

val_every_n_steps = 1 
grad_acc_steps = 20
epochs = 100
mixed_precision = False
# Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'
model.to(device)
optimizer = model.configure_optimizers()
scaler = GradScaler()

# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, verbose=True)

best_val_loss = float('inf')
if os.path.exists('weights.pth'):
    model.load_state_dict(torch.load('weights.pth'))
    print('Loaded best model')

def accumulate_logs(logs, to_log, acc_steps):
    
    to_log = {k: v.item() / acc_steps for k, v in to_log.items()}
    for k in to_log.keys():
        if k in logs:
            logs[k] += to_log[k]
        else:
            logs[k] = to_log[k]
    return logs

def accumulate_logs_val(logs, to_log):
    to_log = {k: v.item()  for k, v in to_log.items()}
    for k in to_log.keys():
        if k in logs:
            logs[k] += to_log[k]
        else:
            logs[k] = to_log[k]
    return logs

# Testing
for epoch in range(epochs):
    #model.train()
    train_logs = {'epoch': epoch}
    val_dataloader_tqdm = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
    
    for step, batch in enumerate(val_dataloader_tqdm):
        
        flag='test'
                
        if ((step+1) % val_every_n_steps == 0) or (step == len(val_dataloader)-1 or flag=='test'):
            model.eval()
            #val_to_log = {'epoch': epoch, 'steps': step}\
            val_acc={}
            val_to_log={}
            val_loss_total = 0.0
            
            with tqdm(val_dataloader, desc='Validation', unit='batch', leave=False) as val_dataloader_tqdm:
                for val_step, val_batch in enumerate(val_dataloader_tqdm):
                    # Move val_batch to device
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    
                    with torch.no_grad():
                        val_loss, to_log_val = model.validation_step(val_batch, val_step)
                        val_loss_total += val_loss.item()
                        val_acc = accumulate_logs_val(val_acc, to_log_val)
                    

            
            precision = val_acc['TP'] / (val_acc['TP'] + val_acc['FP'] + 1e-8)
            recall    = val_acc['TP'] / (val_acc['TP'] + val_acc['FN'] + 1e-8)
            val_to_log['f1_score'] = (2 * precision * recall) /(precision+recall + 1e-8)
            val_to_log['val_acc'] = (val_acc['TP'] + val_acc['TN']) / (val_acc['TP'] + val_acc['TN'] + val_acc['FP'] + val_acc['FN'] )

            for k, v in val_acc.items():
                 val_to_log[k] = v / len(val_dataloader)

            avg_val_loss = val_loss_total / len(val_dataloader)
            val_to_log['val_loss'] = avg_val_loss
            wandb.log(val_to_log)
            

            
            torch.cuda.empty_cache()

    
    train_dataloader_tqdm.close()
