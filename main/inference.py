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



def AU_plot_ellipsoid(gray1,au,x,shapes):
    [x1,y1,x2,y2,w,h] = x
    
    att_map = np.zeros((gray1.shape[0],gray1.shape[1]))
    
    if au==0:
        (l_x1,l_y1) = (shapes[20])
        (r_x2,r_y2) = (shapes[23])
        cv2.ellipse(gray1,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==1:
        (l_x1,l_y1) = (shapes[18])
        (r_x2,r_y2) = (shapes[25])
        cv2.ellipse(gray1,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
       
    elif au==2:
        l_x,l_y = (shapes[19])
        r_x,r_y = (shapes[24])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==3:        
        (l_x1,_) = (shapes[36])
        (_,l_y1) = (shapes[38])
        (r_x1,_) = (shapes[39])
        (_,r_y1) = (shapes[41])
        (l_x2,_) = (shapes[42])
        (_,l_y2) = (shapes[44])
        (r_x2,_) = (shapes[45])
        (_,r_y2) = (shapes[47])
        cv2.ellipse(gray1,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1-l_y1)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1-l_y1)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(gray1,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2-l_y2)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2-l_y2)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)


        
    elif au==4:
        (l_x1,l_y1) = (shapes[41])
        (r_x1,r_y1) = (shapes[46])
        cv2.ellipse(gray1,(l_x1-round(w/10),l_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x1-round(w/10),r_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x1-round(w/10),l_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x1+round(w/10),r_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==5:
        (l_x1,_) = (shapes[36])
        (_,l_y1) = (shapes[38])
        (r_x1,_) = (shapes[39])
        (_,r_y1) = (shapes[41])
        (l_x2,_) = (shapes[42])
        (_,l_y2) = (shapes[44])
        (r_x2,_) = (shapes[45])
        (_,r_y2) = (shapes[47])
        cv2.ellipse(gray1,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==6:
        (l_x1,l_y1) = (shapes[29])
        (r_x1,r_y1) = (shapes[31])
        (r_x2,r_y2) = (shapes[35])
        
        cv2.ellipse(gray1,(int(r_x1),int(l_y1)),(20,20),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(int(r_x2),int(l_y1)),(20,20),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int(r_x1),int(l_y1)),(20,20),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(int(r_x2),int(l_y1)),(20,20),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==7:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[63])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==8:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
        cv2.ellipse(gray1,(l_x,l_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x,r_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x,l_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x,r_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==9:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
#         print(l_x,l_y,r_x,r_y)
        cv2.ellipse(gray1,(l_x,l_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x,r_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x,l_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x,r_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==10:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
        cv2.ellipse(gray1,(l_x,l_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x,r_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x,l_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x,r_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)


        
    elif au==11:
        l_x,l_y = (shapes[59])
        r_x,r_y = (shapes[9])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)



    elif au==12:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)


    elif au==13:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)


    elif au==14:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==15:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==16:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
    elif au==17:
        (l_x1,_) = (shapes[36])
        (_,l_y1) = (shapes[38])
        (r_x1,_) = (shapes[39])
        (_,r_y1) = (shapes[41])
        (l_x2,_) = (shapes[42])
        (_,l_y2) = (shapes[44])
        (r_x2,_) = (shapes[45])
        (_,r_y2) = (shapes[47])
        cv2.ellipse(gray1,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    img_h, img_w= np.shape(gray1)
    
    xw1 = max(int(x1 - 10), 0)  #<---left side
    yw1 = max(int(y1 - 10), 0)  #<---head
    xw2 = min(int(x2 + 10), img_w - 1) #<---right side
    yw2 = min(int(y2 + 10), img_h - 1) #<--- bottom

    #att_map1 = cv2.resize(att_map[yw1:yw2,xw1:xw2], dsize=(28,28))

    return att_map



def process_image_batch(video_batch):
    # Initialize output tensor
    B, C, N, H, W = video_batch.shape
    device = video_batch.device
    no_of_AU_maps = 5  # Assuming we generate 16 AU maps for each frame
    au_maps_batch = torch.zeros((B, no_of_AU_maps, N, H, W)).to(device)
    
    # Convert video batch to numpy for OpenCV processing
    video_batch_np = video_batch.cpu().numpy()
    
    for b in range(B):
        for n in range(N):
            # Extract the frame and convert to RGB
            frame = video_batch_np[b, :, n, :, :]
            frame = frame.transpose(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # convert to 8bit gray
            gray_image *= 255
            gray_image = gray_image.astype(np.uint8)
            
            faces = ddetector(gray_image)
            if len(faces) > 0:
                (x1, y1, w, h) = face_utils.rect_to_bb(faces[0])
                shapes = dpredictor(gray_image, faces[0])
                shapes = face_utils.shape_to_np(shapes)
                x2 = x1 + w
                y2 = y1 + h
                
                # Generate AU maps
                attention_maps = []
                for au in range(1, no_of_AU_maps + 1):
                    att_map = AU_plot_ellipsoid(gray_image, au, [x1, y1, x2, y2, w, h], shapes)
                    attention_maps.append(att_map)
                
                # Convert attention maps to tensor and stack them
                attention_maps_tensor = torch.tensor(np.stack(attention_maps))
                au_maps_batch[b, :, n, :, :] = attention_maps_tensor
    au_maps_batch = au_maps_batch.to(device)/255
    return au_maps_batch

import os
import random
from collections import defaultdict


def get_paths(directory):
    real_videos = []
    real_videos1 = []
    fake_videos1 = []
    real_videos2 = []
    fake_videos2 = []
    real_videos3 = []
    fake_videos3 = []
    manipulated_videos = defaultdict(list)
    
    # Traverse the real_videos folder

    # real_videos_path = os.path.join('/mnt/sangraha/tharun/fake_videos/train/real')

    # if os.path.exists(real_videos_path):
    #     for file_name in os.listdir(real_videos_path):
    #         if file_name.endswith('.mp4'):
    #             real_videos_test.append(os.path.join(real_videos_path, file_name))

    import json
    import os

    # Path to the metadata file
    metadata_file_path = '/mnt/data/tarun/dfdc/dfdc_train_part_46/metadata.json'

    # Path to the directory containing all videos
    real_videos_directory_path = '/mnt/data/tarun/dfdc/dfdc_train_part_46'
    fake_videos_directory_path = '/mnt/data/tarun/dfdc/dfdc_train_part_46'


    # Load the metadata
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)
    

    # Filter for real videos and add their paths to the list
    for file_name, info in metadata.items():
        if info['label'] == 'REAL' and info['split'] == 'train':
            video_path = os.path.join(real_videos_directory_path, file_name)
            if os.path.exists(video_path) and file_name.endswith('.mp4'):
                real_videos1.append(video_path)

    for file_name, info in metadata.items():
        if info['label'] == 'FAKE' and info['split'] == 'train':
            video_path = os.path.join(fake_videos_directory_path, file_name)
            if os.path.exists(video_path) and file_name.endswith('.mp4'):
                fake_videos1.append(video_path)

    random.shuffle(real_videos1)
    split_index_real = int(0.1 * len(real_videos1))
    real_train1 = real_videos1[:split_index_real]
    real_test1 = real_videos1[split_index_real:]


    random.shuffle(fake_videos1)
    split_index_real = int(0.1 * len(fake_videos1))
    fake_train1 = fake_videos1[:split_index_real]
    fake_test1 = fake_videos1[split_index_real:]



    # metadata_file_path = '/mnt/sangraha/tharun/dfdc/dfdc_train_part_2/metadata.json'

    # # Path to the directory containing all videos
    # real_videos_directory_path = '/mnt/sangraha/tharun/dfdc/dfdc_train_part_2'
    # fake_videos_directory_path = '/mnt/sangraha/tharun/dfdc/dfdc_train_part_2'


    # # Load the metadata
    # with open(metadata_file_path, 'r') as f:
    #     metadata = json.load(f)
    

    # # Filter for real videos and add their paths to the list
    # for file_name, info in metadata.items():
    #     if info['label'] == 'REAL' and info['split'] == 'train':
    #         video_path = os.path.join(real_videos_directory_path, file_name)
    #         if os.path.exists(video_path) and file_name.endswith('.mp4'):
    #             real_videos2.append(video_path)

    # for file_name, info in metadata.items():
    #     if info['label'] == 'FAKE' and info['split'] == 'train':
    #         video_path = os.path.join(fake_videos_directory_path, file_name)
    #         if os.path.exists(video_path) and file_name.endswith('.mp4'):
    #             fake_videos2.append(video_path)

    # random.shuffle(real_videos2)
    # split_index_real = int(0.9 * len(real_videos2))
    # real_train2 = real_videos2[:split_index_real]
    # real_test2 = real_videos2[split_index_real:]


    # random.shuffle(fake_videos1)
    # split_index_real = int(0.9 * len(fake_videos2))
    # fake_train2 = fake_videos2[:split_index_real]
    # fake_test2 = fake_videos2[split_index_real:]



    # Traverse the manipulated_videos folder
    
                
    # fake_videos_path = os.path.join('/mnt/sangraha/tharun/fake_videos/train/fake')

    # if os.path.exists(fake_videos_path):
    #     for file_name in os.listdir(fake_videos_path):
    #         if file_name.endswith('.mp4'):
    #             fake_videos_test.append(os.path.join(fake_videos_path, file_name))



    ###Celeb V

    real_videos_path = os.path.join('/mnt/data/tarun/Celeb-HQ/Celeb-real')
    if os.path.exists(real_videos_path):
        for file_name in os.listdir(real_videos_path):
            if file_name.endswith('.mp4'):
                real_videos3.append(os.path.join(real_videos_path, file_name))
    

    fake_videos_path = os.path.join('/mnt/data/tarun/Celeb-HQ/Celeb-synthesis')
    if os.path.exists(fake_videos_path):
        for file_name in os.listdir(fake_videos_path):
            if file_name.endswith('.mp4'):
                fake_videos3.append(os.path.join(fake_videos_path, file_name))
                
    

    
    # Shuffle and split real videos
    random.shuffle(real_videos3)
    split_index_real = int(0.9 * len(real_videos3))
    real_train3 = real_videos3[:split_index_real]
    real_test3 = real_videos3[split_index_real:]
    
    random.shuffle(fake_videos3)
    split_index_real = int(0.9 * len(fake_videos3))
    fake_train3 = fake_videos3[:split_index_real]
    fake_test3 = fake_videos3[split_index_real:]

    ###

    real_videos_path = os.path.join(directory, 'Real_Videos')
    if os.path.exists(real_videos_path):
        for file_name in os.listdir(real_videos_path):
            if file_name.endswith('.mp4'):
                real_videos.append(os.path.join(real_videos_path, file_name))
                
    
    # Traverse the manipulated_videos folder
    manipulated_videos_path = os.path.join(directory, 'manipulated_sequences')
    if os.path.exists(manipulated_videos_path):
        for folder_name in os.listdir(manipulated_videos_path):
            if folder_name not in ['DeepFakeDetection', 'Deepfakes']:
                folder_path = os.path.join(manipulated_videos_path, folder_name)
                if os.path.isdir(folder_path):
                    for file_name in os.listdir(os.path.join(folder_path, 'c23', 'videos')):
                        if file_name.endswith('.mp4'):
                            person_id = file_name.split('_')[0]  # Extract person ID from file name
                            manipulated_videos[person_id].append(os.path.join(folder_path, 'c23', 'videos', file_name))
    
    # Shuffle and split real videos
    random.shuffle(real_videos)
    split_index_real = int(0.9 * len(real_videos))
    real_train = real_videos[:split_index_real]
    real_test = real_videos[split_index_real:]
    
    # Shuffle and split manipulated videos by person_id
    person_ids = list(manipulated_videos.keys())
    random.shuffle(person_ids)
    split_index_manipulated = int(0.9 * len(person_ids))
    manipulated_train = []
    manipulated_test = []
    
    for person_id in person_ids[:split_index_manipulated]:
        manipulated_train.extend(manipulated_videos[person_id])
    
    for person_id in person_ids[split_index_manipulated:]:
        manipulated_test.extend(manipulated_videos[person_id])
    
    #Ensure test set has equal number of real and fake videos
    # half_test_size = min(len(real_test), len(manipulated_test)) // 2
    # real_test = real_test[:half_test_size]
    # manipulated_test = manipulated_test[:half_test_size]
    
    # c=len(manipulated_train)-len(real_train) - len(real_train1) - len(real_train3) + len(fake_train1) +  + len(fake_train3)
    # i=0

    extra_real_videos=[]
    
    real_videos_path = os.path.join('/mnt/data/tarun/LivePortrait/test_videos')
    if os.path.exists(real_videos_path):
        for file_name in os.listdir(real_videos_path):
            #if file_name.endswith('.mp4') and i<c:
                extra_real_videos.append(os.path.join(real_videos_path, file_name))
                #i=i+1

    extra_real_videos1=[]
    
    real_videos_path = os.path.join('/mnt/data/tarun/Celeb-real')
    if os.path.exists(real_videos_path):
        for file_name in os.listdir(real_videos_path):
            #if file_name.endswith('.mp4') and i<c:
                extra_real_videos1.append(os.path.join(real_videos_path, file_name))
                #i=i+1

    extra_real_videos2=[]
    
    real_videos_path = os.path.join('/mnt/data/tarun/CelebV-HQ/35666')
    if os.path.exists(real_videos_path):
        for file_name in os.listdir(real_videos_path):
            #if file_name.endswith('.mp4') and i<c:
                extra_real_videos2.append(os.path.join(real_videos_path, file_name))           

    train_paths = real_train + manipulated_train 
    # print(len(real_train))
    # print(len(manipulated_train))

    
    train_labels = [0] * len(real_train) + [1] * len(manipulated_train) 
    print(len(real_train),len(manipulated_train))
    
    fake_test = manipulated_test
    test_paths = extra_real_videos1[0:25] + extra_real_videos2[0:25] + extra_real_videos[0:25]
    test_labels = [0] * 25 + [0] * 25 + [1] * 25

    print(len(real_test),len(fake_test))

    #test_paths = real_test[0:50] + manipulated_test[0:50]
    # print(len(fake_test))
    # print(len(real_test))
    #test_labels = [0] * 50 + [1] * 50

    # test_paths = extra_real_videos
    # test_labels = [1] * len(extra_real_videos) 
    

    # print(len(real_test))
    # print(len(manipulated_test))
    #test_labels =   [1] * 23 + [1] * 27 + [1] * 50 + [0] * 23 + [0] * 27 + [0] * 50


   
    
    # test_paths = extra_real_videos[0:100]
    # print(len(extra_real_videos))
    # test_labels = [1] * 100
    # print(len(real_train),'real_train_paths')
    # print(len(manipulated_train),'manipulated_train_paths')
    # #print(len(extra_real_videos),'extra_real_videos_paths')
    # print(len(real_test),'real_test_paths')
    # print(len(manipulated_test),'manipulated_test_paths')
   # print(len(train_labels),'train_labels')
    return (train_paths, train_labels), (test_paths, test_labels)

'''
def get_paths(directory):
    real_videos = []
    real_train_videos = []
    fake_videos = []
    manipulated_videos = defaultdict(list)
    
    # Traverse the real_videos folder
    
    #real_videos_path = os.path.join(directory, 'Real_Videos')
    real_videos_path = os.path.join('/mnt/sangraha/tharun/fake_videos/train/real')

    if os.path.exists(real_videos_path):
        for file_name in os.listdir(real_videos_path):
            if file_name.endswith('.mp4'):
                real_videos.append(os.path.join(real_videos_path, file_name))
    
    real_train_path = os.path.join('/mnt/sangraha/tharun/FF++/Real_Videos')

    if os.path.exists(real_train_path):
        for file_name in os.listdir(real_train_path):
            if file_name.endswith('.mp4'):
                real_train_videos.append(os.path.join(real_train_path, file_name))

    
                
    fake_videos_path = os.path.join('/mnt/sangraha/tharun/fake_videos/train/fake')

    if os.path.exists(fake_videos_path):
        for file_name in os.listdir(fake_videos_path):
            if file_name.endswith('.mp4'):
                fake_videos.append(os.path.join(fake_videos_path, file_name))
    
    # Traverse the manipulated_videos folder
    manipulated_videos_path = os.path.join(directory, 'manipulated_sequences')
    if os.path.exists(manipulated_videos_path):
        for folder_name in os.listdir(manipulated_videos_path):
            if folder_name not in ['DeepFakeDetection', 'Deepfakes']:
                folder_path = os.path.join(manipulated_videos_path, folder_name)
                if os.path.isdir(folder_path):
                    for file_name in os.listdir(os.path.join(folder_path, 'c23', 'videos')):
                        if file_name.endswith('.mp4'):
                            person_id = file_name.split('_')[0]  # Extract person ID from file name
                            manipulated_videos[person_id].append(os.path.join(folder_path, 'c23', 'videos', file_name))
    
    # Shuffle and split real videos
    # random.shuffle(real_videos)
    # split_index_real = int(0.8 * len(real_videos))
    # real_train = real_videos[:split_index_real]
    #real_test = real_videos[split_index_real:]
    real_test = real_videos
    fake_test = fake_videos
    
    # Shuffle and split manipulated videos by person_id
    person_ids = list(manipulated_videos.keys())
    random.shuffle(person_ids)
    split_index_manipulated = int(1 * len(person_ids))
    manipulated_train = []
    # manipulated_test = []
    
    for person_id in person_ids[:split_index_manipulated]:
        manipulated_train.extend(manipulated_videos[person_id])
    
    # for person_id in person_ids[split_index_manipulated:]:
    #     manipulated_test.extend(manipulated_videos[person_id])
    
    # # Ensure test set has equal number of real and fake videos
    # half_test_size = min(len(real_test), len(manipulated_test)) // 2
    # real_test = real_test[:half_test_size]
    # manipulated_test = manipulated_test[:half_test_size]
    
    c=len(manipulated_train)-len(real_train_videos)
    i=0

    extra_real_videos=[]
    
    real_videos_path = os.path.join('/mnt/sangraha/tharun/CelebV-HQ_10000')
    if os.path.exists(real_videos_path):
        for file_name in os.listdir(real_videos_path):
            if file_name.endswith('.mp4') and i<c:
                extra_real_videos.append(os.path.join(real_videos_path, file_name))
                i=i+1
                
    train_paths = real_train_videos + manipulated_train + extra_real_videos
    #train_paths = extra_real_videos

    train_labels = [0] * len(real_train_videos) + [1] * len(manipulated_train) + [0] *( len(manipulated_train)-len(real_train_videos))
    #train_labels = [0]*len(extra_real_videos)
    
    #test_paths = real_test + manipulated_test
    test_paths = real_test + fake_test
    #test_labels = [0] * len(real_test) + [1] * len(manipulated_test)
    test_labels = [0] * len(real_test) + [1] * len(fake_test) 

    
    #print(len(train_paths),'train_paths')
    #print(len(train_labels),'train_labels')
    return (train_paths, train_labels), (test_paths, test_labels)

#Example usage
# (train_paths, train_labels), (test_paths, test_labels) = get_paths('/path/to/dataset')
'''


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
        # links = [link for link, label in data]
        # labels = [label for link, label in data]
        logger.info(f"Processing {len(links)} videos")

        data = self.loader.process({'video_links': links})
        data = self.sampler.process(data)
        data = self.cropper.process(data)
        data = self.vnorm.process(data)
        data = self.stacker.process(data)
        # data['labels'] = torch.tensor(labels).to(self.device)
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
    

(train_paths, train_labels), (test_paths, test_labels) = get_paths('/mnt/data/tarun/FF++')
#print(len(train_paths),len(test_paths))
processing_pipeline = ProcessingPipeLine(device='cuda:0')
import joblib

if os.path.exists('dataset.joblib'):
    dataset = joblib.load('dataset.joblib')
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    print('Loaded dataset')
else:
    train_dataset = MyDataset(train_paths, train_labels, processing_pipeline)
    val_dataset = MyDataset(test_paths, test_labels, processing_pipeline)
    joblib.dump({'train': train_dataset,
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


train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                                 collate_fn=collate_fn)


val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True,
                                 collate_fn=collate_fn)


# train_dataloader, val_dataloader = val_dataloader, train_dataloader # For debugging

batch = next(iter(train_dataloader))
logger.info(f"Loaded first batch with {len(batch['video'])} videos.")

# from pytorch_lightning import Trainer, LightningModule
# from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import f1_score, roc_auc_score

# API key
os.environ['WANDB_API_KEY'] = "c3e27013774e26c468877ac5a16745e0b19ba267"
# os.environ['WANDB_MODE'] = "disabled"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
wandb.init(project='DDP_DEEPFAKE_testing')
# wandb_logger = WandbLogger(project='DDP_DEEPFAKE_testing')


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
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
        # self.au_decoder1 = nn.Sequential(nn.Linear(768,384),VitBackBone(num_heads=6,embed_dim=384,depth=4),nn.Linear(384,512), nn.Sigmoid())
        # self.au_decoder2 = nn.Sequential(nn.Linear(768,384),VitBackBone(num_heads=6,embed_dim=384,depth=4),nn.Linear(384,512), nn.Sigmoid())
        # self.au_decoder3 = nn.Sequential(nn.Linear(768,384),VitBackBone(num_heads=6,embed_dim=384,depth=4),nn.Linear(384,512), nn.Sigmoid())
        # self.au_decoder4 = nn.Sequential(nn.Linear(768,384),VitBackBone(num_heads=6,embed_dim=384,depth=4),nn.Linear(384,512), nn.Sigmoid())
        # self.au_decoder5 = nn.Sequential(nn.Linear(768,384),VitBackBone(num_heads=6,embed_dim=384,depth=4),nn.Linear(384,512), nn.Sigmoid())

        self.video_encoder = ViTVideo()
        #self.video_encoder.load_weights('/home/sushanth/deepfake_detection/tharun/initialization_weights/video.pt')
        self.patcher = SimplePatchify3D(16, 224, 224, 16, 16, 2)
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.clf_head = nn.Sequential( nn.Linear(768,2))
        self.dice_loss = DiceLoss()
        #self.video_encoder.freeze_parameters()

    def forward(self, videos):
        # videos: B * 3 * 16 * 224 * 224
        video_encodings = self.video_encoder(videos) 
        x = self.norm(video_encodings)


        # au_embeds1 = self.au_decoder1(video_encodings)
        # au_embeds2 = self.au_decoder2(video_encodings)
        # au_embeds3 = self.au_decoder3(video_encodings)
        # au_embeds4 = self.au_decoder4(video_encodings)
        # au_embeds5 = self.au_decoder5(video_encodings)
        
        #au_embeds = torch.cat((au_embeds1,au_embeds2,au_embeds3,au_embeds4,au_embeds5),dim=1)

        # au_maps_pred1 = self.patcher.unpatch(au_embeds1)
        # au_maps_pred2 = self.patcher.unpatch(au_embeds2)
        # au_maps_pred3 = self.patcher.unpatch(au_embeds3)
        # au_maps_pred4 = self.patcher.unpatch(au_embeds4)
        # au_maps_pred5 = self.patcher.unpatch(au_embeds5)

        #au_maps_pred = torch.cat((au_maps_pred1,au_maps_pred2,au_maps_pred3,au_maps_pred4,au_maps_pred5),dim=1)
        #attn_wts = au_maps_pred.mean(1, keepdim=True).repeat(1,3,1,1,1)
        #return au_maps_pred

        x =  self.fc_norm(x.mean(1))
        logits = self.clf_head(self.fc_dropout(x))
        # return logits,au_maps_pred

        # # attn_wts = au_maps_pred.mean(1, keepdim=True).repeat(1,3,1,1,1)
        # logits = self.clf_head(video_encodings).mean(1)
        # # logits = self.clf_head(video_encodings + video_encodings * self.patcher(attn_wts)).mean(1)
        return logits
    
    def calculate_clf_loss(self, logits,gt):
        #gt = gt.long()
        #return F.cross_entropy(logits, gt)
        #print(logits, gt)
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
        au_maps = process_image_batch(videos)

        logits= self(videos)
        clf_loss = self.calculate_clf_loss(logits, labels)
        #au_loss, au_losses1 = self.calculate_au_loss(au_maps_pred, au_maps)
        #au_losses = {'train_' + k: v for k, v in au_losses1.items()}
        loss = clf_loss
        to_log = {'train_loss': loss, 'train_clf_loss': clf_loss}
        #to_log.update(au_losses)
        return loss, to_log
    
    # def show_img(self,img):
    #     img = np.asarray(img)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.show()
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch['video'], batch['labels']
        au_maps = process_image_batch(videos)

        with torch.no_grad():
            logits = self(videos)
        

        if batch_idx==0:
            attn_maps = []
            
            for block_idx in range(11,12):
                # Access the attention map from the current block for a single video
                #attn_map = self.au_decoder1[1].blocks[block_idx].attn.attn_map  # Shape: (B, num_heads, 1568, 1568)
                attn_map = self.video_encoder.backbone1.blocks[block_idx].attn.attn_map

                #print(attn_map,'attn_map')
                attn_maps.append(attn_map[0])
            
            att_mat = torch.mean(torch.stack(attn_maps), dim=1)
            residual_att = torch.eye(att_mat[0].size(1))

            aug_att_mat = att_mat.detach().cpu() + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size())
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
                
            # Attention from the output token to the input space.
            v = joint_attentions[-1]
            avg_attention = v.mean(dim=0)

            video_mask = avg_attention.reshape(8,14,14).detach().numpy()
            mask = video_mask[0]
            #mask = cv2.resize(mask / mask.max(), (224,224))[..., np.newaxis]
            mask = cv2.applyColorMap((cv2.resize(mask / mask.max(), (224,224))*255).astype(np.uint8), cv2.COLORMAP_JET)

            mask = mask / mask.max()

            im = np.transpose(np.array(videos[0,:,0,:,:].detach().cpu()),(1,2,0))

            #print(im,'im')
            #print(mask*im,'mask')
            op = mask * im

            result = op/op.max()
            
            result1 = (mask * im)/op.max() + im
            result1 = result1/result1.max()
            result2 = 0.6 * im + 0.4 * mask
            result2 = result2/result2.max()
            #print(result,'result')
            images = [result,result1,result2,mask] 
            captions = ['attn_map','att_map1','att_map2','only_attn_map']
            images = [(img * 255).astype(np.uint8) for img in images]
            wandb.log({"sample_images_video": [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]})



        
 
            # attn_map = attn_map.mean(dim=1)  # Shape: (B, 1568, 1568)
            # print(attn_map.shape,'attn_map1')
            # # Process attention map for each frame
            # num_frames = 16
            # spatial_dim = 224 // 16  # Assuming patch size of 16

            # # Ensure the attention map has the correct size
            # expected_size = 1 * num_frames * spatial_dim * spatial_dim * num_frames * spatial_dim * spatial_dim

            # # Now reshape the attention map
            # if attn_map.numel() == expected_size:
            #     spatial_temporal_map = attn_map.view(1, num_frames, spatial_dim, spatial_dim, num_frames, spatial_dim, spatial_dim)
            #     spatial_temporal_map = spatial_temporal_map.permute(0, 1, 3, 4, 2, 5, 6).reshape(num_frames, num_frames, 224, 224)
            # else:
            #     raise ValueError(f"Unexpected tensor size. Expected {expected_size}, got {attn_map.numel()} instead.")
            
            # spatial_temporal_map = attn_map.view(-1, num_frames, spatial_dim, spatial_dim, num_frames, spatial_dim, spatial_dim)
            # spatial_temporal_map = spatial_temporal_map.permute(0, 2, 3, 1, 4, 5).reshape(num_frames, num_frames, 224, 224)
            
            # # Log the attention maps and overlayed images
            # self.log_attention_maps_to_wandb(spatial_temporal_map, videos, block_idx, num_frames=num_frames)

        clf_loss = self.calculate_clf_loss(logits, labels)
        #au_loss, au_losses = self.calculate_au_loss(au_maps_pred, au_maps)
        loss = clf_loss
        to_log = {'val_loss': loss, 'val_clf_loss': clf_loss}
        #au_losses = {'val_' + k: v for k, v in au_losses.items()}
        #to_log.update(au_losses)
        # if batch_idx == 0:
        #     self.log_images(videos, au_maps_pred, au_maps)
        # Calculate Acc, F1, AUC
        preds = torch.argmax(logits, dim=1)
        #print(preds,'preds')
        acc = torch.sum(preds == labels) / len(labels)
        f1 = f1_score(labels.cpu(), preds.cpu())
        to_log['val_acc'] = acc
        to_log['val_f1'] = f1
        to_log['TP'] = torch.sum((preds == 1) & (labels == 1)).float()
        to_log['TN'] = torch.sum((preds == 0) & (labels == 0)).float()
        to_log['FN'] = torch.sum((preds == 1) & (labels == 0)).float()
        to_log['FP'] = torch.sum((preds == 0) & (labels == 1)).float()
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
len(train_dataloader), len(val_dataloader)

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm

wandb.watch(model)

val_every_n_steps = 200
grad_acc_steps = 20
epochs = 100
mixed_precision = False
# Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
model.to(device)
optimizer = model.configure_optimizers()
scaler = GradScaler()

# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5, verbose=True)

best_val_loss = float('inf')
if os.path.exists('/home/sushanth/deepfake_detection/tharun/test_1.pth'):
    model.load_state_dict(torch.load('/home/sushanth/deepfake_detection/tharun/test_1.pth'))
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

# Training loop
for epoch in range(epochs):
    #model.train()
    train_logs = {'epoch': epoch}
    train_dataloader_tqdm = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
    
    for step, batch in enumerate(train_dataloader_tqdm):
        '''
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast(enabled=mixed_precision):
            loss, to_log = model.training_step(batch, step)

            loss = loss / grad_acc_steps
   
        scaler.scale(loss).backward()
        train_logs = accumulate_logs(train_logs, to_log, grad_acc_steps)

        if (step + 1) % grad_acc_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_logs['steps'] = step
            #print(train_logs)
            wandb.log(train_logs)
            train_logs = {'epoch': epoch}
            torch.cuda.empty_cache()
    
        #num_val_steps = 100
        '''
        flag=1
        
        if ((step+1) % val_every_n_steps == 0) or (step == len(train_dataloader)-1 or flag==1):
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
                    
                    # if val_step + 1 >= num_val_steps:
                    #      break
            
            # for k, v in to_log.items():
            #     val_to_log[k] = v / len(val_dataloader)
            
            precision = val_acc['TP'] / (val_acc['TP'] + val_acc['FP'] + 1e-8)
            recall    = val_acc['TP'] / (val_acc['TP'] + val_acc['FN'] + 1e-8)
            val_to_log['f1_score'] = (2 * precision * recall) /(precision+recall + 1e-8)
            val_to_log['val_acc'] = (val_acc['TP'] + val_acc['TN']) / (val_acc['TP'] + val_acc['TN'] + val_acc['FP'] + val_acc['FN'] )

            for k, v in val_acc.items():
                 val_to_log[k] = v / len(val_dataloader)

            avg_val_loss = val_loss_total / len(val_dataloader)
            val_to_log['val_loss'] = avg_val_loss
            wandb.log(val_to_log)
            
            # Save the best model
            
            # if avg_val_loss < best_val_loss:
            #     best_val_loss = avg_val_loss
            #     torch.save(model.state_dict(), 'test_1.pth')
            
            torch.cuda.empty_cache()
            # scheduler.step(epoch + step / len(train_dataloader))
            #model.train()

    #torch.save(model.state_dict(), 'model'+str(epoch)+'.pth')
    
    train_dataloader_tqdm.close()
