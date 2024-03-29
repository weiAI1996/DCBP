import glob
import os
import datetime
import sys
from tkinter.tix import Tree

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.model import DCBP
from nets.DCBP_training import (get_lr_scheduler, set_optimizer_lr,
                                 weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

if __name__ == "__main__":
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    ROOT = "/data2/jw/runway_seg/dataset/runway"
    train_paths = glob.glob(os.path.join("/data2/jw/runway_seg/dataset/runway/train", "*.png"))
    val_paths = glob.glob(os.path.join("/data2/jw/runway_seg/dataset/runway/test", "*.png"))
    label_paths = glob.glob(os.path.join("/data2/jw/runway_seg/dataset/runway/labels", "*.png"))
    num_classes     = 3
    backbone        = "hrnetv2_w32"
    pretrained      = False
    model_path      = "/data2/jw/runway_seg/DCBP/model_data/pretrained.pth"
    input_shape     = [512, 512]
    save_dir            = 'logs_line_nolocal_spatial_newdata_eval_dice_nocopypaste111'
    Init_Epoch          = 0
    Freeze_Epoch        = 20
    Freeze_batch_size   = 4
    
    UnFreeze_Epoch      = 400
    Unfreeze_batch_size = 4
   
    Freeze_Train        = True

    Init_lr             = 5e-4
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "adamw"
    momentum            = 0.9
    weight_decay        = 1e-2

    lr_decay_type       = 'cos'

    save_period         = 4


    eval_flag           = True
    eval_period         = 4

    dice_loss       = True

    focal_loss      = False

    cls_weights     = np.ones([num_classes], np.float32)
    cls_weights_line     = np.array([0.01,2], np.float32)

    num_workers     = 8

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0


    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model   = DCBP(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    val_name_list = []
    for item in val_paths:
        if 'win' in sys.platform:
            name = item.split('.')[0].split('\\')[-1]
        else:
            name =item.split('.')[0].split('/')[-1]
        val_name_list.append(name)
    num_val = int(len(val_paths))
    num_train = int(len(train_paths))

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        wanted_step = 1.5e4 if optimizer_type == "adamw" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:

            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            


    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False


        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        print(Init_lr_fit)
        print(Min_lr_fit)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]


        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        
        train_dataset   = SegmentationDataset(train_paths,  input_shape, num_classes, True, ROOT)
        val_dataset     = SegmentationDataset(val_paths,  input_shape, num_classes, False, ROOT)
    
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_paths, ROOT, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size



                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler)

                UnFreeze_flag   = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
                dice_loss, focal_loss, cls_weights,cls_weights_line, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
