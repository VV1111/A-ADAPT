# -*- coding: utf-8 -*-
"""
AADAPT pre-training script
"""

# setup environment
import argparse
import os
join = os.path.join
import time
import json

import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
from tqdm import tqdm

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import *
from data.dataset import GeneralMedSegDB
from utils.loss import DiceBCELoss
from utils.logger import get_logger
from utils.metric import SegmentMetrics

# setup seeds
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# setup parser
parser = argparse.ArgumentParser("AADAPT training", add_help=False)
# model
parser.add_argument("--checkpoint", type=str, default="./playground/SAM",
                    help="path to SAM checkpoint folder")
parser.add_argument("--model_type", type=str, default="vit_b",
                    help="SAM model scale (e.g vit_b, vit_l, vit_h)")
parser.add_argument("--task_name", type=str, default="AADAPT")
parser.add_argument("--method", type=str, default="AADAPT")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
# data
parser.add_argument("--data_path", type=str, default="./playground/MedSegDB",
                    help="path to MedSegDB data folder")
# env
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0,1,2,3,4,5,6,7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--work_dir", type=str, default="./playground")
# train
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=32)
parser.add_argument("--resume", type=str, default=None, 
                    help="resume training from checkpoint")
# optimizer
parser.add_argument("--modality", type=str, default=['ct','mr'],
                    help="none means all modalities,ct-->only ct,mr-->only mr")
parser.add_argument("--uncert",action="store_true", default=False,
                    help="whether to use uncertainty based loss") 
parser.add_argument("--tgt_size", type=int, default=512)
parser.add_argument("--loss_topo",action="store_true", default=False,
                    help="whether to use topo based loss") 
parser.add_argument("--loss_cl",action="store_true", default=False,
                    help="whether to use centerline based loss") 
parser.add_argument("--setlr",action="store_true", default=False,
                    help="whether to use different lr for encoder") # 
parser.add_argument("--lr", type=float, default=0.001, metavar="LR", 
                    help="learning rate (absolute lr default: 0.001)")
parser.add_argument("--lr_en", type=float, default=0.0001, metavar="LR_EN", 
                    help="learning rate (absolute encoder lr default: 0.0001)")
parser.add_argument("--weight_decay", type=float, default=0.01, 
                    help="weight decay (default: 0.01)")
parser.add_argument("--use_amp", action="store_true", default=False, 
                    help="whether to use amp")


def build_joint_optimizer(
    model,
    lr_adapter=1e-4,
    lr_decoder=1e-4,
    lr_encoder=1e-5,      # encoder 主干小一点
    lr_prompt = 1e-4,
    weight_decay=1e-4,
):
    adapter_params = []
    embed_params = []
    decoder_params = []
    encoder_params = []
    prompt_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "image_encoder" in name and "adapter" in name:
            adapter_params.append(p)
        elif "image_encoder.modal_embed" in name or "image_encoder.organ_embed" in name:
            embed_params.append(p)
        elif "mask_decoder" in name:
            decoder_params.append(p)
        elif "prompt_encoder" in name:
            prompt_params.append(p)
        elif "image_encoder.blocks" in name:
            encoder_params.append(p)
        else:
            pass

    param_groups = [
        {"params": adapter_params, "lr": lr_adapter, "weight_decay": weight_decay},
        {"params": embed_params, "lr": lr_adapter, "weight_decay": weight_decay},
        {"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay},
        {"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": prompt_params, "lr": lr_prompt, "weight_decay": weight_decay},

    ]

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer

import torch.nn.functional as F

def compute_uncertainty_weight(
    logits: torch.Tensor,
    alpha_unc: float = 1.0,
    alpha_bd: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)  # B×1×H×W

    with torch.no_grad():
        prob = torch.sigmoid(logits)              # B×1×H×W

        unc = 4.0 * prob * (1.0 - prob)           # B×1×H×W

        max_pool = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
        boundary = max_pool - min_pool            # B×1×H×W

        w = 1.0 + alpha_unc * unc + alpha_bd * boundary
        w = torch.clamp(w, min=1.0, max=3.0)

    return w


def main(args):
    device = torch.device(args.device)

    checkpoint = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    print('checkpoint',checkpoint)
    sam_model = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=checkpoint)
    if args.method == "AADAPT":
        model = AADAPT(sam_model, args.bottleneck_dim, args.embedding_dim, args.expert_num,fine_tune_last_n_blocks=12,active_freq = True,active_tube = True).to(device)
        args.setlr=True 
        args.uncert = True


    else:
        raise NotImplementedError("Method {} not implemented!".format(args.method))
    dsc_metric = SegmentMetrics(["dsc"]).to(device)

    model = nn.DataParallel(model, device_ids=args.device_ids)
    dsc_metric = nn.DataParallel(dsc_metric, device_ids=args.device_ids)

    work_dir = join(args.work_dir, args.method)
    os.makedirs(work_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=work_dir)
    logger = get_logger(log_file=os.path.join(work_dir, 'output.log'))
    logger.info(f"args: {json.dumps(vars(args), indent=2)}")

    logger.info("Model: %s" % str(model))
    logger.info(
        "Number of total parameters: %d" % (
            sum(p.numel() for p in model.parameters()))
    )
    logger.info(
        "Number of trainable parameters: %d" % (
            sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    if args.setlr:
        optimizer = build_joint_optimizer(model,args.lr,args.lr,args.lr_en,args.lr,args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
    criterion = DiceBCELoss(sigmoid=True, squared_pred=True, reduction='none')
    logger.info("Criterion: %s" % str(criterion))

    train_dataset = GeneralMedSegDB(join(args.data_path, "train/ID"),tgt_size =args.tgt_size, train=True,modality=args.modality)
    logger.info(f"Number of training samples: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = GeneralMedSegDB(join(args.data_path, "test/ID"),tgt_size =args.tgt_size, train=False,modality=args.modality)
    logger.info(f"Number of validation samples: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    num_epochs = args.num_epochs
    start_epoch = 0
    best_loss = 1e10
    best_dsc = 0
    best_epoch = -1
    loss_log = []
    lr_log = []
    dsc_log = []

    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print(f"load model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            model.module.load_parameters(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        # train
        epoch_loss = 0
        step = 0
        model.train()
        pbar_train = tqdm(train_dataloader)
        pbar_train.set_description(f"Epoch [{epoch}/{num_epochs}] Train")
        for data, label in pbar_train:
            optimizer.zero_grad()
            step += 1

            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)), 
                                                                data["img"].shape[-2:]).reshape(-1, 4)
                data["img"] = img_transform(data["img"])
            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)

            label = label.to(device, non_blocking=True)          
            if args.loss_topo:
                if args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mask_pred,_,pred_dis, pred_ske = model(data)
                else:
                    mask_pred,_,pred_dis, pred_ske = model(data)
            else:
                if args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mask_pred = model(data)
                else:
                    mask_pred = model(data)
            if mask_pred.shape[-1] != label.shape[-1]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)

            losses = []
            if args.loss_cl:
                output = mask_pred[:,0].unsqueeze(1)
                loss = loss_calculation(output.float(), label, criterion,
                cl_alpha=2.0, cl_gamma=2.0,   
                cl_lambda=0.3,                
                lambda_dice=0.8, lambda_ce=0.2)  
                losses.append(loss)     
            elif args.uncert:
                if mask_pred.shape[1] == 1:
                    output = mask_pred[:, 0].unsqueeze(1)  # B×1×H×W
                    loss_vec = criterion(output.float(), label) 

                    if loss_vec.dim() == 2:  # [B,1] → [B]
                        loss_vec = loss_vec.squeeze(1)

                    weight_map = compute_uncertainty_weight(output, alpha_unc=1.0, alpha_bd=0.5)  # B×1×H×W
                    hardness = weight_map.mean(dim=[1, 2, 3])  # B
                    hardness_norm = hardness / (hardness.mean().detach() + 1e-6)  # 让平均值≈1

                    loss = (loss_vec * hardness_norm).mean()
                    losses.append(loss)
                else:
                    
                    num_masks = model.module.sam.mask_decoder.num_multimask_outputs
                    for i in range(num_masks):
                        output = mask_pred[:, i].unsqueeze(1)  # B×1×H×W

                        loss_vec = criterion(output.float(), label)
                        if loss_vec.dim() == 2:
                            loss_vec = loss_vec.squeeze(1)  # [B,1] → [B]

                        weight_map = compute_uncertainty_weight(output, alpha_unc=1.0, alpha_bd=0.5)
                        hardness = weight_map.mean(dim=[1, 2, 3])                            # [B]
                        hardness_norm = hardness / (hardness.mean().detach() + 1e-6)         # [B]

                        loss_i = (loss_vec * hardness_norm).mean()   # scalar
                        losses.append(loss_i)           
            else:  

                if mask_pred.shape[1]==1:  
                    output = mask_pred[:, 0].unsqueeze(1)
                    loss = criterion(output.float(), label)
                    losses.append(loss)                    
                else:
                    for i in range(model.module.sam.mask_decoder.num_multimask_outputs):
                        output = mask_pred[:, i].unsqueeze(1)
                        loss = criterion(output.float(), label)
                        losses.append(loss)
            loss = torch.stack(losses, dim=0).min(dim=0)[0]
            loss = loss.mean()
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            lr_main = lrs[0]
            if len(lrs) == 1:
                pbar_train.set_postfix({"lr": lr_main, "loss": loss.item()})
            else:  
                lr_en = lrs[-1]
                pbar_train.set_postfix({"lr": lr_main, "lr_en": lr_en, "loss": loss.item()})
                     
            # lr = optimizer.state_dict()['param_groups'][0]['lr']
            # pbar_train.set_postfix({"lr": lr, "loss": loss.item()})

            epoch_1000x = int((epoch + step / len(train_dataloader)) * 1000)
            for i, lr_i in enumerate(lrs):
                log_writer.add_scalar(f"batch/lr_group{i}", lr_i, epoch_1000x)
            # log_writer.add_scalar('batch/lr', lr, epoch_1000x)
            log_writer.add_scalar('batch/loss', loss.item(), epoch_1000x)

        # lr_log.append(lr)
        lr_log.append(lr_main)

        epoch_loss /= step
        loss_log.append(epoch_loss)
        # log_writer.add_scalar('epoch/lr', lr, epoch + 1)
        log_writer.add_scalar('epoch/lr_group0', lrs[0], epoch + 1)
        log_writer.add_scalar('epoch/loss', epoch_loss, epoch + 1)
        print(
            f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )

        ## save the latest model
        checkpoint = {
            "model": model.module.save_parameters(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(work_dir, "model_latest.pth"))

        ## save the lowest model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(work_dir, "model_lowest.pth"))

        ## save the model
        if epoch%5==0:
            checkpoint = {
                "model": model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(work_dir, f"model_{epoch}.pth"))

        # eval
        epoch_dsc = 0
        size = 0
        model.eval()
        pbar_val = tqdm(val_dataloader)
        pbar_val.set_description(f"Epoch [{epoch}/{num_epochs}] Val")
        with torch.no_grad():
            for data, label in pbar_val:
                if data["img"].shape[-1] != img_size:
                    data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)), 
                                                                    data["img"].shape[-2:]).reshape(-1, 4)
                    data["img"] = img_transform(data["img"])
                data["img"] = data["img"].to(device, non_blocking=True)
                data["box"] = data["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                mask_pred = model(data)
                if mask_pred.shape[-1] != label.shape[-1]:
                    mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)
                mask_prob = torch.sigmoid(mask_pred)
                mask = (mask_prob > 0.5).bool()

                dsc_ambiguous = []
                if args.loss_cl:
                    dsc_ambiguous.append(dsc_metric(mask[:, 0].unsqueeze(1), label)["dsc"])
                else:
                    if mask_pred.shape[1]==1:  
                        dsc_ambiguous.append(dsc_metric(mask[:, 0].unsqueeze(1), label)["dsc"])
                    else:

                        for idx in range(model.module.sam.mask_decoder.num_multimask_outputs):
                            dsc_ambiguous.append(dsc_metric(mask[:, idx].unsqueeze(1), label)["dsc"])
                dsc = torch.stack(dsc_ambiguous, dim=0).max(dim=0)[0]

                epoch_dsc += dsc.sum().item()
                size += dsc.shape[0]
                pbar_val.set_postfix({"dsc": dsc.mean().item()})

        epoch_dsc /= size
        dsc_log.append(epoch_dsc)
        log_writer.add_scalar('epoch/dsc', epoch_dsc, epoch + 1)
        print(
            f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, DSC: {epoch_dsc}'
        )

        ## save the best model
        if epoch_dsc > best_dsc:
            best_dsc = epoch_dsc
            best_epoch = epoch
            checkpoint = {
                "model": model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(work_dir, "model_best.pth"))

        # plot loss
        plt.plot(loss_log)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(work_dir, "train_loss.png"))
        plt.close()

        # plot lr
        plt.plot(lr_log)
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.savefig(join(work_dir, "lr.png"))
        plt.close()

        # plot dsc
        plt.plot(dsc_log)
        plt.title("Validation DSC")
        plt.xlabel("Epoch")
        plt.ylabel("DSC")
        plt.savefig(join(work_dir, "val_dsc.png"))
        plt.close()

        logger.info(f"Epoch [{epoch}] - LR: {lr_main}, Loss: {epoch_loss}, DSC: {epoch_dsc}")
        log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info(f"Best epoch: {best_epoch}, Best DSC: {best_dsc}")
    logger.info(f"Time cost: {total_time_str}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    '''
    
    '''