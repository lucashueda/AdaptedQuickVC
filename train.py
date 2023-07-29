import os
import json
import argparse
import itertools
import math
import multiprocessing
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from pqmf import PQMF

import commons
import qvc_utils as utils

from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  subband_stft_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
#from text.symbols import symbols

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '65520'
#   n_gpus = 1

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)


  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,40,50,60,70,80,90,100,110,120,160,200,230,260,300,350,400,450,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioCollate()

  num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()

  train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
      collate_fn=collate_fn) # batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
    eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
        batch_size=1, pin_memory=False,
        drop_last=False, collate_fn=collate_fn)
  
  net_g = SynthesizerTrn(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()



def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers
  tmp=0
  tmp1=1000000000
  #train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, items in enumerate(train_loader):
    c, f0, spec, y, spk, lengths, uv, energy = items

    g = spk.cuda(rank, non_blocking=True)
    spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
    c = c.cuda(rank, non_blocking=True)
    f0 = f0.cuda(rank, non_blocking=True)
    uv = uv.cuda(rank, non_blocking=True)
    energy = energy.cuda(rank, non_blocking=True)
    lengths = lengths.cuda(rank, non_blocking=True)
    mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax)
    # print(mel.shape)

    # print(spec.shape, mel.shape)

    with autocast(enabled=hps.train.fp16_run):
        y_hat, y_hat_mb, ids_slice, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(c, f0, uv, spec, energy=energy, g=g, c_lengths=lengths, spec_lengths=lengths)

        # print(y_hat.shape, ids_slice)

        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

        with autocast(enabled=False):
          loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
          loss_disc_all = loss_disc

    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    


    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        #loss_dur = torch.sum(l_length.float())
        # print(y_mel.shape, y_hat_mel.shape)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        
        if hps.model.mb_istft_vits == True:
          pqmf = PQMF(y.device)
          y_mb = pqmf.analysis(y)
          loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
        else:
          loss_subband = torch.tensor(0.0)

        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_subband#+ loss_dur 

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, loss_subband]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel,  "loss/g/kl": loss_kl, "loss/g/subband": loss_subband})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            #"all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1

  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))
  print(tmp,tmp1)
    

 
def evaluate(hps, generator, eval_loader, writer_eval):
  generator.eval()
  image_dict = {}
  audio_dict = {}
  with torch.no_grad():
    for batch_idx, items in enumerate(eval_loader):
      c, f0, spec, y, spk, _, uv, energy = items
      g = spk[:1].cuda(0)
      spec, y = spec[:1].cuda(0), y[:1].cuda(0)
      c = c[:1].cuda(0)
      f0 = f0[:1].cuda(0)
      energy = energy[:1].cuda(0)
      uv= uv[:1].cuda(0)
      mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin,
          hps.data.mel_fmax)
      y_hat = generator.module.infer(c, f0, uv, energy = energy, g=g)

      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1).float(),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax
      )

      audio_dict.update({
          f"gen/audio_{batch_idx}": y_hat[0],
          f"gt/audio_{batch_idx}": y[0]
      })
    image_dict.update({
        f"gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    })
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
  main()
