import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import qvc_utils
import commons 
from mel_processing import spectrogram_torch, spec_to_mel_torch
from qvc_utils import load_wav_to_torch, load_filepaths_and_text
#import h5py


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length  = hparams.data.filter_length
        self.hop_length     = hparams.data.hop_length
        self.win_length     = hparams.data.win_length
        self.sampling_rate  = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        # self.use_spk = hparams.model.use_spk
        self.spk_map = hparams.spk
        self.spec_len = hparams.train.max_speclen

        random.seed(1243)
        random.shuffle(self.audiopaths)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        lengths = []
        for audiopath in self.audiopaths:
            lengths.append(os.path.getsize(audiopath[0]) // (2 * self.hop_length))
        self.lengths = lengths

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        
        spec_filename = filename.replace(".wav", ".spec.pt")

        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
            

        spk = filename.split("/")[-2]
        spk = torch.LongTensor([self.spk_map[spk]])

        f0 = np.load(filename + ".f0.npy")
        f0, uv = qvc_utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        c = torch.load(filename+ ".soft.pt")
        c = qvc_utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])

        energy = np.load(filename + '.energy.npy')
        energy = torch.FloatTensor(energy)

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(audio_norm.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv, energy = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin], energy[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None
        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1]-800)
            end = start + 790
            spec, c, f0, uv, energy = spec[:, start:end], c[:, start:end], f0[start:end], uv[start:end], energy[start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]

        return c, f0, spec, audio_norm, spk, uv, energy

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)
    
class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spkids = torch.LongTensor(len(batch), 1)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)
        energy_padded = torch.FloatTensor(len(batch), max_c_len)

        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        energy_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c
            lengths[i] = c.size(1)

            f0 = row[1]
            f0_padded[i, :f0.size(0)] = f0

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav

            spkids[i, 0] = row[4]

            uv = row[5]
            uv_padded[i, :uv.size(0)] = uv

            energy = row[6]
            energy_padded[i, :energy.size(0)] = energy

        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, energy_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)

      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))

      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]

          # add extra samples to make it evenly divisible
          
          rem = num_samples_bucket - len_bucket
          print(len_bucket, rem)
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]

          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)

      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches

      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1

      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
# import utils
from torch.utils.data import DataLoader
if __name__ == "__main__":
    hps = utils.get_hparams()
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,70,100,200,300,400,500,600,700,800,900,1000],
        num_replicas=1,
        rank=0,
        shuffle=True)
    collate_fn = TextAudioCollate(hps)
    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)

    for batch_idx, (c, spec, y) in enumerate(train_loader):
        print(c.size(), spec.size(), y.size())
        #print(batch_idx, c, spec, y)
        #break
