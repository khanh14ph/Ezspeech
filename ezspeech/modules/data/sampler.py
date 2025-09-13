import logging
from collections import Counter
from operator import itemgetter
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import lognorm
from torch.utils.data import BatchSampler, Sampler
from tqdm import tqdm


def get_logger(name):
    return logging.getLogger(name)
logger = get_logger(__name__)
import random


class DynamicBatchSampler(BatchSampler):

    def __init__(
        self,
        sampler,
        max_batch_duration: int,
        num_buckets: Optional[int] = None,

    ):
        self.tracker=0
        self._dataset = sampler.dataset
        self.max_batch_duration=max_batch_duration

        sampler_idx = list(x for x in sampler)
        count=0
        max_duration=0
        for i in sampler_idx:
            dur=self._dataset.get_dur(i)
            if dur > max_duration:
                max_duration=dur
        bucket_size=max_duration/num_buckets

        self.buckets_map=dict()

        for i in tqdm(sampler_idx,desc=f"Add samples to bucket map - [{self.tracker}]..."):
            count+=1
            dur=self._dataset.get_dur(i)

            bucket_identifier=int(dur//bucket_size)
            temp=self.buckets_map.get(bucket_identifier,[])
            temp.append((i,dur))
            self.buckets_map[bucket_identifier]=temp

        self._generate_batches()
    def __iter__(self):
        for batch in self.batches:
            yield batch
        self._generate_batches()
    def _generate_batches(self):

        batch_list = []
        for  i in tqdm(self.buckets_map, desc=f"Generating bucket data - [{self.tracker}]..."):
            cur_batch = []
            cur_dur = 0
            element_in_a_bucket=self.buckets_map[i]
            random.shuffle(element_in_a_bucket)
            for idx,dur in element_in_a_bucket:
                if cur_dur+dur <self.max_batch_duration :
                    cur_batch.append(idx)
                    cur_dur += dur

                    if idx==len(element_in_a_bucket):
                        batch_list.append(cur_batch)
                else:
                    if len(cur_batch) > 0:
                        batch_list.append(cur_batch)
                    if dur <= self.max_batch_duration:
                        cur_batch = [idx]
                        cur_dur = dur

                    else:
                        cur_batch = []
                        cur_dur = 0

        if len(cur_batch) > 0:
            batch_list.append(cur_batch)
        total_samples=0
        for i in batch_list:
            total_samples+=len(i)
        print(f"GPU rank {dist.get_rank()} handle {total_samples} samples")
        random.shuffle(batch_list)
        self.tracker+=1

        self.batches = batch_list
    def __len__(self):
        return len(self.batches)
