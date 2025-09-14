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
random.seed(10)
def divide_list_simple(lst, n):
    """Divide list into n parts using a simple approach"""
    length = len(lst)
    chunk_size = length // n
    remainder = length % n
    
    result = []
    start = 0
    
    for i in range(n):
        # Add one extra element to the first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    
    return result
def count_list_of_list(x):
    count=0
    for i in x:
        count+=len(i)
    return count    
class DynamicBatchSampler(Sampler):

    def __init__(
        self,
        sampler,
        max_batch_duration: int,
        num_buckets: Optional[int] = None,

    ):
        self.rank=dist.get_rank()
        self.world_size=dist.get_world_size()

        self._dataset = sampler.data_source
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

        for i in tqdm(sampler_idx,desc=f"Add samples to bucket map - [{self.rank}]..."):
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
        for  i in tqdm(self.buckets_map, desc=f"Generating bucket data - [{self.rank}]..."):
            cur_batch = []
            cur_dur = 0
            element_in_a_bucket=self.buckets_map[i]
            random.shuffle(element_in_a_bucket)
            current_bucket_batch_list=[]
            for idx,dur in element_in_a_bucket:
                if cur_dur+dur <self.max_batch_duration :
                    cur_batch.append(idx)
                    cur_dur += dur
                else:
                    if len(cur_batch) > 0:
                        current_bucket_batch_list.append(cur_batch)
                    if dur <= self.max_batch_duration:
                        cur_batch = [idx]
                        cur_dur = dur

                    else:
                        cur_batch = []
                        cur_dur = 0
            current_bucket_batch_list.append(cur_batch)
        
            num_sample_in_bucket=0
            original_length=len(current_bucket_batch_list)
            valid_length=len(current_bucket_batch_list)//self.world_size*self.world_size
            redundant_batches=current_bucket_batch_list[valid_length:]
            current_bucket_batch_list=current_bucket_batch_list[:valid_length]
            if original_length!=valid_length:
                
                redundant_batches = [item for sublist in redundant_batches for item in sublist]
                
                if len(redundant_batches)>self.world_size:
                    
                    redundant_batches=divide_list_simple(redundant_batches,self.world_size)
                    current_bucket_batch_list=current_bucket_batch_list+redundant_batches
            current_bucket_batch_list=current_bucket_batch_list[self.rank::self.world_size]

            batch_list.extend(current_bucket_batch_list)
        total_samples=0
        for i in batch_list:
            total_samples+=len(i)
        print(f"GPU rank {self.rank} handle {total_samples} samples")

        self.batches = batch_list
    def __len__(self):
        return len(self.batches)
