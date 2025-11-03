import math
from typing import Iterable

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm


def num_to_groups(num: int, divisor: int) -> Iterable[int]:
    full = num // divisor
    remainder = num % divisor
    for _ in range(full):
        yield divisor
    if remainder > 0:
        yield remainder


class FIDEvaluation:
    """
    Minimal FID evaluation loop in the spirit of Ho et al. (2020).
    Computes Inception activations for 50k real and generated images
    every time the routine is run, without caching or classifier-free features.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        dataloader,
        sampler,
        device: str = "cuda",
        channels: int = 3,
        num_samples: int = 50_000,
        inception_block_dim: int = 2048,
    ):
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.sampler = sampler
        self.device = device
        self.channels = channels
        self.num_samples = num_samples

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_dim]
        self.inception = InceptionV3([block_idx]).to(device)
        self.inception.eval()

    def _inception_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.channels == 1:
            images = repeat(images, "b 1 ... -> b c ...", c=3)

        with torch.no_grad():
            features = self.inception(images)[0]
            if features.shape[-1] != 1 or features.shape[-2] != 1:
                features = adaptive_avg_pool2d(features, (1, 1))
            features = rearrange(features, "... 1 1 -> ...")
        return features

    def _collect_real_features(self) -> np.ndarray:
        feats = []
        data_iter = iter(self.dataloader)
        total_batches = math.ceil(self.num_samples / self.batch_size)

        for _ in tqdm(range(total_batches), desc="real activations"):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            batch = batch.to(self.device)
            feats.append(self._inception_features(batch))

        feats = torch.cat(feats, dim=0)[: self.num_samples]
        return feats.cpu().numpy()

    def _collect_fake_features(self) -> np.ndarray:
        feats = []
        pbar = tqdm(total=self.num_samples, desc="fake activations", unit="img")  # ✅ explicit total in images

        for cur_bs in num_to_groups(self.num_samples, self.batch_size):
            samples = self.sampler.sample(batch_size=cur_bs).to(self.device)
            feats.append(self._inception_features(samples))
            pbar.update(cur_bs)  # ✅ advance by how many images we just made

        pbar.close()
        feats = torch.cat(feats, dim=0)[: self.num_samples]
        return feats.cpu().numpy()


    def fid_score(self) -> float:
        real_features = self._collect_real_features()
        fake_features = self._collect_fake_features()

        m1, s1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        m2, s2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, m2, s2)
