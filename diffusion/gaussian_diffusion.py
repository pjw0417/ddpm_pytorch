import math

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: torch.Tensor) -> torch.Tensor:
    return (t + 1) * 0.5


class GaussianDiffusion(nn.Module):
    """
    Vanilla DDPM noise scheduler and samplers matching Ho et al. (2020).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        image_size: int,
        channels: int = 3,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        auto_normalize: bool = True,
    ):
        super().__init__()
        self.model = model
        self.channels = channels

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert len(image_size) == 2, "image size must be an int or a (h, w) tuple"
        self.image_size = image_size

        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(betas.shape[0])

        register = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register("betas", betas)
        register("alphas_cumprod", alphas_cumprod)
        register("alphas_cumprod_prev", alphas_cumprod_prev)
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register("posterior_variance", posterior_variance)
        register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else (lambda x: x)
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else (lambda x: x)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t):
        b, *_, device = *x.shape, x.device
        time = torch.full((b,), t, device=device, dtype=torch.long)

        model_output = self.model(x, time)
        x_start = self.predict_start_from_noise(x, time, model_output)
        x_start = torch.clamp(x_start, -1.0, 1.0)

        model_mean, _, model_log_variance = self.q_posterior(x_start, x, time)
        if t == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        b = shape[0]
        device = self.device
        img = torch.randn(shape, device=device)
        imgs = [img] if return_all_timesteps else None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling", total=self.num_timesteps):
            img = self.p_sample(img, t)
            if return_all_timesteps:
                imgs.append(img)

        if return_all_timesteps:
            return torch.stack(imgs, dim=0)
        return img

    @torch.no_grad()
    def sample(self, batch_size: int = 16, return_all_timesteps = False):
        h, w = self.image_size
        shape = (batch_size, self.channels, h, w)
        samples = self.p_sample_loop(shape, return_all_timesteps=return_all_timesteps)
        return self.unnormalize(samples)

    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_t, t)
        return F.mse_loss(model_out, noise)

    def forward(self, img):
        b, c, h, w = img.shape
        assert (h, w) == self.image_size, f"expected image size {self.image_size}"
        assert c == self.channels, f"expected {self.channels} channels"

        img = self.normalize(img)
        t = torch.randint(0, self.num_timesteps, (b,), device=img.device, dtype=torch.long)
        return self.p_losses(img, t)
