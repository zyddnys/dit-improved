
from copy import deepcopy
import os
import numpy as np
from turtle import forward
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from ema_pytorch import EMA

from dit import DitConditional

from utils import AvgMeter
from tqdm import tqdm

@dataclass
class UnetConfig :
    downsample_layers: int
    bottleneck_layers: int
    planes: int

@dataclass
class DiffusionConfig :
    steps: int
    unet_cfg: UnetConfig


class VarianceSchedule(nn.Module) :
    def __init__(self, steps: int) -> None:
        super(VarianceSchedule, self).__init__()
        self.a = 0.00085#1e-4
        self.b = 0.012#0.02
        self.n_steps = steps
        self.steps = torch.tensor(list(range(self.n_steps)), dtype = torch.long)
        self.betas = torch.linspace(self.a**0.5, self.b**0.5, steps, dtype = torch.float64)**2
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)
        self.alpha_bars_prev = self.alpha_bars.clone()
        self.alpha_bars_prev[1:] = self.alpha_bars_prev[:-1].clone()
        self.alpha_bars_prev[0] = 1
        self.sqrt_one_minus_alpha_bars = (1.0 - self.alpha_bars).sqrt()
        self.sigmas = 0 * (
            (1 - self.alpha_bars_prev) / (1 - self.alpha_bars) *
            (1 - self.alpha_bars / self.alpha_bars_prev)
        ).sqrt()

    def beta(self, t: torch.LongTensor) -> torch.DoubleTensor :
        """
        beta
        """
        return self.betas.to(t.device).gather(0, t)

    def alpha(self, t: torch.LongTensor) -> torch.DoubleTensor :
        """
        1-beta
        """
        return self.alphas.to(t.device).gather(0, t)

    def alpha_bar(self, t: torch.LongTensor) -> torch.DoubleTensor :
        """
        cum prod of alpha
        """
        return self.alpha_bars.to(t.device).gather(0, t)

    def alpha_bar_prev(self, t: torch.LongTensor) -> torch.DoubleTensor :
        """
        cum prod of alpha
        """
        return self.alpha_bars_prev.to(t.device).gather(0, t)

    def sqrt_one_minus_alpha_bar(self, t: torch.LongTensor) -> torch.DoubleTensor :
        """
        cum prod of alpha
        """
        return self.sqrt_one_minus_alpha_bars.to(t.device).gather(0, t)

    def sigma(self, t: torch.LongTensor) -> torch.DoubleTensor :
        """
        cum prod of alpha
        """
        return self.sigmas.to(t.device).gather(0, t)

    def make_ddim_schedule(self, n_steps: int = 50) :
        ret = deepcopy(self)
        c = self.n_steps // n_steps
        ret.steps = torch.tensor(list(range(0, self.n_steps, c)), dtype = torch.long) + 1
        ret.n_steps = n_steps
        ret.betas = self.betas[ret.steps]
        ret.alphas = self.alphas[ret.steps]
        ret.alpha_bars = self.alpha_bars[ret.steps]
        ret.alpha_bars_prev = torch.tensor([self.alpha_bars[0]] + self.alpha_bars[ret.steps[:-1]].tolist())
        ret.sqrt_one_minus_alpha_bars = (1.0 - ret.alpha_bars).sqrt()
        ret.sigmas = 0 * (
            (1 - ret.alpha_bars_prev) / (1 - ret.alpha_bars) *
            (1 - ret.alpha_bars / ret.alpha_bars_prev)
        ).sqrt()
        return ret


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler: VarianceSchedule):
    # fix beta: zero terminal SNR
    print(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # print("original:", noise_scheduler.betas)
    # print("fixed:", betas)

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alpha_bars = alphas_cumprod
    noise_scheduler.alpha_bars_prev = noise_scheduler.alpha_bars.clone()
    noise_scheduler.alpha_bars_prev[1:] = noise_scheduler.alpha_bars_prev[:-1].clone()
    noise_scheduler.alpha_bars_prev[0] = 1
    noise_scheduler.sqrt_one_minus_alpha_bars = (1.0 - noise_scheduler.alpha_bars).sqrt()
    noise_scheduler.sigmas = 0 * (
        (1 - noise_scheduler.alpha_bars_prev) / (1 - noise_scheduler.alpha_bars) *
        (1 - noise_scheduler.alpha_bars / noise_scheduler.alpha_bars_prev)
    ).sqrt()

class UnetClass(nn.Module) :
    def __init__(self, dim, cdt_dim, num_class) -> None:
        super().__init__()
        #self.denoiser = UnetConditional(dim, cdt_dim)
        self.denoiser = DitConditional(512, cdt_dim, num_blocks = 5, in_dim = 3, out_dim = 3)
        self.class_embd = nn.Embedding(num_class, cdt_dim)

    def forward(self, x, t, y) :
        return self.denoiser(x, t, self.class_embd(y))

class DiffusionModelConditional(nn.Module) :
    def __init__(self, cfg: DiffusionConfig) -> None :
        super(DiffusionModelConditional, self).__init__()
        self.cfg = cfg
        self.denoiser = UnetClass(32, 32, 11)
        self.denoiser_ema = EMA(self.denoiser, beta = 0.999)
        self.var_sch = VarianceSchedule(self.cfg.steps)
        fix_noise_scheduler_betas_for_zero_terminal_snr(self.var_sch)

    def sample(self, shape, n_cls: int, n_samples_per_cls: int) -> torch.FloatTensor :
        process = []
        with torch.no_grad() :
            self.denoiser_ema.eval()
            all_cls = torch.tensor(range(n_cls), dtype = torch.long).repeat_interleave(n_samples_per_cls).cuda()
            shape = (n_cls * n_samples_per_cls, *shape)
            result = torch.randn(shape, dtype = torch.float64).cuda()
            for i in tqdm(range(self.cfg.steps)) :
                t = torch.tensor([self.cfg.steps - i - 1], dtype = torch.int64).repeat(n_cls * n_samples_per_cls).cuda()
                alpha = self.var_sch.alpha(t).view(-1, 1, 1, 1)
                alpha_bar = self.var_sch.alpha_bar(t).view(-1, 1, 1, 1)
                term1 = alpha.rsqrt()
                term2 = (1 - alpha) * (1 - alpha_bar).rsqrt()
                noise_pred = self.denoiser_ema(result.float(), t, all_cls).double()
                if i < self.cfg.steps - 1 :
                    z = torch.randn(shape).cuda()
                    #sigma = ((1 - self.var_sch.alpha_bar(t - 1).view(-1, 1, 1, 1)) / (1 - alpha_bar) * self.var_sch.beta(t).view(-1, 1, 1, 1)).sqrt()
                    sigma = self.var_sch.beta(t).sqrt().view(-1, 1, 1, 1)
                    result = term1 * (result - term2 * noise_pred) + sigma * z
                else :
                    result = term1 * (result - term2 * noise_pred)
                if (i + 1) % 10 == 0 :
                    process.append(torch.clip(result[0], -1, 1))
        return torch.clip(result, -1, 1), torch.stack(process, dim = 0)

    def sample_ddim(self, shape, n_cls: int, n_samples_per_cls: int) -> torch.FloatTensor :
        process = []
        with torch.no_grad() :
            self.denoiser_ema.eval()
            all_cls = torch.tensor(range(n_cls), dtype = torch.long).repeat_interleave(n_samples_per_cls).cuda()
            shape = (n_cls * n_samples_per_cls, *shape)
            result = torch.randn(shape, dtype = torch.float64).cuda()
            xt = torch.randn(shape, dtype = torch.float64).cuda()
            for i in tqdm(range(self.cfg.steps)) :
                t = torch.tensor([self.cfg.steps - i - 1], dtype = torch.int64).repeat(n_cls * n_samples_per_cls).cuda()
                noise_pred = self.denoiser_ema(xt.float(), t, all_cls).double()
                alpha_bar = self.var_sch.alpha_bar(t).view(-1, 1, 1, 1)
                alpha_bar_prev = self.var_sch.alpha_bar_prev(t).view(-1, 1, 1, 1)
                sigma = self.var_sch.sigma(t).view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar = self.var_sch.sqrt_one_minus_alpha_bar(t).view(-1, 1, 1, 1)
                pred_x0 = ((xt - sqrt_one_minus_alpha_bar * noise_pred) / alpha_bar.sqrt())
                dir_xt = (1.0 - alpha_bar_prev - sigma ** 2).sqrt() * noise_pred
                if i < self.cfg.steps - 1 :
                    z = torch.randn(shape).cuda()
                    x_prev = alpha_bar_prev.sqrt() * pred_x0 + dir_xt + sigma * z
                else :
                    x_prev = alpha_bar_prev.sqrt() * pred_x0 + dir_xt
                xt = x_prev
                if (i + 1) % 10 == 0 :
                    process.append(torch.clip(xt[0], -1, 1))
        return torch.clip(xt, -1, 1), torch.stack(process, dim = 0)

    def sample_ddim_acc(self, shape, n_cls: int, n_samples_per_cls: int, n_steps: int = 50) -> torch.FloatTensor :
        process = []
        with torch.no_grad() :
            sch = self.var_sch.make_ddim_schedule()
            self.denoiser_ema.eval()
            all_cls = torch.tensor(range(n_cls), dtype = torch.long).repeat_interleave(n_samples_per_cls).cuda()
            shape = (n_cls * n_samples_per_cls, *shape)
            xt = torch.randn(shape, dtype = torch.float64).cuda()
            s = list(torch.flip(sch.steps, dims = [0]))
            for i, t in enumerate(tqdm(s)) :
                t = torch.tensor([t], dtype = torch.int64).repeat(n_cls * n_samples_per_cls).cuda()
                idx = torch.tensor([sch.n_steps - i - 1], dtype = torch.int64).repeat(n_cls * n_samples_per_cls).cuda()
                noise_pred = self.denoiser_ema(xt.float(), t, all_cls).double()
                alpha_bar = sch.alpha_bar(idx).view(-1, 1, 1, 1)
                alpha_bar_prev = sch.alpha_bar_prev(idx).view(-1, 1, 1, 1)
                sigma = sch.sigma(idx).view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar = sch.sqrt_one_minus_alpha_bar(idx).view(-1, 1, 1, 1)
                pred_x0 = ((xt - sqrt_one_minus_alpha_bar * noise_pred) / alpha_bar.sqrt())
                dir_xt = (1.0 - alpha_bar_prev - sigma ** 2).sqrt() * noise_pred
                if i < len(s) - 1 :
                    z = torch.randn(shape).cuda()
                    x_prev = alpha_bar_prev.sqrt() * pred_x0 + dir_xt + sigma * z
                else :
                    x_prev = alpha_bar_prev.sqrt() * pred_x0 + dir_xt
                xt = x_prev
                process.append(torch.clip(xt[0], -1, 1))
        return torch.clip(xt, -1, 1), torch.stack(process, dim = 0)

    def add_noise(self, x: torch.FloatTensor, t: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor] :
        alpha_bars = self.var_sch.alpha_bar(t)
        sqrtalpha_bars = alpha_bars.sqrt().to(x.device)
        sqrtone_minus_alpha_bars = (1.0 - alpha_bars).sqrt().to(x.device)
        noise = torch.randn_like(x)
        corrupted_image = sqrtalpha_bars.view(-1, 1, 1, 1) * x.double() + sqrtone_minus_alpha_bars.view(-1, 1, 1, 1) * noise.double()
        return corrupted_image, noise

    def train_step(self, x: torch.FloatTensor, label: torch.FloatTensor) :
        x = x.cuda()
        label = label.cuda()
        t = torch.randint(0, self.cfg.steps, (x.size(0), ), dtype = torch.int64).cuda()
        corrupted_image, noise_gt = self.add_noise(x, t)
        corrupted_image = corrupted_image.float()
        noise_gt = noise_gt.float()
        with torch.autocast('cuda', enabled = True) :
            noise_pred = self.denoiser(corrupted_image, t, label)
        loss = F.mse_loss(noise_pred, noise_gt)
        return loss

    def run_sampling(self, save_dir: str) :
        print('sampling')
        samples, proc = self.sample((3, 32, 32), 11, 10)
        proc = torchvision.utils.make_grid((proc + 1) * 0.5 * 255, nrow = 10, normalize = False)
        samples = torchvision.utils.make_grid((samples + 1) * 0.5 * 255, nrow = 10, normalize = False)
        proc = proc.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        samples = samples.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        Image.fromarray(samples).save(os.path.join(save_dir, f'sample.png'))
        Image.fromarray(proc).save(os.path.join(save_dir, f'sample-proc.png'))

    def run_sampling_ddim(self, save_dir: str) :
        print('sampling')
        samples, proc = self.sample_ddim((3, 32, 32), 11, 10)
        proc = torchvision.utils.make_grid((proc + 1) * 0.5 * 255, nrow = 10, normalize = False)
        samples = torchvision.utils.make_grid((samples + 1) * 0.5 * 255, nrow = 10, normalize = False)
        proc = proc.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        samples = samples.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        Image.fromarray(samples).save(os.path.join(save_dir, f'sample-ddim.png'))
        Image.fromarray(proc).save(os.path.join(save_dir, f'sample-ddim-proc.png'))

    def run_sampling_ddim_acc(self, save_dir: str) :
        print('sampling')
        samples, proc = self.sample_ddim_acc((3, 32, 32), 11, 10)
        proc = torchvision.utils.make_grid((proc + 1) * 0.5 * 255, nrow = 10, normalize = False)
        samples = torchvision.utils.make_grid((samples + 1) * 0.5 * 255, nrow = 10, normalize = False)
        proc = proc.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        samples = samples.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        Image.fromarray(samples).save(os.path.join(save_dir, f'sample-ddim-acc.png'))
        Image.fromarray(proc).save(os.path.join(save_dir, f'sample-ddim-acc-proc.png'))

    def train(self, dataset, loader_args: dict, save_dir: str, n_epochs: int = 800, sample_freq: int = 1000) :
        loader = DataLoader(dataset, **loader_args)
        counter = 0
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'samples'))
        os.makedirs(os.path.join(save_dir, 'checkpoints'))

        opt = optim.AdamW(self.denoiser.parameters(), 1e-4, (0.9, 0.99))
        sch = optim.lr_scheduler.MultiStepLR(opt, [500, 700], gamma = 0.1)

        for epoch in range(n_epochs) :
            loss_avg = AvgMeter()
            self.denoiser.train()
            for x, y in tqdm(loader) :
                opt.zero_grad()
                y[:y.size(0)//10] = 10 # empty
                x = x.cuda()
                y = y.cuda()
                t = torch.randint(0, self.cfg.steps, (x.size(0), ), dtype = torch.int64).cuda()
                corrupted_image, noise_gt = self.add_noise(x, t)
                corrupted_image = corrupted_image.float()
                noise_gt = noise_gt.float()
                with torch.autocast('cuda', enabled = False) :
                    noise_pred = self.denoiser(corrupted_image, t, y)
                    loss = F.mse_loss(noise_pred, noise_gt)
                loss.backward()
                opt.step()
                self.denoiser_ema.update()
                loss_avg(loss.item())
                try :
                    if counter % sample_freq == 0 :
                        print('sampling')
                        self.denoiser.eval()
                        samples, proc = self.sample_ddim_acc((3, 32, 32), 11, 10)
                        proc = torchvision.utils.make_grid((proc + 1) * 0.5 * 255, nrow = 10, normalize = False)
                        samples = torchvision.utils.make_grid((samples + 1) * 0.5 * 255, nrow = 10, normalize = False)
                        proc = proc.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                        samples = samples.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                        Image.fromarray(samples).save(os.path.join(save_dir, 'samples', f'{counter}.png'))
                        Image.fromarray(proc).save(os.path.join(save_dir, 'samples', f'{counter}-proc.png'))
                        self.denoiser.train()
                    if counter % (sample_freq * 10) == 0 :
                        torch.save({'denoiser': self.denoiser.state_dict(), 'denoiser_ema': self.denoiser_ema.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict()}, os.path.join(save_dir, 'checkpoints', f'epoch-{epoch+1}.ckpt'))
                except Exception :
                    pass
                counter += 1
            sch.step()
            print(f'Epoch[{epoch+1}] loss={loss_avg()}')
        torch.save({'denoiser': self.denoiser.state_dict(), 'denoiser_ema': self.denoiser_ema.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict()}, os.path.join(save_dir, 'checkpoints', f'epoch-{epoch+1}.ckpt'))
                
    def load_checkpoint(self, ckpt: str) :
        ck = torch.load(ckpt)
        self.denoiser.load_state_dict(ck['denoiser'])
        self.denoiser_ema.load_state_dict(ck['denoiser_ema'])

def main() :
    unet_cfg = UnetConfig(4, 8, 64)
    cfg = DiffusionConfig(1000, unet_cfg)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    model = DiffusionModelConditional(cfg).cuda()
    #model.load_checkpoint('cifar10-ddim-dit-v2-fixed-sch-fp32-ema/checkpoints/epoch-511.ckpt')
    model.train(trainset, {'batch_size': 256, 'num_workers': 8, 'pin_memory': True, 'shuffle': True}, 'cifar10-ddim-dit-moe-fp32-ema')
    #model.run_sampling('test_sampling')
    #model.run_sampling_ddim_acc('test_sampling')

if __name__ == '__main__' :
    main()
