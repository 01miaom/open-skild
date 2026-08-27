from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass
class ActionExpertConfig:
    action_dim:int=14; horizon:int=16; condition_dim:int=256; d_model:int=256; nhead:int=8; num_layers:int=6; dropout:float=.1; diffusion_steps:int=1000; sampling_steps:int=50

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self,t):
        half=self.dim//2; scale=math.log(10000)/max(half-1,1); e=torch.exp(torch.arange(half,device=t.device)*-scale); e=t.float()[:,None]*e[None]; return torch.cat((e.sin(),e.cos()),-1)

class ActionDiffusion(nn.Module):
    """Conditional DDPM action expert. Actions are [B, horizon, action_dim]."""
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.action_in=nn.Linear(cfg.action_dim,cfg.d_model)
        self.time=nn.Sequential(SinusoidalEmbedding(cfg.d_model),nn.Linear(cfg.d_model,cfg.d_model),nn.SiLU(),nn.Linear(cfg.d_model,cfg.d_model))
        self.condition=nn.Sequential(nn.Linear(cfg.condition_dim,cfg.d_model),nn.SiLU(),nn.Linear(cfg.d_model,cfg.d_model))
        self.pos=nn.Parameter(torch.zeros(1,cfg.horizon,cfg.d_model))
        layer=nn.TransformerEncoderLayer(cfg.d_model,cfg.nhead,4*cfg.d_model,cfg.dropout,batch_first=True,norm_first=True)
        self.net=nn.TransformerEncoder(layer,cfg.num_layers); self.out=nn.Sequential(nn.LayerNorm(cfg.d_model),nn.Linear(cfg.d_model,cfg.action_dim))
        beta=torch.linspace(1e-4,2e-2,cfg.diffusion_steps); self.register_buffer('betas',beta); self.register_buffer('alphas',1-beta); self.register_buffer('abar',torch.cumprod(1-beta,0))
    def predict_noise(self,noisy,t,condition):
        if condition.ndim == 2:
            condition = condition[:, None, :]
        action_tokens = self.action_in(noisy) + self.pos[:, :noisy.shape[1]] + self.time(t)[:, None]
        condition_tokens = self.condition(condition)
        fused = self.net(torch.cat([condition_tokens, action_tokens], dim=1))
        return self.out(fused[:, -noisy.shape[1]:])
    def loss(self,actions,condition):
        t=torch.randint(0,self.cfg.diffusion_steps,(actions.shape[0],),device=actions.device); noise=torch.randn_like(actions); a=self.abar[t].sqrt()[:,None,None]; s=(1-self.abar[t]).sqrt()[:,None,None]; return nn.functional.mse_loss(self.predict_noise(a*actions+s*noise,t,condition),noise)
    @torch.no_grad()
    def sample(self,condition,horizon=None,steps=None):
        n=condition.shape[0]; h=horizon or self.cfg.horizon; x=torch.randn(n,h,self.cfg.action_dim,device=condition.device); times=torch.linspace(self.cfg.diffusion_steps-1,0,steps or self.cfg.sampling_steps,device=condition.device).long()
        for i,t in enumerate(times):
            ti=t.expand(n); beta,alpha,abar=self.betas[t],self.alphas[t],self.abar[t]; eps=self.predict_noise(x,ti,condition); mean=(x-beta/(1-abar).sqrt()*eps)/alpha.sqrt(); x=mean if i==len(times)-1 else mean+beta.sqrt()*torch.randn_like(x)
        return x
