import argparse,yaml,torch
from torch.utils.data import DataLoader
from datasets.action_dataset import ActionDataset
from models.action import ActionDiffusion,ActionExpertConfig
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--output',default='action_expert.pt'); a=p.parse_args(); cfg=ActionExpertConfig(**yaml.safe_load(open(a.config))); ds=ActionDataset(a.data); model=ActionDiffusion(cfg); opt=torch.optim.AdamW(model.parameters(),1e-4)
 for _ in range(a.epochs):
  for b in DataLoader(ds,32,shuffle=True): opt.zero_grad(); loss=model.loss(b['actions'],b['condition']); loss.backward(); opt.step()
  print(f'loss={loss.item():.6f}')
 torch.save({'config':cfg.__dict__,'model':model.state_dict()},a.output)
if __name__=='__main__': main()
