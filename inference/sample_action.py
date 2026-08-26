import argparse,yaml,torch
from models.action import ActionDiffusion,ActionExpertConfig
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint'); a=p.parse_args(); cfg=ActionExpertConfig(**yaml.safe_load(open(a.config))); model=ActionDiffusion(cfg)
 if a.checkpoint: model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model'])
 print(model.sample(torch.zeros(1,cfg.condition_dim)).shape)
if __name__=='__main__': main()
