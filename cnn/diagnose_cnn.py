import torch
from pathlib import Path
from torch.utils.data import DataLoader
import a1_prepare_data as prep
from dataloader import ShardDataset
from NN_Classifier import build_model
prep.IMG_RESIZE[:] = [48]
torch.set_num_threads(4)
x,y=next(iter(DataLoader(ShardDataset('train',48),batch_size=256,shuffle=True)))
s=torch.load(Path(__file__).parent / 'runs/fixed_100epochs/cnn.pt',map_location='cpu',weights_only=False)
m=build_model(s['model_type'],s['n_class'],s['n_input'],s['n1'],s['n2'],s['dropout_p'])
m.load_state_dict(s['model']); m.eval()
print('input', x.min().item(),x.max().item(),x.std().item(),'labels',y.unique().numel())
for name,module in m.named_modules():
 if isinstance(module,(torch.nn.ReLU, torch.nn.LeakyReLU)):
  module.register_forward_hook(lambda mod,inp,out,name=name: print(name,'nonzero', (out!=0).float().mean().item(),'std',out.std().item()))
z=m(x); print('prediction classes',z.argmax(1).unique().tolist(),'logit sample std',z.std(0).mean().item())
torch.nn.functional.cross_entropy(z,y).backward()
for name,p in m.named_parameters():
 print(name,'grad norm',p.grad.norm().item())
