import torch
from torch.utils.data import DataLoader
import a1_prepare_data as prep
from dataloader import ShardDataset,n_labels
from NN_Classifier import build_model
prep.IMG_RESIZE[:] = [48]
torch.manual_seed(42)
torch.set_num_threads(4)
x,y=next(iter(DataLoader(ShardDataset('train',48),batch_size=256,shuffle=True)))
x,y=x.cuda(),y.cuda()
m=build_model('cnn_v2',n_labels(),2304,512,256,.2).cuda()
o=torch.optim.Adam(m.parameters(),lr=.001)
for step in range(301):
 o.zero_grad(); z=m(x); loss=torch.nn.functional.cross_entropy(z,y); loss.backward(); o.step()
 if step in (0,50,100,200,300): print(step,loss.item(),(z.argmax(1)==y).float().mean().item(),m.features[0].weight.grad.norm().item(),flush=True)
assert loss.item()<1.0
assert (z.argmax(1)==y).float().mean().item()>.9
print('Real-batch learning check passed')
