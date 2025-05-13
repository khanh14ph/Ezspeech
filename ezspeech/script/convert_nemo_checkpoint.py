import torch
from omegaconf import OmegaConf
savepath="/home4/khanhnd/models/parakeet/ezspeech.ckpt"
checkpoint=torch.load("/home4/khanhnd/models/parakeet/model_weights.ckpt",weights_only=False,map_location=torch.device('cpu'))
final=dict()
module_set=[]
for i in checkpoint.keys():
    module_set.append(i.split(".")[0])
module_set=list(set(module_set))
for i in module_set:
    final[i]=dict()
    for j in checkpoint.keys():
        name,new_name=j.split(".",1)
        if name==i:
            final[i][new_name]=checkpoint[j]
config=OmegaConf.load("/home4/khanhnd/models/parakeet/config.yaml")
checkpoint = {
        "state_dict": final,
        "hyper_parameters": config,
    }
print(final.keys())
torch.save(checkpoint, savepath)