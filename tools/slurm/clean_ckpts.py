from glob import glob
from tqdm import tqdm
import torch


models = glob("models/ssl/voxceleb2/*/*/")

for model in models:
    ckpts = glob(f"{model}/checkpoints/*")
    
    ckpts = [c.replace(model + '/checkpoints/', '') for c in ckpts]

    print(model)
    print('model_avg.pt' in ckpts)
    print('model_latest.pt' in ckpts)
    print('model_best.pt' in ckpts)
    i = 0
    for c in ckpts:
        if 'epoch' not in c:
            continue
        # print(c)
        i += 1
    print('model_epoch', i)
    print()
    print()
    # exit()

# find models/ssl/ -type f -name 'model_epoch-*.pt' -regex '.*/model_epoch-\([0-8][0-9]\|[0-9]\).pt'


# ckpts = [
#     "models/ssps/voxceleb2/simclr_e-ecapa/_simclr/checkpoints/model_latest.pt",
#     "models/ssps/voxceleb2/swav_e-ecapa/_swav/checkpoints/model_latest.pt",
#     "models/ssps/voxceleb2/vicreg_e-ecapa/_vicreg/checkpoints/model_latest.pt",
# ]
# ckpts = glob("models/ssl/voxceleb2/*/*ecapa*/checkpoints/*.pt")
# ckpts = glob("models/ssps/voxceleb2/simclr_e-ecapa/*/checkpoints/model_latest.pt")

# for ckpt in tqdm(ckpts):
#     c = torch.load(ckpt, map_location='cpu')

#     model = c["model"]

#     for k in [
#         "encoder.asb_bn.weight",
#         "encoder.asb_bn.bias",
#         "encoder.asb_bn.running_mean",
#         "encoder.asb_bn.running_var",
#         "encoder.asb_bn.num_batches_tracked"
#     ]:
#         if k not in model:
#             continue

#         print(ckpt, k, k.replace('asb', 'asp'))
#         model[k.replace('asb', 'asp')] = model.pop(k)

#     # torch.save(c, ckpt)