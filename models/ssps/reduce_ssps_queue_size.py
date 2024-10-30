import torch


input_ckpt_path = "models/ssps/voxceleb2/simclr/model_base.pt"
output_ckpt_path = "models/ssps/voxceleb2/simclr/model_base_knn.pt"
new_queue_size = 1000000

ckpt = torch.load(input_ckpt_path, map_location="cpu")

ckpt["model"]["ssps.queue_indices"] = ckpt["model"]["ssps.queue_indices"][
    -new_queue_size:
]
ckpt["model"]["ssps.queue_embeddings"] = ckpt["model"]["ssps.queue_embeddings"][
    :, -new_queue_size:
]

torch.save(ckpt, output_ckpt_path)
