#!/bin/bash

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=sslsv_create_pl
#SBATCH --output=slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@v100

module purge

module load pytorch-gpu/py3/1.12.1

srun python -u tools/create_pseudolabels.py $@
EOT


# NMI (RDINO): 0.9599684972581182

# ./create_pl_ddp_jz.sh \
#     models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/embeddings_vox2_avg.pt \
#     data/pl/utt2spk_dino_km-50000_ahc-7500 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 7500 \
#     --output_format kaldi
# NMI: 0.9547139679570206

# ./create_pl_ddp_jz.sh \
#     models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/embeddings_vox2_avg.pt \
#     data/pl/utt2spk_dino_km-7500 \
#     --nb_clusters 7500 \
#     --nb_clusters_ahc 0 \
#     --output_format kaldi
# NMI: 0.9295308482917882

# ./create_pl_ddp_jz.sh \
#     models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/embeddings_vox2_avg.pt \
#     data/pl/utt2spk_dino_km-50000_ahc-5000 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 5000 \
#     --output_format kaldi
# NMI: 0.9495251148263739

# ./create_pl_ddp_jz.sh \
#     models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/embeddings_vox2_avg.pt \
#     data/pl/utt2spk_dino_km-50000_ahc-6000 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 6000 \
#     --output_format kaldi
# NMI: 0.9537478534673268

# ./create_pl_ddp_jz.sh \
#     models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/embeddings_vox2_avg.pt \
#     data/pl/utt2spk_dino_km-50000_ahc-10000 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 10000 \
#     --output_format kaldi
# NMI: 0.9511569657087564











# ./create_pl_ddp_jz.sh \
#     ~/wespeaker/examples/voxceleb/v4/exp/ssl_f-wavlm-base+_b-mhfa_km-50000_ahc-7500_dlg_ft/embeddings/vox2_dev/xvector.scp \
#     data/pl/utt2spk_dino_km-50000_ahc-7500_ipl-1 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 7500 \
#     --output_format kaldi
# NMI: 0.9668080207608786 -> 0.9676310411122394

# ./create_pl_ddp_jz.sh \
#     ~/wespeaker/examples/voxceleb/v4/exp/ssl_f-wavlm-base+_b-mhfa_km-50000_ahc-7500_dlg_ipl-1_ft/embeddings/vox2_dev/xvector.scp \
#     data/pl/utt2spk_dino_km-50000_ahc-7500_ipl-2 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 7500 \
#     --output_format kaldi
# NMI: 0.971663717040893 -> 0.9720770535538048

# ./create_pl_ddp_jz.sh \
#     ~/wespeaker/examples/voxceleb/v4/exp/ssl_f-wavlm-base+_b-mhfa_km-50000_ahc-7500_dlg_ipl-2_ft/embeddings/vox2_dev/xvector.scp \
#     data/pl/utt2spk_dino_km-50000_ahc-7500_ipl-3 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 7500 \
#     --output_format kaldi
# NMI: 0.9734085729939076 -> 0.97426505162974

# ./create_pl_ddp_jz.sh \
#     ~/wespeaker/examples/voxceleb/v4/exp/ssl_f-wavlm-base+_b-mhfa_km-50000_ahc-7500_dlg_ipl-3_ft/embeddings/vox2_dev/xvector.scp \
#     data/pl/utt2spk_dino_km-50000_ahc-7500_ipl-4 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 7500 \
#     --output_format kaldi
# NMI: 0.9747894005310549 -> 0.974974364484428

# ./create_pl_ddp_jz.sh \
#     ~/wespeaker/examples/voxceleb/v4/exp/ssl_f-wavlm-base+_b-mhfa_km-50000_ahc-7500_dlg_ipl-4_ft/embeddings/vox2_dev/xvector.scp \
#     data/pl/utt2spk_dino_km-50000_ahc-7500_ipl-5 \
#     --nb_clusters 50000 \
#     --nb_clusters_ahc 7500 \
#     --output_format kaldi
# NMI: 0.9756054748962792 -> 0.9747215825726626