#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-03:00      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=12000          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zldm_cond_sample_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zldm_cond_sample_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 	# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 

export CUDA_VISIBLE_DEVICES=0

## Unconditional Sampling (need modify config)
# conda run -n ldm python3 -u scripts/sample_diffusion.py \
# 	--resume protons64_ldm_logs/2024-09-24T22-43-29_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir protons64_sample \
# 	--n_samples 50000 \
# 	--batch_size 256 \
# 	--custom_steps 200


checkpoint_path="protons64sqrt_3cond/runs/checkpoints/epoch=000125.ckpt"
# checkpoint_path="protons64x_sqrt_3cond/runs/checkpoints/epoch=000145.ckpt"

## Conditional Sampling 
# conda run -n ldm python3 scripts/txt2img.py \
python3 scripts/txt2img.py \
	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
	--outdir one_mom_sample \
	--px "314.0" \
	--py "-126.4" \
	--pz "249.1" \
	--ddim_steps 200 \
	--ddim_eta 0.0 \
	--W 64 \
	--H 64 \
	--n_samples 128 \
	--n_iter 250 \
	--scale 5.0 \
	--resume "$checkpoint_path"

	## Down right diag " 360.0, 168.6, -66.06"
	## Down left diag: " 389.8, -245.2, -32.53"
	## Straight up: "-382.5, 2.785, -16.33


	## Normalized Momentum 
	# --resume protons64_cond_ldm_v1/2024-11-12T23-06-03_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
	
	## Bimodal Momentum (Px)
	# --resume protons64_cond_ldm_x/2024-11-13T11-24-13_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \


## Pretrained
# python3 scripts/txt2img_orig.py 


## Sample LDM - using main.py
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train 


# conda run -n ldm python3 -u scripts/sample_diffusion.py \
# 	--resume protons64_ldm_logs_v3/2024-08-30T01-25-12_lartpc_64-ldm-kl-8/checkpoints/last.ckpt \
# 	--logdir protons64_ldm_sample_logs \
# 	--vanilla_sample True \
# 	--n_samples 50000 \
# 	--batch_size 128 

	#  -c <\#ddim steps> -e <\#eta> 

## Sample LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train \
# 	--gpus 0,
