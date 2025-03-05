#!/bin/bash
#SBATCH -c 1                	# Number of cores (-c)
#SBATCH -t 0-12:00      	    # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu					# Partition to submit to
#SBATCH --mem=16000          	# Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o zldm_x_sqrt_3cond_%j.out  	 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH -e zldm_x_sqrt_3cond_%j.err  	 	# File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1	 		# Request GPUs (number and/or type)
#SBATCH --signal=SIGTERM@120	# Terminate program @x seconds before time limit 

## -m trace --listfuncs
## -m trace --trace
## -m pdb
# python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir zzz_protons64_testing \
# 	--scale_lr False \
# 	--log_wandb False \
# 	--train \
# 	--gpus 0,

## Train Conditional LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
# 	--logdir protons64x_sqrt_3cond \
# 	--scale_lr False \
# 	--log_wandb True \
# 	--train \
# 	--gpus 0,

	# --resume protons64sqrt_3cond/runs/checkpoints/last.ckpt \

	# --resume protons64_cond_attn_x/runs/checkpoints/epoch=000010.ckpt \


## Get Inputs, Latents, Recos (~30 min for full val dataset)
# conda run -n ldm python3 -u main.py \
python3 -u main.py \
	--base configs/latent-diffusion/protons64-ldm-kl.yaml \
	--resume protons64sqrt_3cond/runs/checkpoints/last.ckpt \
	--logdir zzz_log2 \
	--scale_lr False \
	--save_latents "zzz_sample2" \
	--plot_latents True \
	--log_wandb False \
	--train \
	--gpus 0,

## Unconditional Sample LDM 
# conda run -n ldm python3 -u main.py \
# 	--base configs/latent-diffusion/lartpc_64-ldm-kl-8.yaml \
# 	--logdir protons64_sample \
# 	--scale_lr False \
# 	--sample True \
# 	--train \
# 	--gpus 0,

	# --custom_latents protons64_ae_posteriors_gen_unscaled.npy\


	# --save_latents protons64_ae_disc_latents_test \
	# --save_latents zz_test \
	# --plot_latents False \
# protons64_ae_disc_post_test
	# --custom_latents protons64_ae_disc_latents_gen_unscaled.npy\

	# --resume ztest_lartpc64_ldm_logs/2024-07-30T20-04-59_lartpc_64-ldm-kl-8/checkpoints/N-Step-Checkpoint_epoch=0_global_step=2000.ckpt \
	# --logdir protons64_ldm_logs \


## Flags 
# "conda run -n ldm" == run within conda enviroment ldm 
# "python3 -u" = run without buffering output 

## Submission partition (SBATCH -p)
# gpu_requeue
# gpu_test
# gpu 
# iaifi_gpu

## GPU Request 2 A100
# SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2

## Memory needed (for 64x64x1)
# 7000 (works for training)
# 6000 (mem crash for latents?) 