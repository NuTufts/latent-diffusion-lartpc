import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--px",
        type=str,
        default="389.8",
        help="P_x momentum condition"
    )
    parser.add_argument(
        "--py",
        type=str,
        default="-245.2",
        help="P_y momentum condition"
    )
    parser.add_argument(
        "--pz",
        type=str,
        default="-32.53",
        help="P_z momentum condition"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="how many batches of samples",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        nargs="?",
        help="config yaml file",
    )

    opt = parser.parse_args()


    # config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    # model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    config = OmegaConf.load(opt.base) 
    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    ## Construct and normalize momentum vector  
    momentum = [float(opt.px), float(opt.py), float(opt.pz)]
    prompt = torch.tensor(momentum, dtype=torch.float32).to(device) 
    prompt = prompt / 500 
    prompts = prompt.repeat(opt.n_samples, 1)

    print(prompts.shape)
    print("Using embedding model =", model.cond_stage_model)
    print("Using prompt =", prompt)

    if opt.n_iter > 1:
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            for n in trange(opt.n_iter, desc="Sampling"):

                ## Tokenize and embed condition prompt 
                c = model.cond_stage_model(prompts)
                # print("Embdedded cond:", c.shape)

                ## Null prompt (unconditional condition)
                uc = "" 

                shape = [model.model.diffusion_model.in_channels,
                    model.model.diffusion_model.image_size,
                    model.model.diffusion_model.image_size]

                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)

                x_samples_ddim = x_samples_ddim.cpu() 
                x_samples_ddim = x_samples_ddim.squeeze()

                # print("Output:", x_samples_ddim.shape)

                ## Plot grid of generated samples 
                if opt.n_iter == 1:
                    grid_size = 4
                    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
                    axes = axes.ravel() # Flatten axes array for easy iteration 

                    max_val = x_samples_ddim.max().item() # Max pixel value in batch (careful if all low energy)
                    print("max", max_val)
                    
                    fig.suptitle("Cond: "+str(momentum)+" / 500", fontsize=20)
                    for i in range(grid_size**2):
                        axes[i].imshow(x_samples_ddim[i] , cmap='gray', interpolation='none', vmax=max_val)
                        axes[i].axis('off')

                    plt.tight_layout()
                    plt.savefig("cond_samples/cond_samples.png")
                    print("Saved: cond_samples/cond_samples.png")

                # Save 
                if opt.n_iter > 1:
                    np.save(sample_path + "/batch_"+str(n)+".npy", x_samples_ddim)
