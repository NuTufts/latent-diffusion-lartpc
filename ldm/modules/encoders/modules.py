import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
# import kornia
import math 

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c
 

class MomentumEmbedder(AbstractEncoder): 
    def __init__(self, n_embed=256, prompt_dim=3, device="cuda"):
        super().__init__()
        self.device = device
        self.n_embed = n_embed
        self.prompt_dim = prompt_dim  # px, py, pz (use 2 for gaussian blob toy model)

        if self.n_embed > 0: 
            self.embedding_layer = nn.Linear(1, self.n_embed)
            self.register_buffer("positional_embedding", self._get_sinusoidal_embedding(self.prompt_dim, self.n_embed))
        else: 
            ## No embedding, only positional information 
            self.embedding_layer = nn.Identity()
            self.register_buffer("positional_embedding", self._get_sinusoidal_embedding(self.prompt_dim, 1))


    def _get_sinusoidal_embedding(self, num_positions, dim):
        """Create sinusoidal positional encoding (fixed, not learnable)."""
        pe = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # shape: (num_positions, dim)

    def forward(self, momentum): 
        momentum = momentum.to(self.device)
        batch_size = momentum.shape[0]
        cond_dim = momentum.shape[1]  # should be 3 (px, py, pz)

        momentum = momentum.view(batch_size, cond_dim, 1)  # (batch, 3, 1)

        ## Project to higher dimensions 
        emb = self.embedding_layer(momentum)  # (batch, 3, n_embed)

        ## Sinusoidal positional embedding
        pos_emb = self.positional_embedding.unsqueeze(0)  # (1, 3, n_embed)

        return emb + pos_emb

    def encode(self, x): 
        return self(x)

# class MomentumEmbedder(AbstractEncoder): 

#     def __init__(self, n_embed=256, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.n_embed = n_embed
#         if self.n_embed > 0: 
#             self.embedding_layer = nn.Linear(1, self.n_embed)

#     def forward(self, momentum): 
#         momentum.to(self.device)
#         batch_size = momentum.shape[0]
#         cond_dim = momentum.shape[1]
#         momentum = momentum.view(batch_size, cond_dim, 1) #(batch, 3, 1)
#         if self.n_embed > 0: 
#             emb = self.embedding_layer(momentum) #(batch, 3, n_emb)
#         else: 
#             emb = momentum
#         return emb

#     def encode(self, x): 
#         return self(x) 

    ## Modified timestep embedding from latent-diffusion/ldm/modules/diffusionmodules/model.py
    # def MomentumEmbedder(self, momentum):

    #     assert len(momentum.shape) == 2 
    #     assert momentum.shape[1] == 3

    #     half_dim = self.embedding_dim // 2
    #     emb = math.log(10000) / (half_dim - 1)
    #     emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    #     emb = emb.to(device=momentum.device)

    #     ## Expand embedding for each element in the momentum
    #     emb = momentum.float()[:, :, None] * emb[None, :]
    #     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=2)  

    #     ## Flatten the embedding to have the final embedding_dim size
    #     emb = emb.view(emb.shape[0], -1)

    #     ## Zero pad
    #     if self.embedding_dim % 2 == 1:
    #         emb = torch.nn.functional.pad(emb, (0, 1))

    #     return emb


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


# class FrozenClipImageEmbedder(nn.Module):
#     """
#         Uses the CLIP image encoder.
#         """
#     def __init__(
#             self,
#             model,
#             jit=False,
#             device='cuda' if torch.cuda.is_available() else 'cpu',
#             antialias=False,
#         ):
#         super().__init__()
#         self.model, _ = clip.load(name=model, device=device, jit=jit)

#         self.antialias = antialias

#         self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
#         self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

#     def preprocess(self, x):
#         # normalize to [0,1]
#         x = kornia.geometry.resize(x, (224, 224),
#                                    interpolation='bicubic',align_corners=True,
#                                    antialias=self.antialias)
#         x = (x + 1.) / 2.
#         # renormalize according to clip
#         x = kornia.enhance.normalize(x, self.mean, self.std)
#         return x

#     def forward(self, x):
#         # x is assumed to be in range [-1,1]
#         return self.model.encode_image(self.preprocess(x))

if __name__ == "__main__": 

    bs = 2
    embedder = MomentumEmbedder(n_embed=16, prompt_dim=3).to('cuda')

    mom1 = [389.9, -245.2, -32.53]
    # mom1= [2,10]
    prompt1 = torch.tensor(mom1, dtype=torch.float32) / 500 
    prompt1 = prompt1.repeat(bs, 1) 
    prompt1.to('cuda')

    mom2 = [-245.2, 389.9, -32.53]
    # mom2 = [10,2]
    prompt2 = torch.tensor(mom2, dtype=torch.float32) / 500 
    prompt2 = prompt2.repeat(bs, 1) 
    prompt2.to('cuda')
    
    my_emb1 = embedder.encode(prompt1)
    my_emb2 = embedder.encode(prompt2)

    print("Cond1:", my_emb1.data.cpu().numpy())
    print("Cond2:", my_emb2.data.cpu().numpy())
    print("Cond Shape:", my_emb1.shape)

    # BERT = BERTEmbedder(n_embed=1280,n_layer=32).to('cuda')
    # my_text = "A basket of cherries" 
    # my_emb = BERT.encode(my_text)
    # # my_emb = BERT(encode(my_text)
    # print(my_emb)
    # print(my_emb.shape)
    '''
    tensor([[[ 1.0791,  0.2656,  1.3031,  ...,  2.3130,  0.4883, -1.3679],
         [ 1.0665,  0.2449,  1.3080,  ...,  2.2920,  0.4944, -1.3601],
         [ 1.1391,  0.2267,  1.3406,  ...,  2.2966,  0.4998, -1.2900],
         ...,
         [ 1.1099,  0.2887,  1.3186,  ...,  2.2979,  0.4738, -1.4082],
         [ 1.0878,  0.2788,  1.3758,  ...,  2.2772,  0.4928, -1.3865],
         [ 1.0666,  0.3105,  1.3726,  ...,  2.2346,  0.5152, -1.3982]]],
       device='cuda:0', grad_fn=<SliceBackward0>)

    torch.Size([1, 77, 1280])
    '''