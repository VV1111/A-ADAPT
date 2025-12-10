

import os
join = os.path.join

import math
from typing import Type

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything.modeling.common import MLPBlock,ImagePromptEncoderDirectional
from data.datainfo import *
import matplotlib.pyplot as plt


def get_index(map_idx):
    idx = map_idx
    for i in range(list(map_idx.items())[-1][0]):
        if i not in idx:
            idx[i] = 0
    idx = dict(sorted(idx.items(), key=lambda x: x[0]))
    return torch.LongTensor(list(idx.values()))


class MoEAdaptMLPBlock(nn.Module):
    def __init__(
        self,
        mlp: MLPBlock,
        expert_num: int,
        embedding_num: int,
        embedding_dim: int,
        adapter_bn: int = 64,
        adapter_act: Type[nn.Module] = nn.GELU,
        adapter_dropout: float = 0.1,
        adapter_scalar: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.expert_num = expert_num

        dim = mlp.embedding_dim
        self.adapter_input_embed = nn.Linear(dim, embedding_dim)
        self.adapter_gate = nn.Linear(embedding_dim * embedding_num, expert_num)
        self.adapter_organ_gate = nn.Linear(embedding_dim * 4, 4)

        self.adapter_down = nn.ModuleList([
            nn.Linear(dim, adapter_bn) for _ in range(expert_num)
        ])
        self.adapter_up = nn.ModuleList([
            nn.Linear(adapter_bn, dim) for _ in range(expert_num)
        ])
        self.adapter_act = adapter_act()
        self.adapter_dropout = adapter_dropout
        if adapter_scalar is None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = adapter_scalar

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.expert_num):
            nn.init.kaiming_uniform_(self.adapter_down[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down[i].bias)
            nn.init.zeros_(self.adapter_up[i].weight)
            nn.init.zeros_(self.adapter_up[i].bias)

    def forward(self, x: torch.Tensor,
                modal: torch.Tensor, organ: torch.Tensor) -> torch.Tensor:
        adpt = []
        for i in range(self.expert_num):
            adpt.append(self.adapter_up[i](nn.functional.dropout(
                self.adapter_act(self.adapter_down[i](x)),
                p=self.adapter_dropout, training=self.training)))
        adpt = torch.stack(adpt, dim=1)

        organ_0, organ_1, organ_2, organ_3, task = organ
        organ = torch.cat([organ_0, organ_1, organ_2, organ_3], dim=-1)
        organ_gate = self.adapter_organ_gate(organ)
        organ_gate = torch.softmax(organ_gate, dim=-1)
        organ = torch.stack([organ_0, organ_1, organ_2, organ_3], dim=1)
        organ = torch.einsum('bec,be->bc', organ, organ_gate)

        input = x.mean(dim=1).mean(dim=1)
        input = self.adapter_input_embed(input)

        gate = torch.cat([organ, task, modal, input], dim=-1)
        gate = self.adapter_gate(gate)
        gate = torch.softmax(gate, dim=1)

        adpt = torch.einsum('bemnc,be->bmnc', adpt, gate)
        return self.mlp(x) + adpt * self.adapter_scale, gate




class AAdapt(nn.Module):
    """Applies Tree MoE Adapter to SAM's image encoder.

    Args:
        sam: segment anything model, see 'segment_anything' dir
        bottleneck_dim: bottleneck dimension of adapter
        embedding_dim: modal and organ embedding dimension
        expert_num: number of experts in MoE adapter
        pos: which layer to apply adapter
    """

    def __init__(self, sam: Sam, bottleneck_dim: int, embedding_dim: int, 
                 expert_num: int, pos: list = None,fine_tune_last_n_blocks:int=0,active_freq:bool=True,active_tube:bool=True):
        super(AAdapt, self).__init__()

        assert bottleneck_dim > 0
        assert embedding_dim > 0
        assert expert_num > 0

        # assign Adapter layer position (all layers by default)
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(sam.image_encoder.blocks)))

        # freeze SAM image and prompt encoder
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False

        # modality and organ embedding index
        modal_index = get_index(modal_map_idx)
        organ_index_1 = get_index(organ_level_1_map_idx)
        organ_index_2 = get_index(organ_level_2_map_idx)
        organ_index_3 = get_index(organ_level_3_map_idx)

        sam.image_encoder.register_buffer('modal_index', modal_index, False)
        sam.image_encoder.register_buffer('organ_index_1', organ_index_1, False)
        sam.image_encoder.register_buffer('organ_index_2', organ_index_2, False)
        sam.image_encoder.register_buffer('organ_index_3', organ_index_3, False)

        modal_embed = nn.Embedding(len(modal_map_idx), embedding_dim)
        organ_embed_0 = nn.Embedding(1, embedding_dim)
        organ_embed_1 = nn.Embedding(len(organ_level_1_map_idx), embedding_dim)
        organ_embed_2 = nn.Embedding(len(organ_level_2_map_idx), embedding_dim)
        organ_embed_3 = nn.Embedding(len(organ_level_3_map_idx), embedding_dim)
        organ_embed_4 = nn.Embedding(len(task_list)+1, embedding_dim)
        nn.init.zeros_(modal_embed.weight)
        nn.init.zeros_(organ_embed_0.weight)
        nn.init.zeros_(organ_embed_1.weight)
        nn.init.zeros_(organ_embed_2.weight)
        nn.init.zeros_(organ_embed_3.weight)
        nn.init.zeros_(organ_embed_4.weight)

        sam.image_encoder.modal_embed = modal_embed
        sam.image_encoder.organ_embed = nn.ModuleList([
            organ_embed_0, organ_embed_1, 
            organ_embed_2, organ_embed_3, 
            organ_embed_4, 
        ])

        # apply Adapter to SAM image encoder
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue

            # create moe adapter layers
            blk.mlp = MoEAdaptMLPBlock(
                blk.mlp,
                expert_num=expert_num,
                embedding_num=4,
                embedding_dim=embedding_dim,
                adapter_bn=bottleneck_dim,
            )

        self.sam = sam

        self.active_freq = active_freq
        self.active_tube = active_tube

        if fine_tune_last_n_blocks > 0:
            total_blocks = len(self.sam.image_encoder.blocks)
            start = max(0, total_blocks - fine_tune_last_n_blocks)
            for i, blk in enumerate(self.sam.image_encoder.blocks):
                if i >= start:
                    for p in blk.parameters():
                        p.requires_grad = True
            
        self.img_prompt_encoder = ImagePromptEncoderDirectional(
            embed_dim=256,       
            image_size=256,             
            image_embedding_size=16,  
            in_ch=3,                           
            base_ch=64,
            num_dirs=8,
            morph_kernel_size=9,
        )

    def save_parameters(self) -> dict:
        r"""save both adapter and mask decoder parameters.
        """
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        adapter_tensors = {k: v for k, v in state_dict.items() if "adapter" in k}

        image_encoder_tensors = {}
        for k, v in state_dict.items():
            if (
                ("image_encoder.modal_embed" in k)
                or ("image_encoder.organ_embed" in k)
                or ("image_encoder.blocks" in k)
                or ("image_encoder.freq" in k)
                or ("image_encoder.tube" in k)   
            ):
                image_encoder_tensors[k] = v

        prompt_encoder_tensors = {k: v for k, v in state_dict.items() if "prompt_encoder" in k}

        mask_decoder_tensors = {k: v for k, v in state_dict.items() if "mask_decoder" in k}

        full_state = self.state_dict()  # 包含 sam + img_prompt_encoder
        img_prompt_tensors = {k: v for k, v in full_state.items() if "img_prompt_encoder" in k}

        merged_dict = {
            **adapter_tensors,
            **image_encoder_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
            **img_prompt_tensors
        }
        return merged_dict

    def load_parameters(self, state_dict) -> None:
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # 1) adapter
        adapter_keys = [k for k in sam_keys if ("adapter" in k) and (k in state_dict)]
        adapter_new_state_dict = {k: state_dict[k] for k in adapter_keys}
        sam_dict.update(adapter_new_state_dict)

        # 2) image encoder
        image_encoder_keys = [
            k for k in sam_keys
            if (
                ("modal_embed" in k)
                or ("organ_embed" in k)
                or ("image_encoder.blocks" in k)
                or ("freq" in k)
                or ("tube" in k)
            ) and (k in state_dict)
        ]
        image_encoder_new_state_dict = {k: state_dict[k] for k in image_encoder_keys}
        sam_dict.update(image_encoder_new_state_dict)

        # 3) prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if ("prompt_encoder" in k) and (k in state_dict)]
        prompt_encoder_new_state_dict = {k: state_dict[k] for k in prompt_encoder_keys}
        sam_dict.update(prompt_encoder_new_state_dict)

        # 4) mask decoder
        mask_decoder_keys = [k for k in sam_keys if ("mask_decoder" in k) and (k in state_dict)]
        mask_decoder_new_state_dict = {k: state_dict[k] for k in mask_decoder_keys}
        sam_dict.update(mask_decoder_new_state_dict)

        self.sam.load_state_dict(sam_dict)

        full_model_dict = self.state_dict()
        img_prompt_keys = [k for k in full_model_dict.keys() if ("img_prompt_encoder" in k) and (k in state_dict)]
        if len(img_prompt_keys) > 0:
            for k in img_prompt_keys:
                full_model_dict[k] = state_dict[k]
            self.load_state_dict(full_model_dict, strict=False)

    def forward(self, data,visualize: bool = False,save_dir=None, tag=None):
        img, box = data['img'], data['box']
        modal, organ = data['modal'], data['organ']

        # modal and organ embedding
        B = img.shape[0]

        modal_index = self.sam.image_encoder.modal_index[modal]
        modal_embed = self.sam.image_encoder.modal_embed(modal_index)
        # print('organ',organ) 
        organ_1, organ_2, organ_3, organ_4 = organ
        # print('organ_1, organ_2, organ_3, organ_4 ',organ_1, organ_2, organ_3, organ_4 )
        organ_index_0 = torch.zeros(B, dtype=torch.long, device=img.device)
        organ_embed_0 = self.sam.image_encoder.organ_embed[0](organ_index_0)
        organ_index_1 = self.sam.image_encoder.organ_index_1[organ_1]
        organ_embed_1 = self.sam.image_encoder.organ_embed[1](organ_index_1)
        organ_index_2 = self.sam.image_encoder.organ_index_2[organ_2]
        organ_embed_2 = self.sam.image_encoder.organ_embed[2](organ_index_2)
        organ_index_3 = self.sam.image_encoder.organ_index_3[organ_3]
        organ_embed_3 = self.sam.image_encoder.organ_embed[3](organ_index_3)

        organ_embed_4 = self.sam.image_encoder.organ_embed[4](organ_4)
        organ_embed = (organ_embed_0, organ_embed_1, organ_embed_2, organ_embed_3, organ_embed_4)

        # sparse_embeddings, dense_embeddings = self.img_prompt_encoder(img)
        if visualize:
            sparse_embeddings, dense_embeddings, feat, feat_dir = \
                self.img_prompt_encoder(img, return_intermediate=True)
        else:
            sparse_embeddings, dense_embeddings = self.img_prompt_encoder(img)

        # adapter image encoder
        input_image = self.sam.preprocess(img) # (B, 3, 1024, 1024)

        image_embedding, expert_activation = self.sam.image_encoder(input_image, modal_embed, organ_embed,active_freq =self.active_freq,active_tube = self.active_tube) # (B, 256, 64, 64)

        # predicted masks
        mask_predictions, _,_,_ = self.sam.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (B, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
          )

        if visualize:
            if save_dir is None:
                save_dir = "vis_directional"
            os.makedirs(save_dir, exist_ok=True)
            if tag is None:
                tag = "case"

            self._visualize_directional_process(
                img[0], feat[0], feat_dir[0], dense_embeddings[0],
                os.path.join(save_dir, f"case_{tag}")
            )
         

        return mask_predictions

    def _visualize_directional_process(self, img_b, feat_b, feat_dir_b, dense_b, save_prefix: str):

        # -------  -------
        img_np = img_b[0].detach().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        plt.figure()
        plt.imshow(img_np, cmap="gray")
        plt.axis("off")
        plt.title("Input (resized)")
        plt.savefig(save_prefix + "_0_input.png", bbox_inches="tight", dpi=300)
        plt.close()

        # -------  -------
        feat_np = feat_b.mean(0).detach().cpu().numpy()
        feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-8)

        plt.figure()
        plt.imshow(feat_np, cmap="hot")
        plt.axis("off")
        plt.title("Downsampled feature")
        plt.savefig(save_prefix + "_1_downsample.png", bbox_inches="tight", dpi=300)
        plt.close()

        # ------- -------
        C_total, H_emb, W_emb = feat_dir_b.shape
        num_dirs = self.img_prompt_encoder.morph_block.num_dirs
        base_ch = self.img_prompt_encoder.morph_block.in_ch

        dir_maps = feat_dir_b[:num_dirs].detach().cpu().numpy()  # [num_dirs, H_emb, W_emb]

        n_cols = num_dirs
        plt.figure(figsize=(1.8*n_cols, 2.0))
        for i in range(num_dirs):
            ax = plt.subplot(1, n_cols, i+1)
            m = dir_maps[i]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            ax.imshow(m, cmap="hot")
            ax.axis("off")
            ax.set_title(f"{int(180*i/num_dirs)}°", fontsize=8)
        plt.suptitle("Directional responses (first channel)", fontsize=10)
        plt.savefig(save_prefix + "_2_dir_responses.png", bbox_inches="tight", dpi=300)
        plt.close()

        # ------- dense embedding -------
        dense_np = dense_b.mean(0).detach().cpu().numpy()
        dense_np = (dense_np - dense_np.min()) / (dense_np.max() - dense_np.min() + 1e-8)

        plt.figure()
        plt.imshow(dense_np, cmap="hot")
        plt.axis("off")
        plt.title("Projected dense embedding")
        plt.savefig(save_prefix + "_3_dense.png", bbox_inches="tight", dpi=300)
        plt.close()
