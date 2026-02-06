from __future__ import annotations

import torch
import torch.nn as nn
from .utils import (
    AtomFeaturePlus,
    EdgeFeaturePlus,
    SE3InvariantKernel,
    _build_attn_mask,
    MovementPredictionHead,
    MaskLMHead,
    build_padding_only_attn_mask,
)
from .encoder import EncoderBlock


class MultiMolModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args  # keep reference ¨C other modules rely on it

        # --------------------------------------------------------------
        # Embeddings & basic feature extractors
        # --------------------------------------------------------------
        d_model = getattr(args, "encoder_embed_dim", 768)
        self.embed_tokens = nn.Embedding(
            128, d_model, getattr(args, "padding_idx", 0)
        )

        self.atom_feature = AtomFeaturePlus(
            num_atom=getattr(args, "num_atom", 512),
            num_degree=getattr(args, "num_degree", 128),
            hidden_dim=d_model,
            wo_node=getattr(args, "wo_node", False),
            wo_atom_feat=getattr(args, "wo_atom_feat", None)
            
        )
        self.edge_feature = EdgeFeaturePlus(
            pair_dim=getattr(args, "pair_embed_dim", 512),
            num_edge=getattr(args, "num_edge", 64),
            num_spatial=getattr(args, "num_spatial", 512),
            wo_spd=getattr(args, "wo_spd", False),
            wo_edge=getattr(args, "wo_edge", False),            
        )

        # --------------------------------------------------------------
        # Encoder (token-pair joint reasoning)
        # --------------------------------------------------------------
        self.encoder = EncoderBlock(
            num_encoder_layers=getattr(args, "encoder_layers", 12),
            embedding_dim=d_model,
            pair_dim=getattr(args, "pair_embed_dim", 512),
            pair_hidden_dim=getattr(args, "pair_hidden_dim", 64),
            ffn_embedding_dim=getattr(args, "encoder_ffn_embed_dim", 3072),
            num_attention_heads=getattr(args, "encoder_attention_heads", 48),
            dropout=getattr(args, "dropout", 0.1),
            attention_dropout=getattr(args, "attention_dropout", 0.1),
            activation_dropout=getattr(args, "activation_dropout", 0.1),
            activation_fn=getattr(args, "activation_fn", "gelu"),
            droppath_prob=getattr(args, "droppath_prob", 0.0),
            pair_dropout=getattr(args, "pair_dropout", 0.25),
            wo_triopm=getattr(args, "wo_triopm", False),
            wo_pair=getattr(args, "wo_pair", False),
        )

        # 3-D geometric bias (shared across layers)
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=getattr(args, "pair_embed_dim", 512),
            num_pair=128*128,
            num_kernel=getattr(args, "num_kernel", 128),
            std_width=getattr(args, "gaussian_std_width", 1.0),
            start=getattr(args, "gaussian_mean_start", 0.0),
            stop=getattr(args, "gaussian_mean_stop", 9.0),
        )
        
        # --------------------------------------------------------------
        # Pretrain Heads
        # --------------------------------------------------------------
        self.lm_head = MaskLMHead(
            embed_dim=getattr(args, "encoder_embed_dim", 768),
            output_dim=128,
            weight=self.embed_tokens.weight,
        )
        
        self.movement_pred_head = MovementPredictionHead(
            getattr(args, "encoder_embed_dim", 768),
            getattr(args, "pair_embed_dim", 512),
            getattr(args, "encoder_attention_heads", 48),
        )

        # --------------------------------------------------------------
        # Downstream NN (dimension based on args)
        # --------------------------------------------------------------
        self.reg_head = nn.Sequential(
        nn.Linear(getattr(args, "encoder_embed_dim", 768), 128),
        nn.GELU(),
        nn.Linear(128, 32),
        nn.GELU(),
        nn.Linear(32, getattr(args, "num_tasks", 1))
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch):
        """Args:
            batch (dict): see dataloader for full specification
        Returns:
            node_repr (Tensor): [B, T, D]
            pair_repr (Tensor): [B, T, T, D_p]
        """
        # ---------- unpack -------------------------------------------------
        atom_mask = batch["atom_mask"]  # [B, N]
        seg_id = batch["segment_id"]  # [B, N]
        pos = batch["src_pos"]  # [B, N, 3]
        pair_type = batch["pair_type"]  # [B, N, N]
        token_ids = batch["src_token"]  # [B, N]

        B, N = atom_mask.shape
        n_seg = batch["seg_feat"].size(1)
        special_len = 2 + n_seg  # CLS + GLOB + SEG
        total_len = special_len + N

        # ---------- node embedding -----------------------------------------
        token_feat = self.embed_tokens(token_ids)
        node_repr = self.atom_feature(batch, token_feat)  # [B, T, D]
        if node_repr.dtype == torch.float32 and torch.is_autocast_enabled():
            node_repr = node_repr.to(torch.get_autocast_gpu_dtype())

        # ---------- masks ---------------------------------------------------
        attn_mask = _build_attn_mask(
            batch["base_mask"],
            seg_id,
            atom_mask,
            batch["seg_valid_mask"],
            glob_valid=batch["glob_valid_mask"],
            num_heads=self.args.encoder_attention_heads,
        )
        
        if not self.args.wo_pair:
            attn_mask = _build_attn_mask(
                batch["base_mask"],
                seg_id,
                atom_mask,
                batch["seg_valid_mask"],
                glob_valid=batch["glob_valid_mask"],
                num_heads=self.args.encoder_attention_heads,
            )
        else:
            attn_mask = build_padding_only_attn_mask(
            atom_mask,
            batch["seg_valid_mask"],
            glob_valid=batch["glob_valid_mask"],
            num_heads=self.args.encoder_attention_heads,
        )

        # ---------- pair-wise bias -----------------------------------------
        pair_repr = node_repr.new_zeros(
            B, total_len, total_len, self.args.pair_embed_dim, dtype=node_repr.dtype
        )
        pair_repr = self.edge_feature(batch, pair_repr)

        # 3-D SE(3) bias ¨C inside each segment only
        if self.args.wo_geom_3d:
            pass
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
            dist = delta_pos.norm(dim=-1)  # [B, N, N]
            geom_bias = self.se3_invariant_kernel(dist.detach(), pair_type.long())
            same_seg = (seg_id.unsqueeze(-1) == seg_id.unsqueeze(-2)).unsqueeze(-1)
            geom_bias.masked_fill_(~same_seg, 0.0)
            pair_repr[:, special_len:, special_len:, :].add_(geom_bias)
                
        # ---------- padding masks ------------------------------------------
        cls_mask  = atom_mask.new_ones(B, 1, dtype=torch.bool)
        glob_mask = batch["glob_valid_mask"].bool()
        seg_mask  = batch["seg_valid_mask"].bool()
        node_mask = torch.cat([cls_mask, glob_mask, seg_mask, atom_mask.bool()], dim=1)
        pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)

        # ---------- encoder --------------------------------------------------
        node_repr, pair_repr = self.encoder(
            node_repr,
            pair_repr,
            atom_mask=node_mask,
            pair_mask=pair_mask,
            attn_mask=attn_mask,
        )
        
        # ---------- Downstream REG head --------------------------------------------------
        if self.args.main_task == "finetune":
            mol_rep = node_repr[:, 0, :]
            pred_val = self.reg_head(mol_rep)
            return pred_val
            
        # ---------- Pretrain: Masked-Atom Prediction & Coordinate Reconstruction ---------------
        if self.args.main_task == "pretrain":
        
            # ---- 3.1 Masked Atom Prediction ----
            atom_repr = node_repr[:, special_len:, :]
            logits = self.lm_head(atom_repr)       # [B,N,V] ¡ª predicted token logits for each atom        
            # ---- 3.2 Coordinate Reconstruction ----
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
            atom_repr = node_repr[:, special_len:, :]        # [B, N, d]
            pair_repr_a = pair_repr[:, special_len:, special_len:, :]  # [B, N, N, d_p]
            attn_mask_a = attn_mask[:, :, special_len:, special_len:]  # [B, H, N, N]
            delta = self.movement_pred_head(
                atom_repr,                         # [B,N,d] ¡ª node representations
                pair_repr_a,
                attn_mask_a,                         # [B,N,N,p] ¡ª pairwise representations
                delta_pos.detach(),                   # Noisy input coordinates (¦¤pos will be used inside the head)
            )                                      # Returns ¦¤xyz, shape [B,N,3]        
            pred_pos = pos + delta    # [B,N,3] ¡ª recovered positions        
            # ---- 3.3 Distance Prediction ----
            pred_dist = torch.cdist(pred_pos, pred_pos)        
            return logits, pred_pos, pred_dist               