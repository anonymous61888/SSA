import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
from .attention import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from .position_encoding import PositionEmbeddingSine1D


def match_from_embds(tgt_embds, cur_embds):
    # embeds shape (q, c)
    cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
    tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
    cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))

    cost_embd = 1 - cos_sim

    C = 1.0 * cost_embd
    C = C.detach().cpu()

    indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
    indices = indices[1]  # permutation that makes current aligns to target

    return indices


class SSA(nn.Module):

    @configurable
    def __init__(
            self,
            in_channels,
            aux_loss,
            *,
            hidden_dim: int,
            num_frame_queries: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            enc_layers: int,
            dec_layers: int,
            enc_window_size: int,
            pre_norm: bool,
            enforce_input_project: bool,
            num_frames: int,
            num_classes: int,
            clip_last_layer_num: bool,
            conv_dim: int,
            mask_dim: int,
            sim_use_clip: list,
            use_sim: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.fusion = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.clip_last_layer_num = clip_last_layer_num

        self.enc_layers = enc_layers
        self.window_size = enc_window_size
        self.temporal_enhance = self.enc_layers
        self.sim_use_clip = sim_use_clip
        self.use_sim = use_sim
        self.aux_loss = aux_loss

        if self.temporal_enhance:
            self.short_temporal = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim,
                          kernel_size=5, stride=1,
                          padding='same', padding_mode='replicate'),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_dim, hidden_dim,
                          kernel_size=3, stride=1,
                          padding='same', padding_mode='replicate'),
            )
            self.temporal_norm = nn.LayerNorm(hidden_dim)

            self.long_temporal = SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        self.enc_pos = PositionEmbeddingSine1D(num_pos_feats=hidden_dim//2)
        self.time_aware_attn = nn.Linear(hidden_dim, 1)

        for _ in range(self.num_layers):
            self.fusion.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm
                )
            )

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.vita_mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.vita_mask_features)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries  # video-level queries
        # learnable query features
        # self.query_feat = nn.Embedding(num_queries, hidden_dim)  # 不初始化

        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.fq_pos = nn.Embedding(num_frame_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj_dec = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.input_proj_dec = nn.Sequential()
        self.src_embed = nn.Identity()

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        if self.use_sim:
            self.sim_embed_frame = nn.Linear(hidden_dim, hidden_dim)
            if self.sim_use_clip:
                self.sim_embed_clip = nn.Linear(hidden_dim, hidden_dim)

    @classmethod
    def from_config(cls, cfg, in_channels):
        ret = {}
        ret["in_channels"] = in_channels

        ret["hidden_dim"] = cfg.MODEL.VITA.HIDDEN_DIM
        ret["num_frame_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["num_queries"] = cfg.MODEL.VITA.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.VITA.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.VITA.DIM_FEEDFORWARD

        assert cfg.MODEL.VITA.DEC_LAYERS >= 1
        ret["enc_layers"] = cfg.MODEL.VITA.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.VITA.DEC_LAYERS
        ret["enc_window_size"] = cfg.MODEL.VITA.ENC_WINDOW_SIZE
        ret["pre_norm"] = cfg.MODEL.VITA.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.VITA.ENFORCE_INPUT_PROJ

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        ret["clip_last_layer_num"] = cfg.MODEL.VITA.LAST_LAYER_NUM

        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["sim_use_clip"] = cfg.MODEL.VITA.SIM_USE_CLIP
        ret["use_sim"] = cfg.MODEL.VITA.SIM_WEIGHT > 0.0

        return ret

    def forward(self, frame_query, sentence_feat, lang_mask):
        """
        L: Number of Layers.
        B: Batch size.
        T: Temporal window size. Number of frames per video.
        C: Channel size.
        fQ: Number of frame-wise queries from IFC.
        cQ: Number of clip-wise queries to decode Q.
        """

        if not self.training:
            frame_query = frame_query[[-1]]

        L, BT, fQ, C = frame_query.shape
        B = BT // self.num_frames if self.training else 1
        T = self.num_frames if self.training else BT // B

        frame_query = frame_query.reshape(L * B, T, fQ, C)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous()
        frame_query = self.input_proj_dec(frame_query)

        src = self.src_embed(frame_query)
        for i in range(1, T):
            for j in range(L * B):  # [T, fQ, LB, C]
                indices = match_from_embds(src[i - 1, :, j, :],
                                           src[i, :, j, :])
                src[i, :, j, :] = src[i, indices, j, :]

        if self.temporal_enhance:
            src = self.temporal_enhancement(src)

        time_weight = self.time_aware_attn(src)
        time_weight = F.softmax(time_weight, 0)
        output = (src * time_weight).sum(0)

        frame_query = frame_query[:T].flatten(0, 1)  # TfQ, LB, C

        if self.use_sim:
            pred_fq_embed = self.sim_embed_frame(frame_query)  # TfQ, LB, C
            pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
        else:
            pred_fq_embed = None

        src = src.flatten(0, 1)
        dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L * B, 1).flatten(0, 1)  # TfQ, LB, C

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C

        # NOTE sentence-query cross attention
        # setence_feat [BT, 1, C]
        sentence_feat = sentence_feat.repeat(1, L, 1)  # [BT, L, C]
        sentence_feat = sentence_feat.reshape(B, T, L, C)  # [B, T, L, C]
        sentence_feat = sentence_feat.permute(1, 2, 0, 3).contiguous()  # [T, L, B, C]
        sentence_feat = sentence_feat.reshape(T, L * B, C)

        text_features = sentence_feat

        decoder_outputs = []
        for i in range(self.num_layers):
            # attention: cross-attention first
            output = self.fusion[i](
                output, text_features,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=query_embed
            )

            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=dec_pos, query_pos=query_embed
            )
            # self-attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            if (self.training and self.aux_loss) or (i == self.num_layers - 1):  # only training mode, has D > 1
                dec_out = self.decoder_norm(output)  # cQ, LB, C
                dec_out = dec_out.transpose(0, 1)  # LB, cQ, C
                decoder_outputs.append(dec_out.view(L, B, self.num_queries, C))

        decoder_outputs = torch.stack(decoder_outputs, dim=0)  # D, L, B, cQ, C

        D, L, B, cQ, C = decoder_outputs.shape
        pred_cls = self.class_embed(decoder_outputs)

        pred_mask_embed = self.mask_embed(decoder_outputs)
        if self.use_sim and self.sim_use_clip:
            pred_cq_embed = self.sim_embed_clip(decoder_outputs)
        else:
            pred_cq_embed = [None] * self.num_layers

        out = {
            'pred_logits': pred_cls[-1],  # return last decoder_outputs
            'pred_mask_embed': pred_mask_embed[-1],
            'pred_fq_embed': pred_fq_embed,
            'pred_cq_embed': pred_cq_embed[-1],
            'aux_outputs': self._set_aux_loss(
                pred_cls, pred_mask_embed, pred_cq_embed, pred_fq_embed
            )
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(
            self, outputs_cls, outputs_mask_embed, outputs_cq_embed, outputs_fq_embed
    ):
        return [{"pred_logits": a, "pred_mask_embed": b, "pred_cq_embed": c, "pred_fq_embed": outputs_fq_embed}
                for a, b, c in zip(outputs_cls[:-1], outputs_mask_embed[:-1], outputs_cq_embed[:-1])]

    def temporal_enhancement(self, frame_queries,):
        T, fQ, LB, C = frame_queries.shape
        pos = self.enc_pos(frame_queries)  # (T, fQ, LB, C)

        # (T,fQ,LB,C) -> (fQ*LB, C, T)
        frame_queries = frame_queries.permute(1, 2, 3, 0).contiguous()
        frame_queries = frame_queries.reshape(fQ * LB, C, T)
        frame_queries = self.temporal_norm(
            (self.short_temporal(frame_queries) + frame_queries).transpose(1, 2)).transpose(1, 2)

        # (fQ*LB, C, T) -> (T, fQ*LB, C)
        frame_queries = frame_queries.permute(2, 0, 1).contiguous()
        frame_queries = self.long_temporal(frame_queries,
                                           tgt_mask=None,
                                           tgt_key_padding_mask=None,
                                           query_pos=pos.reshape(T, fQ*LB, C))

        frame_queries = frame_queries.reshape(T, fQ, LB, C)
        return frame_queries
