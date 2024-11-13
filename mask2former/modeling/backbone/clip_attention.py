from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingSineText


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ClipStageAdapterV1(nn.Module):
    def __init__(self, visual_dim, language_dim, hidden_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, visual_dim, kernel_size=1)

        self.act = _get_activation_fn('relu')

        self.cross_attention = CrossAttentionLayer(d_model=hidden_dim, nhead=8, dropout=0.1)
        self.linear_l = nn.Linear(language_dim, hidden_dim)
        self.linear_v = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)
        self.up_v = nn.Linear(hidden_dim, visual_dim)

        self.visual_pos = PositionEmbeddingSine(hidden_dim // 2)
        self.lang_pos = PositionEmbeddingSineText(hidden_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.MoE = MLP(visual_dim * 3, hidden_dim, 3, num_layers=2)

    def forward(self, v, s, l, l_mask):
        """
        Args:
            v: visual feature,    (B, Cv, H, W)
            s: sentence feat,     (B, 1, Cl)
            l: language feat,     (B, Nl, Cl)
            l_mask: language mask,(B, Nl)
        Returns:

        """
        B, Cv, H, W = v.shape
        v_res = v
        v_path = self.act(self.conv1(v))
        v_path = self.act(self.conv2(v_path))

        v = self.linear_v(v)
        v_pos = self.visual_pos(v).flatten(2)  # (B, C, HW)
        v_pos = v_pos.permute(2, 0, 1)  # (HW, B, C)
        v = v.flatten(2).permute(2, 0, 1)

        l = self.linear_l(l)  # (B, N, C)
        l_pos = self.lang_pos(l)
        l = l.permute(1, 0, 2)
        l_pos = l_pos.permute(1, 0, 2)

        vl = self.cross_attention(
            v, l,
            memory_mask=None,
            memory_key_padding_mask=~l_mask,
            pos=l_pos, query_pos=v_pos
        )
        vl = self.up_v(vl)
        vl = vl.permute(1, 2, 0).reshape(B, Cv, H, W)

        fuse = torch.cat([self.pool(v_res)[:, :, 0, 0],
                          self.pool(vl)[:, :, 0, 0],
                          self.pool(v_path)[:, :, 0, 0]], dim=-1)

        gate = self.MoE(fuse)
        gate = torch.softmax(gate, dim=-1)  # 归一化权重

        v = gate[:, 0, None, None, None] * v_res + \
            gate[:, 1, None, None, None] * vl + \
            gate[:, 2, None, None, None] * v_path

        return v


class ClipStageAdapterV2(nn.Module):
    def __init__(self, visual_dim, language_dim, hidden_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, visual_dim, kernel_size=1)

        self.act = _get_activation_fn('relu')

        self.cross_attention = CrossAttentionLayer(d_model=hidden_dim, nhead=8, dropout=0.1)
        self.linear_l = nn.Linear(language_dim, hidden_dim)
        self.linear_v = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)
        self.up_v = nn.Linear(hidden_dim, visual_dim)

        self.visual_pos = PositionEmbeddingSine(hidden_dim // 2)
        self.lang_pos = PositionEmbeddingSineText(hidden_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.MoE = MLP(visual_dim * 3, hidden_dim, 3, num_layers=2)

    def forward(self, v, s, l, l_mask):
        """
        Args:
            v: visual feature,    (B, Cv, H, W)
            s: sentence feat,     (B, 1, Cl)
            l: language feat,     (B, Nl, Cl)
            l_mask: language mask,(B, Nl)
        Returns:

        """
        B, Cv, H, W = v.shape

        v_path = self.act(self.conv1(v))
        v_path = self.act(self.conv2(v_path)) + v

        # ratio = 0.6
        # new_h, new_w = H * ratio, W * ratio
        # v = F.interpolate(v, size=(new_h, new_w), mode="bilinear", align_corners=False)

        v = self.linear_v(v)
        v_pos = self.visual_pos(v).flatten(2)  # (B, C, HW)
        v_pos = v_pos.permute(2, 0, 1)  # (HW, B, C)
        v = v.flatten(2).permute(2, 0, 1)

        l = self.linear_l(l)  # (B, N, C)
        l_pos = self.lang_pos(l)
        l = l.permute(1, 0, 2)
        l_pos = l_pos.permute(1, 0, 2)

        vl = self.cross_attention(
            v, l,
            memory_mask=None,
            memory_key_padding_mask=~l_mask,
            pos=l_pos, query_pos=v_pos
        )
        vl = self.up_v(vl)
        vl = vl.permute(1, 2, 0).reshape(B, Cv, H, W)

        s = self.linear_l(s)  # (B, 1, C)
        s = s.permute(1, 0, 2)  # (1, B, C)
        vs = self.cross_attention(
            v, s,
            memory_mask=None,
            memory_key_padding_mask=None,
            pos=None, query_pos=v_pos
        )
        vs = self.up_v(vs)
        vs = vs.permute(1, 2, 0).reshape(B, Cv, H, W)

        # vl = F.interpolate(vl, size=(H, W), mode="bilinear", align_corners=False)
        # vs = F.interpolate(vs, size=(H, W), mode="bilinear", align_corners=False)

        fuse = torch.cat([self.pool(vs)[:, :, 0, 0],
                          self.pool(vl)[:, :, 0, 0],
                          self.pool(v_path)[:, :, 0, 0]], dim=-1)

        gate = self.MoE(fuse)
        gate = torch.softmax(gate, dim=-1)  # 归一化权重

        v = gate[:, 0, None, None, None] * vs + \
            gate[:, 1, None, None, None] * vl + \
            gate[:, 2, None, None, None] * v_path

        return v