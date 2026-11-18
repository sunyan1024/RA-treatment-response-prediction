"""InceptionV3 三模态融合，关节类型嵌入和可学习位置编码；训练、推理共用。"""
import torch
import torch.nn as nn
from torchvision import models

MODEL_ID = "inceptionv3_joint_position_multimodal_v1"
JOINT_TYPES = ("Knee", "Knuckle", "Wrist", "Ankle", "Toe", "Elbow")
JOINT_ALIASES = dict(zip(("膝关节", "指关节", "腕关节", "踝关节", "趾关节", "肘关节"), JOINT_TYPES))
MAX_INSTANCES = 50

def joint_index(value):
    if not isinstance(value, str):
        raise ValueError("每张影像必须提供 joint 关节类型")
    name = JOINT_ALIASES.get(value.strip(), value.strip())
    if name not in JOINT_TYPES:
        raise ValueError(f"不支持的关节类型 {value!r}，允许：{JOINT_TYPES} 或对应中文名称")
    return JOINT_TYPES.index(name)



class RadiomicsAttentionPooling(nn.Module):
    """Radiomics Attention Pooling (支持 Mask)

    输入:  x -> [B, N, D_radio], mask -> [B, N] (True = padding/忽略)
    输出:  [B, D_radio]
    """
    def __init__(self, radio_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(radio_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        attn_score = self.attn(x)  # [B, N, 1]
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            attn_score = attn_score.masked_fill(mask_expanded, -1e9)
        attn_weight = torch.softmax(attn_score, dim=1)  # [B, N, 1]
        out = torch.sum(attn_weight * x, dim=1)  # [B, D]
        return out


class TransformerMIL(nn.Module):
    """Transformer MIL (支持 Mask)，用 CLS token 聚合变长实例。"""
    def __init__(self, in_dim, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.joint_embedding = nn.Embedding(len(JOINT_TYPES), embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, MAX_INSTANCES + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, mask, joint_indices):
        B, N, _ = x.shape
        if not 1 <= N <= MAX_INSTANCES:
            raise ValueError(f"每位患者影像数量必须为 1..{MAX_INSTANCES}")
        if joint_indices.shape != (B, N) or joint_indices.dtype != torch.long:
            raise ValueError("joint_indices 必须是 [B,N] 的 torch.long")
        if torch.any((joint_indices < 0) | (joint_indices >= len(JOINT_TYPES))):
            raise ValueError("关节类型索引越界")
        if mask is not None and (mask.shape != (B, N) or mask.dtype != torch.bool or mask.all(dim=1).any()):
            raise ValueError("mask 必须为 [B,N] 布尔张量，每位患者至少一张有效影像")
        x = self.proj(x) + self.joint_embedding(joint_indices)
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1) + self.pos_embedding[:, :N+1]

        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # [B, N+1]

        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        patient_feat = x[:, 0]  # CLS token 输出
        return patient_feat


class HierarchicalCrossModalAttention(nn.Module):
    """两层 Cross-Attention：radio←img，然后 out1←clinical。"""
    def __init__(self, dim_img, dim_radio, dim_clinical, embed_dim=256, num_heads=4):
        super().__init__()
        self.q_radio_proj = nn.Linear(dim_radio, embed_dim)
        self.kv_clinical_proj = nn.Linear(dim_clinical, embed_dim)
        self.q_out1_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_img_proj = nn.Linear(dim_img, embed_dim)

        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x_img, x_radio, x_clinical):
        q1 = self.q_radio_proj(x_radio).unsqueeze(1)
        k1 = self.kv_img_proj(x_img).unsqueeze(1)
        v1 = k1.clone()
        out1, _ = self.cross_attn1(q1, k1, v1)
        out1 = self.norm1(out1 + q1)

        q2 = self.q_out1_proj(out1).clone()
        k2 = self.kv_clinical_proj(x_clinical).unsqueeze(1)
        v2 = k2.clone()
        out2, _ = self.cross_attn2(q2, k2, v2)
        out2 = self.norm2(out2 + q2)

        return out2.squeeze(1)  # [B, D]


class TransformerMILFusionModel(nn.Module):
    def __init__(self, backbone, radio_dim, clinical_dim, embed_dim=256, num_classes=2):
        super().__init__()

        feat_dim = backbone.fc.in_features  # InceptionNetv3: 2048
        backbone.fc = nn.Identity()
        backbone.AuxLogits = None
        self.backbone = backbone

        self.mil = TransformerMIL(in_dim=feat_dim, embed_dim=embed_dim)
        self.radio_pool = RadiomicsAttentionPooling(radio_dim, hidden_dim=64)
        self.radio_proj = nn.Linear(radio_dim, embed_dim)
        self.clinical_proj = nn.Linear(clinical_dim, embed_dim)

        self.attn_fusion = HierarchicalCrossModalAttention(
            dim_img=embed_dim, dim_radio=embed_dim, dim_clinical=embed_dim,
            embed_dim=embed_dim, num_heads=4
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, radios, clinical, mask, joint_indices):
        # images: [B, N, C, H, W]; radios: [B, N, D_rad]; clinical: [B, D_clin]; mask: [B, N]
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        feats = self.backbone(x)  # [B*N, feat_dim]
        if hasattr(feats, "logits"):
            feats = feats.logits
        feats = feats.view(B, N, -1)

        img_feat = self.mil(feats, mask, joint_indices)
        radio_pooled = self.radio_pool(radios, mask)
        radio_feat = self.radio_proj(radio_pooled)
        clinical_feat = self.clinical_proj(clinical)

        fused = self.attn_fusion(img_feat, radio_feat, clinical_feat)
        out = self.classifier(fused)
        return out


def build_model(radio_dim, clinical_dim, device="cpu"):
    """构建 InceptionNetv3 版 MIL 融合模型（结构与训练时一致）。"""
    # 必须 transform_input=True：训练用 weights=DEFAULT 会自动开它，但该开关是
    # __init__ 里的普通属性、不进 state_dict，load_state_dict 不会恢复。
    # 用 weights=None 时默认 False，会导致推理输入分布与训练不一致（前向仿射被跳过）。
    backbone = models.inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
    model = TransformerMILFusionModel(
        backbone, radio_dim=radio_dim, clinical_dim=clinical_dim, num_classes=2
    )
    return model.to(device)
