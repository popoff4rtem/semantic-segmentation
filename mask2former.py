import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# убрал scipy import — он тут не нужен

class PixelDecoder(nn.Module):
    def __init__(self, hidden_dim, in_channels_list=[2048, 1024, 512, 256]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_dim, 1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1) for _ in range(len(in_channels_list)-1)
        ])

    def forward(self, features):
        levels = ['layer4', 'layer3', 'layer2', 'layer1']
        laterals = [self.lateral_convs[i](features[level]) for i, level in enumerate(levels)]
        
        current = laterals[0]
        for i in range(1, len(laterals)):
            current = F.interpolate(current, scale_factor=2, mode='bilinear', align_corners=False)
            current = current + laterals[i]
            current = self.output_convs[i-1](current)
            current = F.relu(current)
        
        pixel_features = F.interpolate(current, scale_factor=4, mode='bilinear', align_corners=False)
        return pixel_features  # [B, hidden_dim, H, W]


class SimplifiedMask2Former(nn.Module):
    def __init__(self, num_classes, backbone='resnet50',
                 hidden_dim=128, num_queries=75, num_heads=4, num_layers=4, class_weights = None):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.class_weights = class_weights

        # backbone + pixel decoder
        self.backbone = self._create_backbone(backbone)
        self.pixel_decoder = PixelDecoder(hidden_dim=hidden_dim)

        # transformer
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = TransformerDecoder(hidden_dim, num_heads, num_layers)

        # mask_embed должен возвращать embedding того же размера, что и pixel_features.channels
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # линейная проекция из Q -> num_classes (применяется для каждого пикселя)
        self.query2class = nn.Linear(num_queries, num_classes)

    def _create_backbone(self, backbone_name):
        # Я использую torchvision (можно заменить на torch.hub если нужно)
        if backbone_name == 'resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=True)
            return nn.ModuleDict({
                'conv1': nn.Sequential(*list(backbone.children())[:4]),
                'layer1': backbone.layer1,
                'layer2': backbone.layer2,
                'layer3': backbone.layer3,
                'layer4': backbone.layer4
            })
        else:
            simple_cnn = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU()
            )
            return nn.ModuleDict({
                'conv1': simple_cnn[:3],
                'layer1': simple_cnn[3:5],
                'layer2': simple_cnn[5:8],
                'layer3': simple_cnn[8:10],
                'layer4': simple_cnn[10:]
            })

    def forward(self, x, masks=None):
        # backbone
        features = {}
        out = x
        for name, module in self.backbone.items():
            out = module(out)
            features[name] = out

        # pixel features [B, hidden_dim, H, W]
        pixel_features = self.pixel_decoder(features)

        # transformer queries
        B = x.size(0)
        query_emb = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, hidden_dim]
        decoder_output = self.transformer(query_emb, pixel_features)     # [B, Q, hidden_dim]

        # mask embeddings per-query (same dim as pixel_features channels)
        query_feats = self.mask_embed(decoder_output)  # [B, Q, hidden_dim]

        # per-query masks: bqc, bchw -> bqhw (here c==hidden_dim)
        per_query_masks = torch.einsum('bqc,bchw->bqhw', query_feats, pixel_features)  # [B, Q, H, W]

        # агрегация Q -> num_classes:
        # поменяем форму так, чтобы применить Linear(Q -> num_classes) к последнему измерению
        # per_query_masks: [B, Q, H, W] -> [B, H, W, Q]
        x_q = per_query_masks.permute(0, 2, 3, 1)  # [B, H, W, Q]
        # применяем линейную проекцию на вектор длины Q -> num_classes
        logits_hw_c = self.query2class(x_q)  # [B, H, W, num_classes]
        # вернём в формат [B, num_classes, H, W]
        mask_logits = logits_hw_c.permute(0, 3, 1, 2)  # [B, num_classes, H, W]

        if masks is None:
            return torch.softmax(mask_logits, dim=1)
        else:
            if self.class_weights is None:
                return F.cross_entropy(mask_logits, masks.long(), ignore_index=255)
            else:
                return F.cross_entropy(
                    mask_logits, 
                    masks.long(), 
                    weight=self.class_weights.to(mask_logits.device), 
                    ignore_index=255
                )




class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        # memory: [B, C, H, W] -> flatten to [B, H*W, C]
        batch_size, C, H, W = memory.shape
        memory = memory.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        for layer in self.layers:
            queries = layer(queries, memory)
        return self.norm(queries)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        attn_output, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.dropout(attn_output))
        attn_output, _ = self.multihead_attn(queries, memory, memory)
        queries = self.norm2(queries + self.dropout(attn_output))
        ff_output = self.linear2(F.relu(self.linear1(queries)))
        queries = self.norm3(queries + self.dropout(ff_output))
        return queries

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim,
                                    hidden_dim if i < num_layers - 1 else output_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, Q, input_dim] -> flatten last dim apply sequential on last dim:
        B, Q, D = x.shape
        x_flat = x.view(B * Q, D)
        out = self.layers(x_flat)
        return out.view(B, Q, -1)
