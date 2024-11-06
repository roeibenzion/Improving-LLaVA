import torch
import torch.nn as nn
import re


class PresiverResampler(nn.Module):
    def __init__(self, hidden_size, num_queries=64, num_layers=2):
        super(PresiverResampler, self).__init__()
        self.num_layers = num_layers
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size))  # Learned queries
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x_f):
        x = self.queries.expand(x_f.size(0), -1, -1)  # [Batch, R, d]
        for _ in range(self.num_layers):
            attn_output, _ = self.attn(x, torch.cat([x_f, x], dim=1), torch.cat([x_f, x], dim=1))
            x = x + attn_output
            x = x + self.ff(x)
        return x

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # NOTE: here you chnage the size of the input for the projection. What i did should fix layer 0.
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        #modules = [nn.Linear(1024, 4096)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            #modules.append(nn.Linear(4096, 4096))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    if projector_type == 'presiver':
        return PresiverResampler(config.hidden_size, num_layers=kwargs.get('num_layers', 2))

    raise ValueError(f'Unknown projector type: {projector_type}')
