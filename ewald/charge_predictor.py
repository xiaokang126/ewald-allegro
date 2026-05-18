"""
电荷预测器模块

从 Allegro 的 scalar features 中预测每个原子的部分电荷。
使用一个简单的 MLP 将 scalar features 映射到原子电荷。
"""

import torch
import torch.nn as nn


class ChargePredictor(nn.Module):
    """
    从 Allegro 的 scalar features 预测原子电荷。

    参数:
        input_dim (int): 输入 scalar features 的维度
        hidden_dim (int): 隐藏层维度
        num_types (int): 原子类型数量
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_types: int = 2,
    ):
        super().__init__()

        self.num_types = num_types

        # 每个原子类型有独立的电荷预测 MLP
        # 这样 H 和 O 可以学习不同的电荷范围
        self.type_embed = nn.Embedding(num_types, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 初始化：初始电荷接近零
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        scalar_features: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            scalar_features: (N, input_dim) 每个原子的 scalar features
            atom_types: (N,) 原子类型索引

        返回:
            charges: (N,) 预测的原子电荷
        """
        type_emb = self.type_embed(atom_types)  # (N, hidden_dim)
        x = torch.cat([scalar_features, type_emb], dim=-1)  # (N, input_dim + hidden_dim)
        charges = self.net(x).squeeze(-1)  # (N,)
        return charges
