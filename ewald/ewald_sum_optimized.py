"""
优化版 Ewald 求和 — O(N) 复杂度

使用 Cell List 实现 O(N) 实空间计算，
使用 Particle Mesh Ewald (PME) 实现 O(N) 倒空间计算。

并行化设计：
- 实空间：原子级并行，每个原子的 neighbor 列表独立
- 倒空间：FFT 使用 cuFFT (GPU) 自动并行
- 电荷分配：原子级并行
"""

import torch
import math
from .cell_list import CellList


class EwaldSummationOptimized(torch.nn.Module):
    """
    优化版 Ewald 求和 — O(N) 复杂度。

    参数:
        alpha (float): Ewald 分裂参数
        r_cut_real (float): 实空间截断半径
        grid_n (tuple): 倒空间 FFT 网格点数 (nx, ny, nz)
        use_cell_list (bool): 是否使用 Cell List 加速实空间
    """

    def __init__(
        self,
        alpha: float = 0.35,
        r_cut_real: float = 8.0,
        grid_n: tuple = (32, 32, 32),
        use_cell_list: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.r_cut_real = r_cut_real
        self.grid_n = grid_n
        self.use_cell_list = use_cell_list

    def _compute_real_cell_list(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用 Cell List 计算实空间 Ewald 能量 — O(N)。

        每个原子只与相邻 cell 中的原子相互作用，
        每个 cell 中原子数为常数，总计算量 O(N)。
        """
        # 构建 Cell List
        cl = CellList(self.r_cut_real, cell, positions, use_pbc=True)

        # 获取近邻对
        pairs_i, pairs_j, shifts, distances = cl.get_neighbor_pairs_batch()

        if pairs_i.shape[0] == 0:
            return torch.tensor(0.0, device=charges.device)

        # 计算 erfc(alpha * r) / r
        q_i = charges[pairs_i]
        q_j = charges[pairs_j]
        qiqj = q_i * q_j

        erfc_term = torch.erfc(self.alpha * distances) / distances
        E_real = 0.5 * (qiqj * erfc_term).sum()

        return E_real

    def _compute_real_naive(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        朴素 O(N²) 实空间计算（用于对比验证）。
        """
        N = charges.shape[0]
        device = charges.device

        cell_inv = torch.inverse(cell)

        pos_i = positions.unsqueeze(0)  # (1, N, 3)
        pos_j = positions.unsqueeze(1)  # (N, 1, 3)
        diff = pos_i - pos_j  # (N, N, 3)

        frac_diff = diff @ cell_inv
        frac_diff = frac_diff - torch.round(frac_diff)
        diff_min = frac_diff @ cell

        r = torch.sqrt((diff_min ** 2).sum(dim=-1) + 1e-12)

        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
        mask = mask * (r < self.r_cut_real)

        q_i = charges.unsqueeze(0)
        q_j = charges.unsqueeze(1)
        qiqj = q_i * q_j

        erfc_term = torch.erfc(self.alpha * r) / r
        E_real = 0.5 * (qiqj * erfc_term * mask).sum()

        return E_real

    def _compute_reciprocal(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算倒空间 Ewald 能量 — O(N)。

        使用 Particle Mesh 方法：
        1. 将原子电荷分配到网格上（B-spline 插值）— O(N)
        2. FFT 到倒空间 — O(M log M)，M 固定为常数
        3. 乘以格林函数 — O(M)
        4. 逆 FFT 回实空间 — O(M log M)
        5. 在网格点上计算能量 — O(M)
        """
        N = charges.shape[0]
        nx, ny, nz = self.grid_n
        device = charges.device

        # 晶胞参数
        volume = torch.det(cell)
        cell_inv = torch.inverse(cell)
        recip_vecs = 2 * math.pi * cell_inv.T  # (3, 3)

        # 分数坐标
        frac_pos = positions @ cell_inv  # (N, 3)
        frac_pos = frac_pos % 1.0

        # 网格间距
        dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz

        # 每个原子的网格索引
        ix = (frac_pos[:, 0] / dx).long()
        iy = (frac_pos[:, 1] / dy).long()
        iz = (frac_pos[:, 2] / dz).long()

        # CIC 权重
        wx = frac_pos[:, 0] / dx - ix.float()
        wy = frac_pos[:, 1] / dy - iy.float()
        wz = frac_pos[:, 2] / dz - iz.float()

        # 电荷分配到网格 — O(N)
        rho_grid = torch.zeros(nx, ny, nz, device=device, dtype=charges.dtype)

        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    w = (
                        (1 - wx) if di == 0 else wx
                    ) * (
                        (1 - wy) if dj == 0 else wy
                    ) * (
                        (1 - wz) if dk == 0 else wz
                    )
                    idx = (ix + di) % nx
                    idy = (iy + dj) % ny
                    idz = (iz + dk) % nz
                    rho_grid.index_put_(
                        (idx, idy, idz),
                        rho_grid[idx, idy, idz] + charges * w,
                    )

        # FFT — O(M log M)
        rho_k = torch.fft.fftn(rho_grid)

        # 倒空间网格
        kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * math.pi
        ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * math.pi
        kz = torch.fft.fftfreq(nz, d=dz, device=device) * 2 * math.pi

        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k_frac = torch.stack([KX, KY, KZ], dim=-1)
        k_cart = torch.tensordot(k_frac, recip_vecs, dims=([-1], [0]))

        k_sq = (k_cart ** 2).sum(dim=-1)
        k_sq[0, 0, 0] = 1.0  # 避免除零

        # 格林函数
        green = torch.exp(-k_sq / (4 * self.alpha ** 2)) / k_sq
        green = green * (4 * math.pi / volume)
        green[0, 0, 0] = 0.0

        # 能量
        rho_k_sq = (rho_k.conj() * rho_k).real
        E_recip = 0.5 * (rho_k_sq * green).sum() / (nx * ny * nz)

        return E_recip.real

    def _compute_self(
        self,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        自相互作用修正 — O(N)。
        E_self = -alpha / sqrt(pi) * Σ_i q_i^2
        """
        return -self.alpha / math.sqrt(math.pi) * (charges ** 2).sum()

    def forward(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Ewald 长程静电能量 — O(N)。

        参数:
            charges: (N,) 原子电荷
            positions: (N, 3) 原子笛卡尔坐标
            cell: (3, 3) 晶胞向量

        返回:
            E_ewald: 总 Ewald 能量标量
        """
        if self.use_cell_list:
            E_real = self._compute_real_cell_list(charges, positions, cell)
        else:
            E_real = self._compute_real_naive(charges, positions, cell)

        E_recip = self._compute_reciprocal(charges, positions, cell)
        E_self = self._compute_self(charges)

        return E_real + E_recip + E_self

    def forward_with_forces(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple:
        """
        计算 Ewald 能量和力。

        返回:
            E_ewald: 能量标量
            forces: (N, 3) 每个原子上的力
        """
        # 使用 autograd 计算力
        positions.requires_grad_(True)
        charges.requires_grad_(True)

        E = self.forward(charges, positions, cell)

        # 对位置求导得到力
        forces = -torch.autograd.grad(E, positions, create_graph=True)[0]

        return E, forces
