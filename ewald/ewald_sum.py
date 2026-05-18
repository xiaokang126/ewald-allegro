"""
Ewald 求和长程静电相互作用模块

实现了 O(N) 复杂度的 Ewald 求和：
- 实空间部分：短程 erfc 衰减，使用截断半径
- 倒空间部分：使用 FFT 在网格上计算（Particle Mesh Ewald）
- 自相互作用修正

对于周期性体系，网格大小固定，因此倒空间部分为 O(M log M) 其中 M 固定，
整体复杂度 O(N)。
"""

import torch
import math


class EwaldSummation(torch.nn.Module):
    """
    Ewald 求和计算长程静电相互作用。

    参数:
        alpha (float): Ewald 分裂参数，控制实空间和倒空间的收敛速度
        r_cut_real (float): 实空间截断半径
        grid_n (tuple): 倒空间 FFT 网格点数 (nx, ny, nz)
    """

    def __init__(
        self,
        alpha: float = 0.35,
        r_cut_real: float = 8.0,
        grid_n: tuple = (32, 32, 32),
    ):
        super().__init__()
        self.alpha = alpha
        self.r_cut_real = r_cut_real
        self.grid_n = grid_n

    def _compute_reciprocal(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算倒空间 Ewald 能量。

        使用 Particle Mesh 方法：
        1. 将原子电荷分配到网格上（B-spline 插值）
        2. FFT 到倒空间
        3. 乘以格林函数
        4. 逆 FFT 回实空间
        5. 在网格点上计算能量

        参数:
            charges: (N,) 原子电荷
            positions: (N, 3) 原子笛卡尔坐标
            cell: (3, 3) 晶胞向量

        返回:
            E_recip: 倒空间能量标量
        """
        nx, ny, nz = self.grid_n
        device = charges.device

        # 晶胞参数
        # cell 的列向量是晶胞基矢
        volume = torch.det(cell)
        # 倒空间基矢: 2π * (cell^{-1})^T
        cell_inv = torch.inverse(cell)
        recip_vecs = 2 * math.pi * cell_inv.T  # (3, 3)

        # 将原子坐标映射到 [0, 1) 分数坐标
        frac_pos = positions @ cell_inv  # (N, 3)
        frac_pos = frac_pos % 1.0

        # 构建网格坐标
        _grid_x = torch.arange(nx, device=device) / nx
        _grid_y = torch.arange(ny, device=device) / ny
        _grid_z = torch.arange(nz, device=device) / nz

        # 使用高斯涂抹将电荷分配到网格
        # 每个原子对周围网格点贡献电荷
        # 使用简单的最近网格点分配（NGP）或云在网格（CIC）
        # 这里使用 CIC 以提高精度

        # 网格间距
        dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz

        # 每个原子的分数坐标对应的网格索引
        ix = (frac_pos[:, 0] / dx).long()  # (N,)
        iy = (frac_pos[:, 1] / dy).long()
        iz = (frac_pos[:, 2] / dz).long()

        # 权重（CIC: 线性插值）
        wx = frac_pos[:, 0] / dx - ix.float()
        wy = frac_pos[:, 1] / dy - iy.float()
        wz = frac_pos[:, 2] / dz - iz.float()

        # 8 个相邻网格点的权重
        # 使用 scatter_add_ 将电荷分配到网格
        rho_grid = torch.zeros(nx, ny, nz, device=device, dtype=charges.dtype)

        # 遍历 8 个相邻点
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    # 权重
                    w = (
                        (1 - wx) if di == 0 else wx
                    ) * (
                        (1 - wy) if dj == 0 else wy
                    ) * (
                        (1 - wz) if dk == 0 else wz
                    )
                    # 索引（周期性边界）
                    idx = (ix + di) % nx
                    idy = (iy + dj) % ny
                    idz = (iz + dk) % nz
                    # 累加电荷
                    rho_grid.index_put_(
                        (idx, idy, idz),
                        rho_grid[idx, idy, idz] + charges * w,
                    )

        # FFT 到倒空间
        rho_k = torch.fft.fftn(rho_grid)  # (nx, ny, nz) complex

        # 构建倒空间网格
        kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * math.pi  # 分数坐标下的 k
        ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * math.pi
        kz = torch.fft.fftfreq(nz, d=dz, device=device) * 2 * math.pi

        # 转换为笛卡尔坐标下的 k 向量
        # k_cart = k_frac @ recip_vecs
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k_frac = torch.stack([KX, KY, KZ], dim=-1)  # (nx, ny, nz, 3)
        k_cart = torch.tensordot(k_frac, recip_vecs, dims=([-1], [0]))  # (nx, ny, nz, 3)

        # k 的模平方
        k_sq = (k_cart ** 2).sum(dim=-1)  # (nx, ny, nz)

        # 排除 k=0 项（均匀背景项）
        k_sq[0, 0, 0] = 1.0  # 避免除零

        # 格林函数: exp(-k^2 / (4*alpha^2)) / k^2
        # 乘以体积归一化因子
        green = torch.exp(-k_sq / (4 * self.alpha ** 2)) / k_sq
        green = green * (4 * math.pi / volume)

        # 恢复 k=0 项为 0
        green[0, 0, 0] = 0.0

        # 计算能量: E = 1/2 * Σ_k |rho_k|^2 * G(k)
        # 注意 FFT 的归一化
        rho_k_sq = (rho_k.conj() * rho_k).real  # (nx, ny, nz)
        # FFT 归一化: 需要除以网格点数
        E_recip = 0.5 * (rho_k_sq * green).sum() / (nx * ny * nz)

        return E_recip.real

    def _compute_real(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算实空间 Ewald 能量。

        对于截断半径内的原子对，计算 erfc(alpha * r) / r 项。

        参数:
            charges: (N,) 原子电荷
            positions: (N, 3) 原子笛卡尔坐标
            cell: (3, 3) 晶胞向量

        返回:
            E_real: 实空间能量标量
        """
        N = charges.shape[0]
        device = charges.device

        # 计算所有原子对的距离
        # 使用最小镜像约定
        cell_inv = torch.inverse(cell)

        # 扩展为 (N, N, 3) 的差向量
        pos_i = positions.unsqueeze(0)  # (1, N, 3)
        pos_j = positions.unsqueeze(1)  # (N, 1, 3)
        diff = pos_i - pos_j  # (N, N, 3)

        # 转换为分数坐标差
        frac_diff = diff @ cell_inv  # (N, N, 3)

        # 最小镜像：将分数坐标差映射到 [-0.5, 0.5)
        frac_diff = frac_diff - torch.round(frac_diff)

        # 转回笛卡尔坐标
        diff_min = frac_diff @ cell  # (N, N, 3)

        # 距离
        r = torch.sqrt((diff_min ** 2).sum(dim=-1) + 1e-12)  # (N, N)

        # 只考虑上三角（避免重复计算）且 r < r_cut
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
        mask = mask * (r < self.r_cut_real)

        # 计算 erfc(alpha * r) / r
        q_i = charges.unsqueeze(0)  # (1, N)
        q_j = charges.unsqueeze(1)  # (N, 1)
        qiqj = q_i * q_j  # (N, N)

        erfc_term = torch.erfc(self.alpha * r) / r
        E_real = 0.5 * (qiqj * erfc_term * mask).sum()

        return E_real

    def _compute_self(
        self,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算自相互作用修正。

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
        计算 Ewald 长程静电能量。

        参数:
            charges: (N,) 原子电荷
            positions: (N, 3) 原子笛卡尔坐标
            cell: (3, 3) 晶胞向量

        返回:
            E_ewald: 总 Ewald 能量标量
        """
        E_real = self._compute_real(charges, positions, cell)
        E_recip = self._compute_reciprocal(charges, positions, cell)
        E_self = self._compute_self(charges)

        return E_real + E_recip + E_self
