"""
Cell List 模块 — O(N) 近邻搜索

将模拟盒子划分为网格，每个网格单元边长 >= r_cut，
每个原子只与相邻 3^3 个单元中的原子相互作用。

对于 N 个原子，每个单元平均原子数为常数，
因此总计算量为 O(N)。
"""

import torch


class CellList:
    """
    Cell List 近邻搜索。

    参数:
        r_cut (float): 截断半径
        cell (torch.Tensor): (3, 3) 晶胞矩阵
        positions (torch.Tensor): (N, 3) 原子坐标
        use_pbc (bool): 是否使用周期性边界条件
    """

    def __init__(
        self,
        r_cut: float,
        cell: torch.Tensor,
        positions: torch.Tensor,
        use_pbc: bool = True,
    ):
        self.r_cut = r_cut
        self.cell = cell
        self.positions = positions
        self.use_pbc = use_pbc
        self.device = positions.device
        self.N = positions.shape[0]

        # 晶胞参数
        self.cell_inv = torch.inverse(cell)

        # 计算每个方向的 cell 数
        # 确保每个 cell 的边长 >= r_cut
        cell_norms = torch.norm(cell, dim=1)
        # n_cells = floor(cell_norms / r_cut)，至少为 1
        n_cells_float = torch.floor(cell_norms / r_cut)
        self.n_cells = torch.maximum(
            n_cells_float.long(),
            torch.tensor([1, 1, 1], device=self.device),
        )
        self.nx = self.n_cells[0].item()
        self.ny = self.n_cells[1].item()
        self.nz = self.n_cells[2].item()
        self.total_cells = self.nx * self.ny * self.nz

        # 构建 cell list
        self._build()

    def _build(self):
        """将原子分配到 cell 中"""
        frac = self.positions @ self.cell_inv
        frac = frac % 1.0

        # cell 索引
        cell_idx = (frac * self.n_cells).long()
        # 安全 clamp
        max_vals = (self.n_cells - 1).to(cell_idx.dtype)
        cell_idx = torch.where(cell_idx < 0, torch.zeros_like(cell_idx), cell_idx)
        cell_idx = torch.where(cell_idx > max_vals, max_vals, cell_idx)

        # 展平 cell 索引
        flat_idx = cell_idx[:, 0] * self.ny * self.nz + cell_idx[:, 1] * self.nz + cell_idx[:, 2]

        # 按 cell 排序
        self.sorted_idx = torch.argsort(flat_idx)
        self.sorted_flat = flat_idx[self.sorted_idx]

        # 每个 cell 的起始和结束位置
        self.cell_start = torch.full((self.total_cells,), self.N, device=self.device, dtype=torch.long)
        self.cell_end = torch.full((self.total_cells,), 0, device=self.device, dtype=torch.long)

        unique_cells, counts = torch.unique_consecutive(self.sorted_flat, return_counts=True)
        end_idx = torch.cumsum(counts, dim=0)
        start_idx = end_idx - counts

        self.cell_start[unique_cells] = start_idx
        self.cell_end[unique_cells] = end_idx

        # cell 的 3D 坐标
        self.cell_coords = torch.zeros((self.total_cells, 3), device=self.device, dtype=torch.long)
        for i in range(self.total_cells):
            iz = i % self.nz
            iy = (i // self.nz) % self.ny
            ix = i // (self.ny * self.nz)
            self.cell_coords[i] = torch.tensor([ix, iy, iz], device=self.device)

    def get_neighbor_pairs_batch(self) -> tuple:
        """
        批量获取近邻对（向量化版本）。

        由于 cell 边长 >= r_cut，只需检查相邻 3^3=27 个 cell。
        使用 ci <= nj_flat 避免重复对。

        返回:
            pairs_i: (E,) 第一个原子的索引
            pairs_j: (E,) 第二个原子的索引
            shifts: (E, 3) 晶胞偏移
            distances: (E,) 距离
        """
        # 当只有一个 cell 时，退化为朴素 O(N²) 方法
        if self.total_cells == 1:
            return self._get_all_pairs_naive()

        # 所有 27 个邻居偏移
        all_offsets = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]]

        pairs_i = []
        pairs_j = []
        shifts = []
        distances = []

        for ci in range(self.total_cells):
            si, ei = self.cell_start[ci].item(), self.cell_end[ci].item()
            if si >= ei:
                continue

            atoms_i = self.sorted_idx[si:ei]
            cix, ciy, ciz = self.cell_coords[ci].tolist()

            for dx, dy, dz in all_offsets:
                njx, njy, njz = cix + dx, ciy + dy, ciz + dz

                shift = [0, 0, 0]
                if self.use_pbc:
                    njx = self._apply_pbc_as(njx, self.nx, shift, 0)
                    njy = self._apply_pbc_as(njy, self.ny, shift, 1)
                    njz = self._apply_pbc_as(njz, self.nz, shift, 2)
                else:
                    if not (0 <= njx < self.nx and 0 <= njy < self.ny and 0 <= njz < self.nz):
                        continue

                nj_flat = njx * self.ny * self.nz + njy * self.nz + njz

                # 避免重复对：只处理 ci <= nj_flat
                if ci > nj_flat:
                    continue

                sj, ej = self.cell_start[nj_flat].item(), self.cell_end[nj_flat].item()
                if sj >= ej:
                    continue

                atoms_j = self.sorted_idx[sj:ej]

                # 批量计算所有对
                n_i = atoms_i.shape[0]
                n_j = atoms_j.shape[0]

                idx_i = atoms_i.repeat_interleave(n_j)
                idx_j = atoms_j.repeat(n_i)

                # 同一 cell 内去重
                if ci == nj_flat:
                    mask = idx_i < idx_j
                    idx_i = idx_i[mask]
                    idx_j = idx_j[mask]
                    if idx_i.shape[0] == 0:
                        continue

                # 计算距离
                pos_i = self.positions[idx_i]
                pos_j = self.positions[idx_j]

                shift_t = torch.tensor(shift, device=self.device, dtype=torch.float)
                cell_shift = shift_t @ self.cell

                diff = pos_i - pos_j + cell_shift.unsqueeze(0)
                dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)

                # 截断
                mask = dist <= self.r_cut
                idx_i = idx_i[mask]
                idx_j = idx_j[mask]
                dist = dist[mask]

                if idx_i.shape[0] > 0:
                    pairs_i.append(idx_i)
                    pairs_j.append(idx_j)
                    shifts.append(shift_t.unsqueeze(0).expand(idx_i.shape[0], 3))
                    distances.append(dist)

        if len(pairs_i) == 0:
            return (
                torch.zeros((0,), device=self.device, dtype=torch.long),
                torch.zeros((0,), device=self.device, dtype=torch.long),
                torch.zeros((0, 3), device=self.device, dtype=torch.long),
                torch.zeros((0,), device=self.device),
            )

        return (
            torch.cat(pairs_i),
            torch.cat(pairs_j),
            torch.cat(shifts),
            torch.cat(distances),
        )

    def _get_all_pairs_naive(self) -> tuple:
        """
        朴素 O(N²) 方法获取所有近邻对（用于 n_cells=1 的情况）。
        考虑 PBC 最小镜像约定。
        """
        N = self.N
        device = self.device

        pos_i = self.positions.unsqueeze(0)  # (1, N, 3)
        pos_j = self.positions.unsqueeze(1)  # (N, 1, 3)
        diff = pos_i - pos_j  # (N, N, 3)

        if self.use_pbc:
            frac_diff = diff @ self.cell_inv
            frac_diff = frac_diff - torch.round(frac_diff)
            diff_min = frac_diff @ self.cell
        else:
            diff_min = diff

        r = torch.sqrt((diff_min ** 2).sum(dim=-1) + 1e-12)

        # 上三角
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
        mask = mask * (r < self.r_cut)

        pairs = torch.nonzero(mask)
        if pairs.shape[0] == 0:
            return (
                torch.zeros((0,), device=device, dtype=torch.long),
                torch.zeros((0,), device=device, dtype=torch.long),
                torch.zeros((0, 3), device=device, dtype=torch.long),
                torch.zeros((0,), device=device),
            )

        pairs_i = pairs[:, 0]
        pairs_j = pairs[:, 1]
        dists = r[pairs[:, 0], pairs[:, 1]]

        # 计算 shifts
        if self.use_pbc:
            shifts_frac = frac_diff[pairs[:, 0], pairs[:, 1]] @ self.cell_inv
            shifts = -torch.round(shifts_frac).long()
        else:
            shifts = torch.zeros(pairs.shape[0], 3, device=device, dtype=torch.long)

        return pairs_i, pairs_j, shifts, dists

    def _apply_pbc_as(self, coord, dim_size, shift_list, idx):
        """应用 PBC 并记录偏移"""
        if coord < 0:
            coord += dim_size
            shift_list[idx] = -1
        elif coord >= dim_size:
            coord -= dim_size
            shift_list[idx] = 1
        return coord
