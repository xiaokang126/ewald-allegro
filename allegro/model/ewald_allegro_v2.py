"""
Ewald-Allegro v2: 真正的组合模型。

使用 `SequentialGraphNetwork` 构建 Allegro 图模块链，
然后用纯 PyTorch 做电荷预测 + Ewald 长程。

E_total = E_short(Allegro) + E_long(Ewald) + shift
"""
import torch
import torch.nn as nn
from e3nn import o3
from nequip.data import AtomicDataDict
from nequip.nn import (
    scatter,
    ScalarMLP, AtomwiseReduce,
)
from nequip.nn.embedding import (
    EdgeLengthNormalizer,
)

from allegro.nn import (
    TwoBodyBesselScalarEmbed,
    TwoBodySphericalHarmonicTensorEmbed,
    Allegro_Module,
    EdgewiseReduce,
)
from ewald.charge_predictor import ChargePredictor as ChargePredictor_
from ewald.ewald_sum_optimized import EwaldSummationOptimized


def neighbor_list_pbc(pos, cell, r_max):
    """ASE 近邻列表"""
    from ase import Atoms
    from ase.neighborlist import neighbor_list as nl
    atoms = Atoms(positions=pos.cpu().numpy(), cell=cell.cpu().numpy(), pbc=True)
    i_idx, j_idx, d, d_vec = nl("ijdD", atoms, r_max)

    device = pos.device
    ei = torch.tensor([i_idx.tolist(), j_idx.tolist()], dtype=torch.long, device=device)

    ev = torch.tensor(d_vec, dtype=torch.float32, device=device)
    el = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1).clamp(min=1e-6)
    return ei, ev, el


class AllegroShortRangeModel(nn.Module):
    """
    Allegro 短程势。
    手动链式调用 NequIP 图模块，配合纯 PyTorch 读出层。
    """
    def __init__(self, type_names, r_max=5.0, num_bessels=8, l_max=1,
                 num_layers=2, num_scalar_features=64, num_tensor_features=32,
                 readout_hidden=32):
        super().__init__()
        self.r_max = r_max
        self.num_scalar_features = num_scalar_features
        self.num_layers = num_layers

        irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(lmax=l_max, p=-1))

        # 使用 None 作为初始 irreps（让模块自行推断）
        init_irreps = None

        # 1. 边长度归一化
        self.edge_norm = EdgeLengthNormalizer(
            r_max=r_max, type_names=type_names,
            irreps_in=init_irreps,
        )

        # 2. 两体标量嵌入 — 继承上一个 irreps 链
        self.scalar_embed = TwoBodyBesselScalarEmbed(
            type_names=type_names,
            num_bessels=num_bessels,
            bessel_trainable=False,
            polynomial_cutoff_p=6,
            module_output_dim=num_scalar_features,
            irreps_in=self.edge_norm.irreps_out,
        )

        # 3. 标量 MLP
        self.scalar_mlp = ScalarMLP(
            output_dim=num_scalar_features,
            hidden_layers_depth=1,
            hidden_layers_width=num_scalar_features,
            nonlinearity="silu",
            bias=False,
            field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            irreps_in=self.scalar_embed.irreps_out,
        )

        # 4. 两体张量嵌入
        self.tensor_embed = TwoBodySphericalHarmonicTensorEmbed(
            irreps_edge_sh=irreps_edge_sh,
            num_tensor_features=num_tensor_features,
            irreps_in=self.scalar_mlp.irreps_out,
        )

        # 5. Allegro 核心
        self.allegro = Allegro_Module(
            num_layers=num_layers,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            tensor_track_allowed_irreps=irreps_edge_sh,
            avg_num_neighbors=20.0,  # 水的平均配位数估计值
            type_names=type_names,
            irreps_in=self.tensor_embed.irreps_out,
        )


        # 6. 边能量读出
        self.edge_readout = ScalarMLP(
            output_dim=1,
            hidden_layers_depth=1,
            hidden_layers_width=readout_hidden,
            nonlinearity="silu",
            bias=False,
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.EDGE_ENERGY_KEY,
            irreps_in=self.allegro.irreps_out,
        )

        # 7. 边→原子归约
        self.edge_eng_sum = EdgewiseReduce(
            field=AtomicDataDict.EDGE_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            avg_num_neighbors=20.0,
            type_names=type_names,
            irreps_in=self.edge_readout.irreps_out,
        )



        # 8. 原子→总能量
        self.total_energy_sum = AtomwiseReduce(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            irreps_in=self.edge_eng_sum.irreps_out,
        )

        # === 电荷特征 MLP ===
        edge_feat_dim = num_scalar_features * (num_layers + 1)
        self.charge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, num_scalar_features),
            nn.SiLU(),
            nn.Linear(num_scalar_features, num_scalar_features),
        )

    def build_data(self, pos, atom_types, cell):
        N = pos.shape[0]
        edge_index, edge_vec, edge_len = neighbor_list_pbc(pos, cell, self.r_max)

        data = {
            AtomicDataDict.POSITIONS_KEY: pos,
            AtomicDataDict.CELL_KEY: cell,
            AtomicDataDict.ATOM_TYPE_KEY: atom_types,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.EDGE_VECTORS_KEY: edge_vec,
            AtomicDataDict.EDGE_LENGTH_KEY: edge_len,
            AtomicDataDict.NUM_NODES_KEY: N,
        }

        return data

    def forward(self, pos, atom_types, cell):
        N = pos.shape[0]
        data = self.build_data(pos, atom_types, cell)

        data = self.edge_norm(data)
        data = self.scalar_embed(data)
        data = self.scalar_mlp(data)
        data = self.tensor_embed(data)
        data = self.allegro(data)

        # 保存边特征
        edge_features = data[AtomicDataDict.EDGE_FEATURES_KEY]

        data = self.edge_readout(data)
        data = self.edge_eng_sum(data)
        data = self.total_energy_sum(data)

        E_short = data[AtomicDataDict.TOTAL_ENERGY_KEY]

        # 电荷特征：边→原子聚合
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        node_feats = scatter(edge_features, edge_index[0], dim=0, dim_size=N, reduce="sum")
        charge_feats = self.charge_mlp(node_feats)

        per_atom_energy = data[AtomicDataDict.PER_ATOM_ENERGY_KEY]

        return E_short, charge_feats, per_atom_energy


class EwaldAllegroModelV2(nn.Module):
    """E_total = E_short(Allegro) + E_long(Ewald) + shift"""

    def __init__(
        self,
        type_names=["H", "O"], r_max=5.0, num_bessels=8, l_max=1,
        num_layers=2, num_scalar_features=64, num_tensor_features=32,
        charge_hidden=64, readout_hidden=32,
        ewald_alpha=0.35, ewald_r_cut=8.0, ewald_grid=(32, 32, 32),
        energy_shift_init=0.0,
    ):
        super().__init__()
        self.allegro_short = AllegroShortRangeModel(
            type_names=type_names, r_max=r_max, num_bessels=num_bessels,
            l_max=l_max, num_layers=num_layers,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            readout_hidden=readout_hidden,
        )
        self.num_types = len(type_names)
        self.charge_predictor = ChargePredictor_(
            input_dim=num_scalar_features,
            hidden_dim=charge_hidden,
            num_types=self.num_types,
        )
        self.ewald = EwaldSummationOptimized(
            alpha=ewald_alpha, r_cut_real=ewald_r_cut,
            grid_n=ewald_grid, use_cell_list=True,
        )
        self.energy_shift = nn.Parameter(torch.tensor([energy_shift_init]))

    def forward(self, data: dict) -> dict:
        pos, atom_types, cell = data["pos"], data["z"], data["cell"]
        E_short, charge_features, per_atom_energy = self.allegro_short(pos, atom_types, cell)
        charges = self.charge_predictor(charge_features, atom_types)
        E_long = self.ewald(charges, pos, cell)
        E_total = E_short + E_long + self.energy_shift.squeeze()

        return {
            "energy": E_total,
            "energy_short": E_short,
            "energy_long": E_long,
            "energy_shift": self.energy_shift.squeeze(),
            "charges": charges,
            "per_atom_energy": per_atom_energy,
        }

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
