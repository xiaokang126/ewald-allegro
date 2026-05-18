#!/usr/bin/env python3
"""
测试 Ewald-Allegro v2 模型能否正确前向传播。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 加载一个结构
from ase.io import read  # noqa: E402
data_dir = os.path.join(os.path.dirname(__file__), "data")
atoms = read(os.path.join(data_dir, "train.xyz"), index=0)

pos = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=device)
type_map = {1: 0, 8: 1}
atom_types = torch.tensor([type_map[z] for z in atoms.get_atomic_numbers()], dtype=torch.long, device=device)
energy = torch.tensor(atoms.get_potential_energy(), device=device)


print(f"原子: {len(atoms)} (12H₂O)")
print(f"晶胞:\n{cell.cpu().numpy()}")
print(f"能量: {energy.item():.4f} eV")

# 创建 v2 模型
from allegro.model.ewald_allegro_v2 import EwaldAllegroModelV2  # noqa: E402

model = EwaldAllegroModelV2(
    type_names=["H", "O"],
    r_max=5.0,
    num_bessels=8,
    l_max=1,
    num_layers=2,
    num_scalar_features=64,
    num_tensor_features=32,
    charge_hidden=64,
    readout_hidden=32,
    ewald_alpha=0.35,
    ewald_r_cut=8.0,
    ewald_grid=(32, 32, 32),

).to(device)

print(f"\n模型参数: {model.get_num_params():,}")
print(f"  其中 Allegro 短程: "
      f"{sum(p.numel() for p in model.allegro_short.parameters()):,}")
print(f"  其中 ChargePredictor: "
      f"{sum(p.numel() for p in model.charge_predictor.parameters()):,}")
print(f"  其中 Ewald: "
      f"{sum(p.numel() for p in model.ewald.parameters()):,}")

# 测试前向传播
print("\n=== 测试前向传播 ===")
try:
    data = {"pos": pos, "z": atom_types, "cell": cell}
    output = model(data)
    print(f"能量: {output['energy'].item():.6f} eV")
    print(f"短程: {output['energy_short'].item():.6f} eV")
    print(f"长程: {output['energy_long'].item():.6f} eV")
    print(f"shift: {output['energy_shift'].item():.6f} eV")
    print(f"电荷范围: [{output['charges'].min().item():.4f}, {output['charges'].max().item():.4f}]")
    print(f"每原子能量: min={output['per_atom_energy'].min().item():.6f}, "
          f"max={output['per_atom_energy'].max().item():.6f}")
    print("\n✅ 前向传播成功!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ 失败: {e}")
    sys.exit(1)

# 测试反向传播
print("\n=== 测试反向传播 ===")
try:
    target = torch.tensor(atoms.get_potential_energy(), device=device)

    loss = (output['energy'] - target) ** 2
    loss.backward()

    # 检查梯度
    total_grad = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().sum().item()
    print(f"梯度总和: {total_grad:.6f}")
    print("✅ 反向传播成功!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ 反向传播失败: {e}")

print("\n" + "=" * 50)
print("模型测试完成")
print("=" * 50)
