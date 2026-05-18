#!/usr/bin/env python3
"""
训练 Ewald-Allegro v2 模型。
数据: allegro/data/train.xyz (水的 AIMD 轨迹)
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

from ase.io import read

class WaterDataset(Dataset):
    def __init__(self, xyz_path, split="train", train_ratio=0.8):
        atoms_list = read(xyz_path, index=":")
        n = len(atoms_list)
        n_train = int(n * train_ratio)
        self.atoms_list = atoms_list[:n_train] if split == "train" else atoms_list[n_train:]
        self.type_map = {1: 0, 8: 1}
        print(f"  {split}: {len(self.atoms_list)} 帧")

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        z = torch.tensor([self.type_map[z] for z in atoms.get_atomic_numbers()], dtype=torch.long)
        # ASE extxyz 把能量存在 SinglePointCalculator 中
        energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        return {"pos": pos, "z": z, "cell": cell, "energy": energy}

def collate_fn(batch):
    return batch[0]

data_dir = os.path.join(os.path.dirname(__file__), "data")
train_set = WaterDataset(os.path.join(data_dir, "train.xyz"), "train")
val_set = WaterDataset(os.path.join(data_dir, "train.xyz"), "val")
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn)

from allegro.model.ewald_allegro_v2 import EwaldAllegroModelV2

model = EwaldAllegroModelV2(
    type_names=["H", "O"], r_max=5.0, num_bessels=8,
    l_max=1, num_layers=2, num_scalar_features=64, num_tensor_features=32,
    charge_hidden=64, readout_hidden=32,
    ewald_alpha=0.35, ewald_r_cut=8.0, ewald_grid=(32, 32, 32),
).to(device)
print(f"\n模型参数: {model.get_num_params():,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

energy_list = [d["energy"].item() for d in [train_set[i] for i in range(len(train_set))]]
energy_mean = np.mean(energy_list)
energy_std = np.maximum(np.std(energy_list), 1.0)
print(f"能量: mean={energy_mean:.3f}, std={energy_std:.3f}")

best_val_loss = float("inf")
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        target_energy = batch["energy"]
        output = model(batch)
        pred_energy = output["energy"]
        loss_energy = ((pred_energy - target_energy) / energy_std) ** 2
        charges = output["charges"]
        loss_charge = 0.01 * charges.pow(2).mean()
        loss = loss_energy + loss_charge

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(batch)
            val_loss += ((output["energy"] - batch["energy"]) / energy_std).abs().item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(data_dir, "model_best.pt"))

        print(f"  ep{epoch:3d} train={total_loss/len(train_loader):.4f} val={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f} ⭐")

    if epoch % 10 == 0:
        print(f"  ep{epoch:3d} train={total_loss/len(train_loader):.4f} val={val_loss:.4f}")

    if optimizer.param_groups[0]['lr'] < 1e-6:
        break

print(f"\n完成！最佳val: {best_val_loss:.4f}")
