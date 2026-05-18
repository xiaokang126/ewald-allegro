#!/usr/bin/env python3
"""
统一数据预处理与依赖检查脚本。

功能模式（可通过命令行子命令切换）：
  1. default / --convert          : 从 VASP/step_* → extxyz 训练/测试集（默认）
  2. --extract-steps              : 从 VASP 原始输出 → step_* 目录（旧格式中间步骤）
  3. --check-deps                 : 检查所有 Python 依赖是否安装

数据转换模式功能：
  - 自动扫描原始 VASP MD 输出目录（含 OUTCAR+XDATCAR），提取结构+总能
  - 也支持从已提取的 step_*/ 目录格式读取
  - 自动检测元素组成、晶胞参数、体系大小
  - 异常数据检测与删除（能量离群、结构异常）
  - 输出 extxyz 格式训练/测试集到 allegro/data/

用法：
  # 数据转换
  python prepare_data.py                          # 自动扫描 ../unloaded_data/
  python prepare_data.py --vasp-dir ../unloaded_data/hot3
  python prepare_data.py --steps-dir ../vasp_steps_fixed
  python prepare_data.py --steps-dir ../vasp_steps_fixed --z-score 4 --min-dist 0.5

  # 提取为 step_* 中间目录
  python prepare_data.py --extract-steps --xdatcar MD/XDATCAR --outcar MD/OUTCAR
  python prepare_data.py --extract-steps \
      --xdatcar MD/XDATCAR --outcar MD/OUTCAR \
      --xdatcar2 MD2/XDATCAR --outcar2 MD2/OUTCAR   # 续算合并

  # 依赖检查
  python prepare_data.py --check-deps
"""

import os
import sys
import re
import argparse
import random
import shutil
import logging
import numpy as np
from collections import Counter

# ── logging ──
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("prepare_data")


# ═══════════════════════════════════════════════════════════════
#  Part Z: 依赖检查 (来自 check_deps.py)
# ═══════════════════════════════════════════════════════════════

def cmd_check_deps():
    """
    --check-deps 模式：检查所有 Python 依赖是否安装。
    """
    deps = [
        ("torch",            "pytorch"),
        ("e3nn",             "e3nn"),
        ("nequip",           "nequip"),
        ("allegro",          "allegro"),
        ("pytorch_lightning","pytorch_lightning"),
        ("hydra",            "hydra"),
        ("ase",              "ase"),
        ("scipy",            "scipy"),
        ("numpy",            "numpy"),
    ]

    print("=" * 60)
    print("依赖库检查报告")
    print("=" * 60)

    all_ok = True
    for mod_name, display_name in deps:
        try:
            exec(f"import {mod_name}")
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", getattr(mod, "__version_info__", "installed"))
            print(f"  ✅ OK  {display_name:25s}  v{ver}")
        except Exception as e:
            all_ok = False
            print(f"  ❌ FAIL {display_name:25s}  -> {e}")

    print()
    try:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version:   {torch.version.cuda}")
            print(f"  GPU device:     {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    except Exception:
        print("  CUDA: torch not available")

    print()
    if all_ok:
        print("🎉 所有依赖安装成功！")
    else:
        print("⚠️  部分依赖未安装，请先安装缺失的库。")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  Part Y: VASP → step_* 提取 (来自 extract_vasp_steps.py)
# ═══════════════════════════════════════════════════════════════

def _read_xdatcar_info_legacy(xdatcar_path):
    """读取 XDATCAR 头部信息（legacy，用于 step_* 提取）"""
    with open(xdatcar_path) as f:
        lines = f.readlines()
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[i + 2].strip().split()]
    elements_line = lines[5].strip().split()
    counts_line = lines[6].strip().split()
    element_counts = {}
    for elem, cnt in zip(elements_line, counts_line):
        element_counts[elem] = int(cnt)
    num_atoms = sum(element_counts.values())
    elements = []
    for elem, cnt in zip(elements_line, counts_line):
        elements.extend([elem] * int(cnt))
    num_steps = sum(1 for line in lines if "Direct configuration=" in line)
    return num_atoms, elements, element_counts, cell, num_steps


def _read_xdatcar_coords_legacy(xdatcar_path, num_atoms, num_steps):
    """读取 XDATCAR 坐标（legacy）"""
    with open(xdatcar_path) as f:
        lines = f.readlines()
    config_idxs = [i for i, line in enumerate(lines) if "Direct configuration=" in line]
    coords = np.zeros((num_steps, num_atoms, 3))
    for step_idx, start in enumerate(config_idxs):
        for atom_idx in range(num_atoms):
            parts = lines[start + 1 + atom_idx].strip().split()
            coords[step_idx, atom_idx] = [float(parts[0]), float(parts[1]), float(parts[2])]
    return coords


def _read_outcar_energies_legacy(outcar_path, num_steps):
    """从 OUTCAR 提取 MD 步总能（legacy）"""
    with open(outcar_path) as f:
        content = f.read()
    pattern = r"energy  without entropy\s*=\s*([-.\d]+)"
    energies = [float(x) for x in re.findall(pattern, content)]
    if len(energies) >= num_steps:
        return np.array(energies[:num_steps])
    # fallback: 单空格
    pattern2 = r"energy without entropy\s*=\s*([-.\d]+)"
    energies = [float(x) for x in re.findall(pattern2, content)]
    if len(energies) >= num_steps:
        return np.array(energies[-num_steps:])
    log.error(f"能量不足: {len(energies)} < {num_steps}")
    return None


def _extract_to_steps(xdatcar_path, outcar_path, output_dir, wrap=False,
                      cartesian=False, start_step=1, elements=None,
                      cell_vectors=None):
    """
    从 VASP 原始输出提取到 step_* 目录（保持向后兼容）。
    返回 (提取步数, 元素列表, 晶胞向量).
    """
    if elements is None or cell_vectors is None:
        num_atoms, elements_local, _, cell_vectors_local, num_steps_total = \
            _read_xdatcar_info_legacy(xdatcar_path)
        if elements is None:
            elements = elements_local
        if cell_vectors is None:
            cell_vectors = cell_vectors_local
    else:
        num_atoms = len(elements)
        num_steps_total = sum(1 for line in open(xdatcar_path)
                              if "Direct configuration=" in line)

    log.info(f"  {os.path.basename(xdatcar_path)}: 原子={num_atoms}, 步数={num_steps_total}")

    coords_frac = _read_xdatcar_coords_legacy(xdatcar_path, num_atoms, num_steps_total)
    energies = _read_outcar_energies_legacy(outcar_path, num_steps_total)
    if energies is None:
        return 0, elements, cell_vectors

    os.makedirs(output_dir, exist_ok=True)

    for step in range(num_steps_total):
        global_step = start_step + step
        step_dir = os.path.join(output_dir, f"step_{global_step:05d}")
        os.makedirs(step_dir, exist_ok=True)

        coords = coords_frac[step].copy()
        if wrap:
            coords = coords % 1.0
        if cartesian:
            coords_out = coords @ cell_vectors
            coord_label = "Cartesian (Å)"
        else:
            coords_out = coords
            coord_label = "Fractional"

        # structure.xyz
        xyz_lines = [
            str(num_atoms),
            f"Step {global_step}, Energy={energies[step]:.8f} eV, {coord_label}"
        ]
        for i, elem in enumerate(elements):
            xyz_lines.append(f"{elem}  {coords_out[i,0]:.8f}  {coords_out[i,1]:.8f}  {coords_out[i,2]:.8f}")
        with open(os.path.join(step_dir, "structure.xyz"), "w") as f:
            f.write("\n".join(xyz_lines) + "\n")

        # energy.txt
        with open(os.path.join(step_dir, "energy.txt"), "w") as f:
            f.write(f"# Step {global_step}\n")
            f.write(f"# Energy from OUTCAR: {energies[step]:.8f} eV\n")

    return num_steps_total, elements, cell_vectors


def cmd_extract_steps(args):
    """
    --extract-steps 子命令入口。
    从 VASP MD (OUTCAR+XDATCAR) 提取为 step_* 目录格式。
    """
    if args.xdatcar2 and args.outcar2:
        # 续算合并模式
        log.info("=== 续算合并模式 ===")
        log.info(f"第一段: {args.xdatcar} + {args.outcar}")
        log.info(f"第二段: {args.xdatcar2} + {args.outcar2}")

        num_atoms, elements, elem_counts, cell, num_steps1 = \
            _read_xdatcar_info_legacy(args.xdatcar)
        log.info(f"第一段: 原子={num_atoms} ({elem_counts}), 步数={num_steps1}")
        a, b, c = np.linalg.norm(cell, axis=1)
        log.info(f"晶胞: a={a:.5f} b={b:.5f} c={c:.5f}")

        log.info(f"提取第一段 (步 1 ~ {num_steps1})...")
        n1, elements, cell = _extract_to_steps(
            args.xdatcar, args.outcar, args.out,
            wrap=args.wrap, cartesian=args.cartesian,
            start_step=1, elements=elements, cell_vectors=cell)

        log.info(f"提取第二段 (步 {num_steps1+1} ~ {num_steps1+8000})...")
        n2, _, _ = _extract_to_steps(
            args.xdatcar2, args.outcar2, args.out,
            wrap=args.wrap, cartesian=args.cartesian,
            start_step=num_steps1 + 1,
            elements=elements, cell_vectors=cell)

        total = n1 + n2
        log.info(f"完成！共 {total} 步 → {args.out}/")
    else:
        # 单段模式
        num_atoms, elements, elem_counts, cell, num_steps = \
            _read_xdatcar_info_legacy(args.xdatcar)
        log.info(f"检测: 原子={num_atoms} ({elem_counts}), 步数={num_steps}")
        a, b, c = np.linalg.norm(cell, axis=1)
        log.info(f"晶胞: a={a:.5f} b={b:.5f} c={c:.5f}")

        n, _, _ = _extract_to_steps(
            args.xdatcar, args.outcar, args.out,
            wrap=args.wrap, cartesian=args.cartesian,
            start_step=1)
        log.info(f"完成！共 {n} 步 → {args.out}/")


# ═══════════════════════════════════════════════════════════════
#  Part A: 从 VASP 原始输出提取 (OUTCAR + XDATCAR + POSCAR)
# ═══════════════════════════════════════════════════════════════

def read_incar_metadata(incar_path):
    """读取 INCAR 获取运行参数"""
    meta = {}
    if not os.path.exists(incar_path):
        return meta
    with open(incar_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key, val = key.strip().upper(), val.split("#")[0].split("!")[0].strip()
                if key in ("NSW", "TEBEG", "TEEND", "POTIM", "NELMIN", "NELM"):
                    try:
                        meta[key] = float(val) if "." in val else int(val)
                    except ValueError:
                        pass
    return meta


def read_xdatcar_info(xdatcar_path):
    """
    读取 XDATCAR 头部信息。
    返回 (num_atoms, elements_list, element_counts, cell_3x3, num_steps)
    """
    with open(xdatcar_path) as f:
        lines = f.readlines()

    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[i + 2].strip().split()]

    elements_line = lines[5].strip().split()
    counts_line = lines[6].strip().split()
    element_counts = {}
    for elem, cnt in zip(elements_line, counts_line):
        element_counts[elem] = int(cnt)

    num_atoms = sum(element_counts.values())
    elements = []
    for elem, cnt in zip(elements_line, counts_line):
        elements.extend([elem] * int(cnt))

    num_steps = sum(1 for line in lines if "Direct configuration=" in line)
    return num_atoms, elements, element_counts, cell, num_steps


def read_xdatcar_coords(xdatcar_path, num_atoms, num_steps):
    """读取 XDATCAR 中所有步的分数坐标 (num_steps, num_atoms, 3)"""
    with open(xdatcar_path) as f:
        lines = f.readlines()

    config_idxs = [i for i, line in enumerate(lines) if "Direct configuration=" in line]
    assert len(config_idxs) == num_steps

    coords = np.zeros((num_steps, num_atoms, 3))
    for step_idx, start in enumerate(config_idxs):
        for atom_idx in range(num_atoms):
            parts = lines[start + 1 + atom_idx].strip().split()
            coords[step_idx, atom_idx] = [float(parts[0]), float(parts[1]), float(parts[2])]
    return coords


def read_outcar_energies(outcar_path, num_steps):
    """
    从 OUTCAR 提取 MD 步总能。
    VASP MD: 每步最后一次 "energy  without entropy"（两个空格）即为该步总能。
    """
    with open(outcar_path) as f:
        content = f.read()

    pattern = r"energy  without entropy\s*=\s*([-.\d]+)"
    energies = [float(x) for x in re.findall(pattern, content)]

    if len(energies) < num_steps:
        log.warning(f"  OUTCAR 能量数 ({len(energies)}) < 步数 ({num_steps})，"
                     f"尝试 'energy without entropy'（单空格）...")
        pattern2 = r"energy without entropy\s*=\s*([-.\d]+)"
        energies = [float(x) for x in re.findall(pattern2, content)]
        if len(energies) >= num_steps:
            energies = energies[-num_steps:]
        else:
            log.error(f"  仍不足: {len(energies)} < {num_steps}")
            return None

    return np.array(energies[:num_steps])


def read_poscar_cell(poscar_path):
    """从 POSCAR 读取晶胞（可选）"""
    if not os.path.exists(poscar_path):
        return None
    with open(poscar_path) as f:
        lines = f.readlines()
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[i + 2].strip().split()]
    return cell


def extract_from_vasp_dir(vasp_dir, cartesian=True):
    """
    从 VASP 输出目录提取所有步。
    vasp_dir 应包含 OUTCAR, XDATCAR（可选 POSCAR/INCAR）。
    返回 list of dicts.
    """
    outcar_path = os.path.join(vasp_dir, "OUTCAR")
    xdatcar_path = os.path.join(vasp_dir, "XDATCAR")
    poscar_path = os.path.join(vasp_dir, "POSCAR")
    incar_path = os.path.join(vasp_dir, "INCAR")

    if not os.path.exists(outcar_path):
        log.warning(f"  跳过 {vasp_dir}: 无 OUTCAR")
        return None
    if not os.path.exists(xdatcar_path):
        log.warning(f"  跳过 {vasp_dir}: 无 XDATCAR")
        return None

    log.info(f"读取 VASP 目录: {os.path.basename(vasp_dir)}")

    meta = read_incar_metadata(incar_path)
    if meta:
        log.info(f"  INCAR: NSW={meta.get('NSW','?')} TEBEG={meta.get('TEBEG','?')}K "
                 f"POTIM={meta.get('POTIM','?')}fs")

    num_atoms, elements, elem_counts, cell, num_steps = read_xdatcar_info(xdatcar_path)
    log.info(f"  原子: {num_atoms} ({elem_counts}), 步数: {num_steps}")
    a, b, c = np.linalg.norm(cell, axis=1)
    log.info(f"  晶胞: a={a:.4f} b={b:.4f} c={c:.4f}")

    coords_frac = read_xdatcar_coords(xdatcar_path, num_atoms, num_steps)
    energies = read_outcar_energies(outcar_path, num_steps)
    if energies is None:
        return None

    log.info(f"  能量: {energies[-1]:.4f} eV (最后一步), "
             f"范围 [{energies.min():.4f}, {energies.max():.4f}]")

    poscar_cell = read_poscar_cell(poscar_path)
    if poscar_cell is not None:
        diff = np.linalg.norm(poscar_cell - cell)
        if diff > 1e-4:
            log.info(f"  注意: POSCAR 与 XDATCAR 晶胞差异 {diff:.6f}，使用 XDATCAR")

    results = []
    for step in range(num_steps):
        coords = coords_frac[step].copy()
        coords = coords % 1.0
        positions = coords @ cell if cartesian else coords

        results.append({
            "symbols": elements[:],
            "positions": positions.copy(),
            "energy": float(energies[step]),
            "step": step + 1,
            "cell": cell.copy(),
            "source": os.path.basename(vasp_dir),
        })

    return results


# ═══════════════════════════════════════════════════════════════
#  Part B: 从已提取的 step_* 目录读取
# ═══════════════════════════════════════════════════════════════

def read_step_dir(step_dir):
    """读取一个 step_* 目录的 structure.xyz 和 energy.txt"""
    xyz_path = os.path.join(step_dir, "structure.xyz")
    energy_path = os.path.join(step_dir, "energy.txt")
    if not os.path.exists(xyz_path) or not os.path.exists(energy_path):
        return None

    with open(energy_path) as f:
        lines = f.readlines()
    try:
        if lines[0].strip().startswith("#"):
            step_num = int(lines[0].strip().split()[1])
            energy = float(lines[1].strip().split()[4])
        else:
            step_num = int(lines[0].strip().split()[1])
            energy = float(lines[1].strip().split()[4])
    except (IndexError, ValueError):
        try:
            parts = lines[1].strip().split()
            for p in parts:
                if p.replace(".", "").replace("-", "").isdigit() and "." in p:
                    energy = float(p)
                    break
            step_num = int(lines[0].strip().split()[-1])
        except Exception:
            return None

    with open(xyz_path) as f:
        xyz_lines = f.readlines()
    try:
        num_atoms = int(xyz_lines[0].strip())
        symbols, positions = [], []
        for i in range(num_atoms):
            parts = xyz_lines[2 + i].strip().split()
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except (IndexError, ValueError):
        return None

    return {"symbols": symbols, "positions": np.array(positions),
            "energy": energy, "step": step_num}


def extract_from_steps_dir(steps_dir):
    """
    从 step_*/ 目录提取所有步。
    返回 list of dicts.
    """
    log.info(f"读取步骤目录: {os.path.basename(steps_dir)}")

    entries = sorted(os.listdir(steps_dir))
    step_dirs = [d for d in entries
                 if d.startswith("step_") and os.path.isdir(os.path.join(steps_dir, d))]
    log.info(f"  发现 {len(step_dirs)} 个 step 目录")
    if not step_dirs:
        return None

    # 尝试从注释行获取晶胞
    cell = None
    first_xyz = os.path.join(steps_dir, step_dirs[0], "structure.xyz")
    if os.path.exists(first_xyz):
        with open(first_xyz) as f:
            flines = f.readlines()
        if len(flines) >= 2 and "Lattice=" in flines[1]:
            try:
                import shlex
                for token in shlex.split(flines[1].strip()):
                    if token.startswith("Lattice="):
                        vals = token.split("=", 1)[1].strip('"').split()
                        cell = np.array([float(v) for v in vals]).reshape(3, 3)
                        break
            except Exception:
                pass

    results = []
    for d in step_dirs:
        result = read_step_dir(os.path.join(steps_dir, d))
        if result is not None:
            result["cell"] = cell.copy() if cell is not None else None
            result["source"] = os.path.basename(steps_dir)
            results.append(result)

    log.info(f"  有效帧: {len(results)}/{len(step_dirs)}")
    return results


# ═══════════════════════════════════════════════════════════════
#  Part C: 异常检测
# ═══════════════════════════════════════════════════════════════

def detect_anomalies(data_list, z_score_thresh=4.0, min_dist_thresh=0.3,
                     energy_window=None):
    """
    异常检测 pipeline:
      1. 能量 Z-score 离群
      2. 局部能量尖峰
      3. 原子间距结构异常
    """
    n = len(data_list)
    if n == 0:
        return data_list, []

    all_energies = np.array([d["energy"] for d in data_list])
    anomalies = []

    # 1. 能量 Z-score
    median_e = np.median(all_energies)
    mad = np.median(np.abs(all_energies - median_e))
    if mad < 1e-8:
        mad = all_energies.std()
    modified_z = 0.6745 * (all_energies - median_e) / max(mad, 1e-8)
    z_mask = np.abs(modified_z) > z_score_thresh
    z_bad_idxs = set(np.where(z_mask)[0])
    if np.any(z_mask):
        log.info(f"  能量 Z-score 离群 (>{z_score_thresh}σ): {z_mask.sum()} 帧")
        for idx in sorted(z_bad_idxs):
            anomalies.append({
                "idx": idx, "step": data_list[idx]["step"],
                "energy": data_list[idx]["energy"],
                "reason": f"energy_zscore={modified_z[idx]:.2f}"
            })

    # 2. 滑动窗口尖峰
    if energy_window is not None and energy_window >= 3:
        half_w = energy_window // 2
        for i in range(half_w, n - half_w):
            if i in z_bad_idxs:
                continue
            local = all_energies[i - half_w:i + half_w + 1]
            local_median = np.median(local)
            local_mad = np.median(np.abs(local - local_median))
            if local_mad > 1e-8:
                z_local = 0.6745 * (all_energies[i] - local_median) / local_mad
                if abs(z_local) > z_score_thresh * 1.5:
                    anomalies.append({
                        "idx": i, "step": data_list[i]["step"],
                        "energy": data_list[i]["energy"],
                        "reason": f"local_spike_z={z_local:.2f}"
                    })

    all_bad = set(a["idx"] for a in anomalies)

    # 3. 结构异常
    from scipy.spatial import cKDTree
    struct_bad = 0
    for i, d in enumerate(data_list):
        if i in all_bad or len(d["positions"]) < 2:
            continue
        tree = cKDTree(d["positions"])
        pairs = tree.query_pairs(min_dist_thresh)
        if pairs:
            struct_bad += 1
            all_bad.add(i)
            anomalies.append({
                "idx": i, "step": d["step"],
                "energy": d["energy"],
                "reason": f"min_dist<{min_dist_thresh}A ({len(pairs)} pairs)"
            })

    if struct_bad > 0:
        log.info(f"  结构异常 (min_dist<{min_dist_thresh}A): {struct_bad} 帧")

    clean_list = [d for i, d in enumerate(data_list) if i not in all_bad]
    log.info(f"  异常总计: {len(anomalies)}/{n} = {len(anomalies)/n*100:.1f}%")
    log.info(f"  剩余干净帧: {len(clean_list)}")
    return clean_list, anomalies


# ═══════════════════════════════════════════════════════════════
#  Part D: 输出 extxyz
# ═══════════════════════════════════════════════════════════════

def write_extxyz(filename, atoms_list):
    """写入 extxyz 格式"""
    with open(filename, "w") as f:
        for atoms in atoms_list:
            natoms = len(atoms["symbols"])
            f.write(f"{natoms}\n")
            info_parts = [f"energy={atoms['energy']:.10f}"]
            if atoms.get("step") is not None:
                info_parts.append(f"step={atoms['step']}")
            if atoms.get("source") is not None:
                info_parts.append(f'source="{atoms["source"]}"')
            cell = atoms.get("cell")
            if cell is not None and cell.size > 0:
                c = cell.flatten()
                info_parts.append(
                    f'Lattice="{c[0]:.10f} {c[1]:.10f} {c[2]:.10f} '
                    f'{c[3]:.10f} {c[4]:.10f} {c[5]:.10f} '
                    f'{c[6]:.10f} {c[7]:.10f} {c[8]:.10f}"')
            info_parts.append('pbc="T T T"')
            f.write(" ".join(info_parts) + "\n")
            pos = atoms["positions"]
            for i in range(natoms):
                f.write(f"{atoms['symbols'][i]} {pos[i,0]:.10f} {pos[i,1]:.10f} {pos[i,2]:.10f}\n")


def collect_cell(all_results):
    """从结果中收集晶胞信息"""
    for d in all_results:
        c = d.get("cell")
        if c is not None and c.size > 0 and np.linalg.norm(c) > 0:
            return c
    return np.eye(3) * 10.0


# ═══════════════════════════════════════════════════════════════
#  Main: 数据转换
# ═══════════════════════════════════════════════════════════════

def build_convert_parser(subparsers=None):
    """构建数据转换模式的参数解析器（同时支持作为主解析器和子命令）"""
    if subparsers is not None:
        p = subparsers.add_parser("convert", help="从 VASP/step_* → extxyz 训练/测试集（默认模式）")
    else:
        p = argparse.ArgumentParser(add_help=False)
    # 输入源
    p.add_argument("--vasp-dir", type=str, default=None,
                   help="包含 OUTCAR+XDATCAR 的 VASP 输出目录")
    p.add_argument("--vasp-dirs", type=str, nargs="+", default=None,
                   help="多个 VASP 输出目录（空格分隔）")
    p.add_argument("--steps-dir", type=str, default=None,
                   help="已提取的 step_* 目录")
    p.add_argument("--steps-dirs", type=str, nargs="+", default=None,
                   help="多个 step_* 目录（空格分隔）")
    # 自动扫描
    p.add_argument("--scan-unloaded", action="store_true", default=True,
                   help="自动扫描 ../unloaded_data/ 下所有子目录（默认开启）")
    p.add_argument("--scan-loaded", action="store_true", default=False,
                   help="同时扫描 ../loaded_data/ 下所有子目录")
    # 输出
    p.add_argument("--out", type=str, default=None,
                   help="输出目录（默认 allegro/data/）")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="训练集比例 (默认 0.8)")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 (默认 42)")
    # 异常检测
    p.add_argument("--z-score", type=float, default=4.0,
                   help="能量 Z-score 离群阈值 (默认 4.0)")
    p.add_argument("--min-dist", type=float, default=0.3,
                   help="最小原子间距 A（结构异常阈值）")
    p.add_argument("--spike-window", type=int, default=None,
                   help="能量尖峰滑动窗口大小 (默认 auto: n//20)")
    # 坐标
    p.add_argument("--cartesian", action="store_true", default=True,
                   help="输出笛卡尔坐标 (默认开启)")
    return p


def scan_vasp_dirs(base_dir):
    """扫描目录，返回包含 OUTCAR 的子目录列表"""
    if not os.path.isdir(base_dir):
        return []
    return [os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir))
            if os.path.isdir(os.path.join(base_dir, d))
            and os.path.exists(os.path.join(base_dir, d, "OUTCAR"))
            and os.path.exists(os.path.join(base_dir, d, "XDATCAR"))]


def cmd_convert(args):
    """数据转换主流程：VASP/step_* → extxyz"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 输出目录
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(script_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    # ── 收集输入源 ──
    source_dirs = []

    if args.vasp_dir:
        source_dirs.append(("vasp", args.vasp_dir))
    if args.vasp_dirs:
        for d in args.vasp_dirs:
            source_dirs.append(("vasp", d))
    if args.steps_dir:
        source_dirs.append(("steps", args.steps_dir))
    if args.steps_dirs:
        for d in args.steps_dirs:
            source_dirs.append(("steps", d))
    if args.scan_unloaded:
        base = os.path.join(script_dir, "..", "unloaded_data")
        dirs = scan_vasp_dirs(base)
        if dirs:
            log.info(f"扫描 unloaded_data/: 发现 {len(dirs)} 个 VASP 目录")
            for d in dirs:
                source_dirs.append(("vasp", d))
        else:
            log.info("unloaded_data/: 无 VASP 目录")
    if args.scan_loaded:
        base = os.path.join(script_dir, "..", "loaded_data")
        dirs = scan_vasp_dirs(base)
        if dirs:
            log.info(f"扫描 loaded_data/: 发现 {len(dirs)} 个 VASP 目录")
            for d in dirs:
                source_dirs.append(("vasp", d))
        else:
            log.info("loaded_data/: 无 VASP 目录")

    if not source_dirs:
        log.error("未指定输入源！使用 --vasp-dir / --steps-dir 或将 VASP 输出放入 ../unloaded_data/")
        sys.exit(1)

    # ── 提取 ──
    all_results = []
    for src_type, src_path in source_dirs:
        abs_path = os.path.abspath(src_path)
        if not os.path.exists(abs_path):
            log.warning(f"  路径不存在: {abs_path}")
            continue
        log.info("")
        results = (extract_from_vasp_dir(abs_path, cartesian=args.cartesian)
                   if src_type == "vasp" else extract_from_steps_dir(abs_path))
        if results:
            all_results.extend(results)
        else:
            log.warning(f"  无数据: {abs_path}")

    # 统一晶胞
    unified_cell = collect_cell(all_results)
    for d in all_results:
        c = d.get("cell")
        if c is None or c.size == 0 or np.linalg.norm(c) < 1e-8:
            d["cell"] = unified_cell.copy()

    total_raw = len(all_results)
    log.info(f"\n{'='*60}")
    log.info(f"原始数据总计: {total_raw} 帧")
    if total_raw == 0:
        log.error("无有效数据！")
        sys.exit(1)

    sources = Counter(d.get("source", "unknown") for d in all_results)
    log.info("来源分布:")
    for src, cnt in sources.most_common():
        log.info(f"  {src}: {cnt} 帧")

    # ── 异常检测 ──
    if args.spike_window is None:
        args.spike_window = max(len(all_results) // 20, 5)
    log.info(f"\n异常检测 (Z-score>{args.z_score}, min_dist<{args.min_dist}A, window={args.spike_window}):")
    clean_results, anomalies = detect_anomalies(
        all_results, z_score_thresh=args.z_score,
        min_dist_thresh=args.min_dist, energy_window=args.spike_window)

    if anomalies:
        log.info("\n异常帧详情:")
        log.info(f"  {'Idx':<6} {'Step':<8} {'能量':<16} {'原因'}")
        log.info(f"  {'-'*60}")
        for a in anomalies:
            log.info(f"  {a['idx']:<6} {a['step']:<8} {a['energy']:<16.6f} {a['reason']}")

    # ── 分割 ──
    random.seed(args.seed)
    random.shuffle(clean_results)
    n_total = len(clean_results)
    n_train = int(n_total * args.train_ratio)
    train_data = clean_results[:n_train]
    test_data = clean_results[n_train:]

    train_energies = [d["energy"] for d in train_data]
    test_energies = [d["energy"] for d in test_data]

    log.info(f"\n{'='*60}")
    log.info("最终数据集:")
    log.info(f"  训练集: {len(train_data)} 帧")
    log.info(f"  测试集: {len(test_data)} 帧")
    log.info(f"  能量范围: [{min(train_energies + test_energies):.4f}, "
             f"{max(train_energies + test_energies):.4f}] eV")

    symbols_sets = set(tuple(d["symbols"]) for d in train_data)
    if len(symbols_sets) > 1:
        log.warning(f"  警告: 训练集中存在 {len(symbols_sets)} 种不同原子组成！")
    log.info(f"  体系: {list(symbols_sets)[0] if symbols_sets else '?'}")

    # ── 写出 ──
    train_path = os.path.join(out_dir, "train.xyz")
    test_path = os.path.join(out_dir, "test.xyz")

    if os.path.exists(train_path):
        backup = train_path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(train_path, backup)
            log.info(f"  旧训练集备份: {backup}")

    write_extxyz(train_path, train_data)
    write_extxyz(test_path, test_data)

    log.info(f"  训练集: {train_path}  ({os.path.getsize(train_path)/1024:.1f} KB)")
    log.info(f"  测试集: {test_path}  ({os.path.getsize(test_path)/1024:.1f} KB)")
    log.info(f"  训练集能量: {np.mean(train_energies):.4f} ± {np.std(train_energies):.4f} eV")
    log.info(f"  测试集能量: {np.mean(test_energies):.4f} ± {np.std(test_energies):.4f} eV")

    # 报告
    report_path = os.path.join(out_dir, "data_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Data Preparation Report\n{'='*60}\n")
        f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Sources: {len(source_dirs)} directories\n")
        for st, sp in source_dirs:
            f.write(f"  [{st}] {sp}\n")
        f.write(f"\nRaw frames: {total_raw}\n")
        f.write(f"Anomalies removed: {len(anomalies)}\n")
        f.write(f"Clean frames: {n_total}\n")
        f.write(f"Train: {n_train}, Test: {n_total - n_train}\n")
        f.write(f"Train energy: {np.mean(train_energies):.4f} +/- {np.std(train_energies):.4f} eV\n")
        f.write(f"Test energy: {np.mean(test_energies):.4f} +/- {np.std(test_energies):.4f} eV\n")
        f.write(f"Energy range: [{min(train_energies + test_energies):.4f}, "
                f"{max(train_energies + test_energies):.4f}] eV\n")
        if anomalies:
            f.write(f"\nAnomalies:\n{'Step':<8} {'Energy':<16} {'Reason'}\n{'-'*50}\n")
            for a in anomalies:
                f.write(f"{a['step']:<8} {a['energy']:<16.6f} {a['reason']}\n")
    log.info(f"  数据报告: {report_path}")
    log.info("\n完成！可用 train.py 开始训练，或用 test_model_forward.py 验证。")


# ═══════════════════════════════════════════════════════════════
#  参数解析与入口
# ═══════════════════════════════════════════════════════════════

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="统一数据预处理与依赖检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 自动扫描并转换
  %(prog)s

  # 指定 VASP 目录
  %(prog)s --vasp-dir ../unloaded_data/hot3

  # 从 step_* 目录读取
  %(prog)s --steps-dir ../vasp_steps_fixed

  # 提取 VASP MD 到 step_* 目录
  %(prog)s --extract-steps --xdatcar MD/XDATCAR --outcar MD/OUTCAR

  # 检查依赖
  %(prog)s --check-deps
""")

    # 子命令模式（互斥）
    parser.add_argument("--check-deps", action="store_true",
                        help="检查所有 Python 依赖是否安装")
    parser.add_argument("--extract-steps", action="store_true",
                        help="提取 VASP 原始输出为 step_* 目录（旧格式）")
    parser.add_argument("--xdatcar", type=str, default=None,
                        help="(--extract-steps 模式) XDATCAR 文件路径")
    parser.add_argument("--xdatcar2", type=str, default=None,
                        help="(--extract-steps 模式) 续算 XDATCAR")
    parser.add_argument("--outcar", type=str, default=None,
                        help="(--extract-steps 模式) OUTCAR 文件路径")
    parser.add_argument("--outcar2", type=str, default=None,
                        help="(--extract-steps 模式) 续算 OUTCAR")
    parser.add_argument("--wrap", action="store_true", default=False,
                        help="(--extract-steps 模式) 折叠分数坐标到 [0,1)")
    parser.add_argument("--cartesian-extract", action="store_true", default=False,
                        help="(--extract-steps 模式) 输出笛卡尔坐标")
    # 数据转换模式参数
    parser.add_argument("--vasp-dir", type=str, default=None)
    parser.add_argument("--vasp-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--steps-dir", type=str, default=None)
    parser.add_argument("--steps-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-score", type=float, default=4.0)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--spike-window", type=int, default=None)
    parser.add_argument("--cartesian", action="store_true", default=True)
    parser.add_argument("--scan-unloaded", action="store_true", default=True)
    parser.add_argument("--scan-loaded", action="store_true", default=False)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # ═══ 模式 1: 依赖检查 ═══
    if args.check_deps:
        cmd_check_deps()
        return

    # ═══ 模式 2: VASP → step_* ═══
    if args.extract_steps:
        if not args.xdatcar or not args.outcar:
            log.error("--extract-steps 模式需要 --xdatcar 和 --outcar")
            sys.exit(1)
        if args.out is None:
            args.out = "vasp_steps"
        # 确保 extract_steps 输出目录存在
        os.makedirs(args.out, exist_ok=True)
        cmd_extract_steps(args)
        return

    # ═══ 模式 3 (默认): 数据转换 ═══
    cmd_convert(args)


if __name__ == "__main__":
    main()
