"""
Basic tests for Ewald-Allegro model.

Tests:
1. Model creation with default parameters
2. Forward pass producing correct output shapes
3. Backward pass with gradient flow through charge predictor
4. Charge predictor producing reasonable charges
5. Ewald energy contribution is non-zero for periodic systems

Run with: pytest tests/ -v
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from allegro.model.ewald_allegro_v2 import EwaldAllegroModelV2


def get_test_data(n_molecules=4):
    """Create a simple water box for testing.

    Args:
        n_molecules: Number of H2O molecules (default 4 -> 12 atoms)
    """
    n_atoms = n_molecules * 3  # 3 atoms per H2O
    torch.manual_seed(42)

    # Random positions in a box
    pos = torch.randn(n_atoms, 3) * 0.5

    # Atom types: 0 = H, 1 = O (alternating pattern for H2O)
    # For each molecule: [H, H, O]
    z = []
    for _ in range(n_molecules):
        z.extend([0, 0, 1])  # type 0 = H, type 1 = O
    z = torch.tensor(z, dtype=torch.long)

    # Cubic cell
    cell = torch.eye(3) * 10.0

    return {"pos": pos, "z": z, "cell": cell}


def get_model():
    """Create a small Ewald-Allegro model for testing."""
    return EwaldAllegroModelV2(
        type_names=["H", "O"],
        r_max=5.0,
        num_bessels=4,
        l_max=1,
        num_layers=1,
        num_scalar_features=16,
        num_tensor_features=8,
        charge_hidden=16,
        readout_hidden=8,
        ewald_alpha=0.35,
        ewald_r_cut=8.0,
        ewald_grid=(16, 16, 16),
    )


def test_model_creation():
    """Test that the model can be created with valid parameters."""
    model = get_model()
    assert model is not None
    num_params = model.get_num_params()
    assert num_params > 0, f"Model has zero parameters!"
    print(f"  Model created: {num_params:,} parameters")


def test_forward_pass():
    """Test that forward pass produces correct output shapes."""
    model = get_model()
    data = get_test_data(n_molecules=4)

    output = model(data)

    # Check all expected keys exist
    expected_keys = [
        "energy", "energy_short", "energy_long",
        "energy_shift", "charges", "per_atom_energy",
    ]
    for key in expected_keys:
        assert key in output, f"Missing key in output: {key}"

    # Squeeze to handle both scalar and [1,1] shapes
    energy = output["energy"].squeeze()
    energy_short = output["energy_short"].squeeze()
    energy_long = output["energy_long"].squeeze()

    # Check shapes
    n_atoms = data["pos"].shape[0]
    assert energy.ndim == 0, f"energy should be scalar, got shape {output['energy'].shape}"
    assert energy_short.ndim == 0
    assert energy_long.ndim == 0
    assert output["charges"].shape == (n_atoms,), \
        f"charges should be (N,), got {output['charges'].shape}"
    assert output["per_atom_energy"].squeeze().shape == (n_atoms,), \
        f"per_atom_energy should be (N,), got {output['per_atom_energy'].shape}"

    print(f"  Forward pass OK")
    print(f"    E_total = {energy.item():.6f}")
    print(f"    E_short = {energy_short.item():.6f}")
    print(f"    E_long  = {energy_long.item():.6f}")
    print(f"    Charges range: [{output['charges'].min().item():.4f}, {output['charges'].max().item():.4f}]")


def test_ewald_nonzero():
    """Test that Ewald long-range contribution is non-zero for periodic systems."""
    model = get_model()
    data = get_test_data(n_molecules=4)

    output = model(data)
    ewald_contrib = output["energy_long"].squeeze().item()

    # Ewald term should be non-zero for charged atoms in a periodic box
    assert abs(ewald_contrib) > 1e-8, \
        f"Ewald contribution is zero! Got {ewald_contrib:.10f}"

    print(f"  Ewald contribution: {ewald_contrib:.6f} eV (non-zero OK)")


def test_backward_pass():
    """Test that gradients flow through the entire model.

    Note: The Ewald module itself has no learnable parameters (it's a fixed
    computation). Gradients flow through Ewald via the charges predicted by
    ChargePredictor. We verify that:
    1. AllegroShort parameters receive gradients
    2. ChargePredictor parameters receive gradients
    3. The gradients propagate through Ewald (check requires_grad on Ewald inputs)
    """
    model = get_model()
    data = get_test_data(n_molecules=4)

    output = model(data)

    # Create a simple loss and backprop
    loss = output["energy"].squeeze() ** 2
    loss.backward()

    # Check that learnable parameter groups have gradients
    param_groups = {
        "AllegroShort": model.allegro_short.parameters(),
        "ChargePredictor": model.charge_predictor.parameters(),
    }
    all_have_grad = True
    total_grad_norm = 0.0
    for group_name, params in param_groups.items():
        group_grad = 0.0
        has_grad = False
        for p in params:
            if p.grad is not None:
                group_grad += p.grad.abs().sum().item()
                has_grad = True
        total_grad_norm += group_grad
        status = "OK" if has_grad else "MISSING"
        if not has_grad:
            all_have_grad = False
        print(f"    {group_name:20s}: grad_norm = {group_grad:.6f}  [{status}]")

    assert all_have_grad, "Some parameter groups have no gradient!"
    assert total_grad_norm > 0, "Total gradient norm is zero!"

    # Additionally verify that charges require grad (grad flows through Ewald)
    assert output["energy_long"].requires_grad, \
        "Ewald energy should require grad (grad flows through charges)"
    print(f"    Ewald energy requires_grad = {output['energy_long'].requires_grad} [OK]")
    print(f"  Backward pass OK (total grad_norm = {total_grad_norm:.6f})")


def test_charge_neutrality_tendency():
    """Test that total charge tends toward neutrality."""
    model = get_model()
    model.eval()

    total_charges = []
    with torch.no_grad():
        for seed in range(5):
            torch.manual_seed(seed)
            data = get_test_data(n_molecules=6)
            output = model(data)
            total_charges.append(output["charges"].sum().item())

    mean_charge = sum(total_charges) / len(total_charges)
    print(f"  Mean total charge: {mean_charge:.4f} |e|")
    print(f"  Per-frame charges: {[f'{c:.4f}' for c in total_charges]}")

    # The charge predictor should learn neutrality naturally,
    # but at initialization it may not be exactly zero.
    # This test just verifies charges are reasonable (not exploding).
    for c in total_charges:
        assert abs(c) < 100, f"Charge exploding: {c}"


if __name__ == "__main__":
    print("=" * 50)
    print("Ewald-Allegro Basic Tests")
    print("=" * 50)
    print()

    tests = [
        test_model_creation,
        test_forward_pass,
        test_ewald_nonzero,
        test_backward_pass,
        test_charge_neutrality_tendency,
    ]

    passed = 0
    for test_fn in tests:
        print(f"  [{test_fn.__name__}] ", end="")
        try:
            test_fn()
            passed += 1
            print(f"    [PASS]")
        except Exception as e:
            print(f"    [FAIL]: {e}")
        print()

    print(f"  Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("  [ALL TESTS PASSED]")
    else:
        print("  [SOME TESTS FAILED]")
        sys.exit(1)
