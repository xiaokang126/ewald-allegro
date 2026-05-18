# Theory

## Ewald Summation for Interatomic Potentials

### The Problem

Standard short-range interatomic potentials (including the original Allegro)
truncate atomic interactions at a cutoff radius $r_{\max}$. This is adequate
for systems dominated by covalent bonding, but fails for **polar systems**
(water, electrolytes, ionic liquids) where long-range electrostatic
interactions decay slowly as $1/r$.

### Ewald Summation

The Ewald method splits the electrostatic energy into three terms:

$$
E_{\text{long}} = E_{\text{real}} + E_{\text{reciprocal}} - E_{\text{self}}
$$

**Real-space term** (short-range, computed in direct space):

$$
E_{\text{real}} = \frac{1}{2} \sum_{i,j} \sum_{\mathbf{n}}
q_i \, q_j \,
\frac{\text{erfc}\bigl(\alpha \, |\mathbf{r}_{ij} + \mathbf{n}|\bigr)}
{|\mathbf{r}_{ij} + \mathbf{n}|}
$$

This term decays rapidly and is truncated at $r_{\text{cut}}$.

**Reciprocal-space term** (long-range, computed in Fourier space):

$$
E_{\text{reciprocal}} =
\frac{1}{2\pi V} \sum_{\mathbf{k} \neq \mathbf{0}}
\frac{\exp\bigl(-k^2 / 4\alpha^2\bigr)}{k^2}
\, |S(\mathbf{k})|^2
$$

where $S(\mathbf{k}) = \sum_i q_i \, \exp(i \mathbf{k} \cdot \mathbf{r}_i)$
is the structure factor.

**Self-interaction correction**:

$$
E_{\text{self}} = \frac{\alpha}{\sqrt{\pi}} \sum_i q_i^2
$$

The splitting parameter $\alpha$ controls the balance between real-space and
reciprocal-space computations.

### Differentiability

All three terms are **analytically differentiable** with respect to:

- Atomic positions $\mathbf{r}_i$ ($\rightarrow$ forces)
- Partial charges $q_i$ ($\rightarrow$ charge gradients)
- Lattice vectors ($\rightarrow$ stress)

This enables end-to-end gradient backpropagation through the entire Ewald
summation.

### Why O(N) Complexity

By using:

- **Cell lists** for $O(N)$ real-space neighbor search (instead of $O(N^2)$)
- **FFT** for the reciprocal-space term ($O(N \log N)$, effectively $O(N)$
  for practical grid sizes)
- **Fixed FFT grid** independent of system size

The overall complexity scales as $O(N)$.

---

## Ewald-Allegro Architecture

```
Input(pos, z, cell)
    │
    ├─── Allegro Short-Range Network ──── E_short
    │        │
    │        └── Edge Features (from Allegro latent layers)
    │                    │
    │              [scatter sum over neighbors]
    │                    │
    │              Node Features
    │                    │
    │                    ↓
    │           ChargePredictor (MLP)
    │                    │
    │                    ↓  {q_i}
    │                    │
    │                    ├── Ewald Real-space ── E_real
    │                    ├── Ewald Reciprocal ── E_reciprocal
    │                    └── Self Correction ─── E_self
    │                              │
    │                              ↓
    │                          E_long
    │                              │
    └──────────────────────────────┤
                                    ↓
                             E_total = E_short + E_long + shift
```

### ChargePredictor

The charge predictor is a neural network that maps node features from the
Allegro backbone to per-atom partial charges:

$$
q_i = \text{ChargePredictor}(\mathbf{h}_i, z_i)
$$

It is a simple MLP with:

- **Input**: node features from Allegro (scalar features aggregated from edge
  features)
- **Hidden**: configurable width (default 64)
- **Output**: per-atom charge
- **Constraint**: predicted charges are not explicitly neutralized
  (electroneutrality emerges from training)

### Training

The model is trained end-to-end with a combined loss:

$$
\mathcal{L} = \mathcal{L}_{\text{energy}} + \lambda \, \mathcal{L}_{\text{charge}}
$$

where:

- $\mathcal{L}_{\text{energy}} = \text{MSE}(E_{\text{pred}}, E_{\text{ref}})$
- $\mathcal{L}_{\text{charge}} = \text{mean}(q_i^2)$ (regularization to
  prevent charge explosion)
- $\lambda = 0.01$ (charge regularization weight)

The Ewald term forces physically meaningful charge distributions: if charges
are too large or unbalanced, the electrostatic energy penalty will be high.

---

## Why It Works: The Key Insight

For water, the **intermolecular interactions** (hydrogen bonds, dipole-dipole)
extend well beyond typical Allegro cutoff radii ($5 \, \text{\AA}$).

Short-range models suffer from a **systematic bias**:

- Molecules within $r_{\max}$: well-described
- Molecules beyond $r_{\max}$: interaction completely ignored

The Ewald correction captures these missing interactions through
electrostatics:

- Partial charges are learned during training
- The electrostatic interaction has the correct $1/r$ decay
- No artificial truncation of long-range physics

This is empirically demonstrated by **Figure 3** (MAE vs intermolecular
distance):

- Short-only model: error increases sharply beyond $r_{\max}$
- Ewald-Allegro: error remains constant across all distances

---

## References

1. P. P. Ewald, "Die Berechnung optischer und elektrostatischer
   Gitterpotentiale," *Annalen der Physik* 369, 253–287 (1921)
2. T. Darden, D. York, L. Pedersen, "Particle mesh Ewald: An N·log(N) method
   for Ewald sums in large systems," *J. Chem. Phys.* 98, 10089–10092 (1993)
3. A. Musaelian et al., "Learning local equivariant representations for
   large-scale atomistic dynamics," *Nat. Commun.* 14, 579 (2023)
