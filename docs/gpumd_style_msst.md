# GPUMD-style MSST integrator for ASE

## Status and provenance

`nepactive.gpumd_msst.GPUMDMSST` is an independent implementation of the
single-axis MSST equations for ASE. Its behavior was designed from public MSST
equations and documented algorithmic behavior. No GPUMD source code was copied,
translated line by line, or incorporated into this module.

GPUMD source files declare `GPL-3.0-or-later`, while BRIDGE uses the BRIDGE
Academic and Non-Commercial License. This implementation intentionally remains
structurally independent and makes no legal compatibility claim.
The existing `MSST.py` and `omdMSST.py` are not modified or replaced.

This is an experimental integrator. CPU analytic-calculator tests establish
finite numerical behavior and regression coverage; they do **not** establish
production scientific equivalence with GPUMD or validate MatterSim shock
trajectories.

## Public API

```python
from ase import units
from nepactive.gpumd_msst import GPUMDMSST

dyn = GPUMDMSST(
    atoms,
    timestep=0.4 * units.fs,
    shock_direction="x",
    v_shock=9.556,  # km/s
    qmass=7.0e7,
    mu=10.0,
    tscale=0.1,
    p0=None,
    v0=None,
    e0=None,
)
dyn.run(steps)
print(dyn.diagnostics)
```

`p0`, `v0`, and `e0` default to the initial ASE state when omitted. The attached
ASE calculator must implement energy, forces, and stress because construction
immediately evaluates `get_stress(..., include_ideal_gas=True)`. Pressure is
positive in compression and includes the ideal-gas kinetic contribution.
Cell, position, velocity, mass, and initial thermodynamic values are checked for
finiteness at construction. ASE constraints are rejected explicitly in this
first version.

The `tscale` startup follows the documented GPUMD-style convention exactly:
atomic velocities are multiplied by `sqrt(1-tscale)` and
`omega=-sqrt(tscale*M*K0/Q)`. Consequently, the cell kinetic term receives
`0.5*tscale*K0`, while atomic kinetic energy loses `tscale*K0`. The resulting
extended-energy offset is `-0.5*tscale*K0`; it is reported explicitly as
`tscale_energy_offset_eV` and must not be described as exact energy transfer.

## Units

The implementation uses ASE internal units:

| quantity | API / stored unit |
|---|---|
| timestep | ASE time, normally `value * ase.units.fs` |
| shock velocity input | km/s, converted to ASE velocity |
| position / cell | Å |
| volume | Å³ |
| energy | eV |
| mass | amu |
| pressure and stress | eV/Å³; diagnostics also report GPa |
| `omega = dV/dt` | Å³ / ASE-time |
| `qmass` | effective ASE MSST unit equivalent to amu²/Å⁴ |
| `mu` | effective ASE viscosity unit equivalent to `sqrt(amu*eV)/Å²` |

`qmass` and `mu` are effective integrator parameters. Numerical equality with a
GPUMD input requires an explicit unit-mapping validation; it is not assumed by
the current tests.

## State and equations

For total mass `M`, current volume `V`, reference state `(V0, E0, P0)`, shock
speed `vs`, longitudinal pressure `P`, and `omega = dV/dt`, define

```text
p_msst = M * vs² * (V0 - V) / V0²
A      = M * (P - P0 - p_msst) / Q
B      = M * mu / (Q * V)
domega/dt = A - B * omega
```

The fixed-coefficient omega substep is evaluated with the analytic linear-ODE
solution and a Taylor/`expm1` zero limit.
As a GPUMD-style expansion safeguard, if `V>V0` while `A>0`, the sign of `A` is
reversed so that the cell is not accelerated farther into expansion.

For atom `i`, Cartesian component `a`, force `F`, mass `m`, and
`S = sum_i |v_i|²`, the velocity substep is

```text
C[i,a] = F[i,a] / m[i]
D[i,a] = omega² * mu / (S * m[i] * V)
D[i,shock_axis] -= 2 * omega / V
dv[i,a]/dt = C[i,a] + D[i,a] * v[i,a]
```

The exact fixed-coefficient update uses

```text
v_new = exp(z) * v + dt * phi1(z) * C
z = D * dt
phi1(z) = expm1(z) / z
phi1(0) = 1
```

This avoids the old `L[j] / qdot[i,j]` decomposition: zero or tiny individual
velocity components are not denominators. If the *entire* speed norm is zero
while the viscosity coupling is nonzero, the equation itself is undefined and
the implementation fails explicitly rather than inventing an epsilon.
The trial first-half velocity update supplies one predicted `S`; that same
predicted speed sum is retained for both physical velocity kicks until the
post-kick thermodynamic refresh, matching the intended split semantics.

## Split integration order

Each `step()` performs:

1. half-step `omega` at fixed atomic state;
2. trial velocity half-step to estimate the updated speed sum;
3. restore velocities and perform the physical velocity half-step with that
   predicted speed sum;
4. half-step volume update and axial remap;
5. full Cartesian position drift;
6. second half-step volume update and axial remap;
7. refresh MD forces from the ASE calculator using `md=True`;
8. second velocity half-step using the retained predicted speed sum;
9. refresh thermodynamic scalars and the actual post-kick speed sum;
10. second half-step `omega`;
11. advance the Lagrangian position and store a completed-state diagnostic
    snapshot for ASE observers.

Only the shock-axis cell vector, Cartesian coordinates, and velocity components
are dilated during remap. The two transverse cell vectors remain unchanged.

## Diagnostics

The diagnostic properties use the following sign convention, matching the
publicly documented GPUMD-style residual orientation:

```text
dHugoniot = 0.5 * (P + P0) * (V0 - V) + E0 - E

dRayleigh = P - P0 - M * vs² * (1 - V/V0) / V0

E_extended = E
           + Q * omega² / (2M)
           + P0 * (V - V0)
           - 0.5 * M * vs² * (1 - V/V0)²
```

`diagnostics` reports volume, pressure, temperature, omega, Lagrangian position,
the tscale startup offset, both residuals, and the extended conserved-energy
diagnostic. `last_diagnostics` exists immediately after construction and is
refreshed after each completed step, so ASE observers do not receive a stale
pre-step snapshot. Non-finite diagnostics raise `FloatingPointError`.

## First-version limitations

- Only fully periodic cells are accepted.
- ASE constraints are rejected; constrained remap/RATTLE dynamics are not yet
  implemented.
- The shock cell vector must be Cartesian-aligned. Transverse lattice vectors
  may retain shear components along that Cartesian axis, matching the accepted
  GPUMD-style box form; a shock vector with off-axis components fails fast.
- There is no MPI/GPU implementation.
- There is no restart-state serialization yet.
- No automatic center-of-mass momentum removal is performed each step.
- The implementation has not yet been compared step-by-step against GPUMD under
  a unit-matched physical fixture.
- The analytic 1000-step test is a numerical regression test, not a scientific
  validation of a shocked material.

Before use in T37 or another production campaign, run an isolated GPU smoke with
MatterSim, compare a controlled short trajectory against GPUMD using identical
structure/seed/units, audit `V/V0`, pressure, temperature, both residuals, and
conserved-energy drift, and preserve the old failed trajectories separately.
