import importlib
import warnings
from typing import Any, ClassVar

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from nepactive.gpumd_msst import GPUMDMSST, linear_ode_step


class HarmonicOriginCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, stiffness=0.02, pressure=0.0):
        super().__init__()
        self.stiffness = stiffness
        self.pressure = pressure

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        self.results["energy"] = 0.5 * self.stiffness * float(
            np.sum(positions * positions)
        )
        self.results["forces"] = -self.stiffness * positions
        self.results["stress"] = np.array(
            [-self.pressure, -self.pressure, -self.pressure, 0.0, 0.0, 0.0]
        )


def make_atoms():
    atoms = Atoms(
        "Ar2",
        positions=[[1.0, 1.5, 2.0], [3.0, 2.5, 2.0]],
        cell=[4.0, 5.0, 6.0],
        pbc=True,
    )
    atoms.set_velocities([[0.0, 1.0e-300, 0.2], [-0.1, 0.3, 0.0]])
    atoms.calc = HarmonicOriginCalculator(stiffness=0.02, pressure=0.04)
    return atoms


def test_public_module_api_is_explicit_and_importable():
    module = importlib.import_module("nepactive.gpumd_msst")

    assert module.GPUMDMSST is GPUMDMSST
    assert module.linear_ode_step is linear_ode_step
    assert module.__all__ == ["GPUMDMSST", "linear_ode_step"]


def test_invalid_runtime_parameters_fail_fast():
    invalid_cases = [
        ({"timestep": 0.0}, "timestep"),
        ({"v_shock": 0.0}, "v_shock"),
        ({"qmass": 0.0}, "qmass"),
        ({"mu": -1.0}, "mu"),
        ({"tscale": 1.0}, "tscale"),
        ({"v0": 0.0}, "v0"),
        ({"p0": np.nan}, "finite"),
    ]
    defaults: dict[str, Any] = {
        "timestep": 0.2 * units.fs,
        "shock_direction": "x",
        "v_shock": 5.0,
        "qmass": 10000.0,
        "mu": 0.0,
    }
    for overrides, message in invalid_cases:
        atoms = make_atoms()
        kwargs: dict[str, Any] = {**defaults, **overrides}
        with np.testing.assert_raises_regex(ValueError, message):
            GPUMDMSST(atoms, **kwargs)

    atoms = make_atoms()
    atoms.set_pbc([True, True, False])
    with np.testing.assert_raises_regex(ValueError, "periodic"):
        GPUMDMSST(atoms, **defaults)


def test_constraints_are_rejected_until_constraint_dynamics_are_supported():
    atoms = make_atoms()
    atoms.set_constraint(FixAtoms(indices=[0]))

    with np.testing.assert_raises_regex(ValueError, "constraint"):
        GPUMDMSST(
            atoms,
            timestep=0.2 * units.fs,
            shock_direction="x",
            v_shock=5.0,
            qmass=10000.0,
            mu=0.0,
        )


def test_nonfinite_initial_cell_positions_or_velocities_fail_fast():
    defaults: dict[str, Any] = {
        "timestep": 0.2 * units.fs,
        "shock_direction": "x",
        "v_shock": 5.0,
        "qmass": 10000.0,
        "mu": 0.0,
        "p0": 0.0,
        "v0": 120.0,
        "e0": 0.0,
    }

    atoms = make_atoms()
    positions = atoms.get_positions()
    positions[0, 0] = np.inf
    atoms.set_positions(positions)
    with np.testing.assert_raises_regex(ValueError, "finite.*position|position.*finite"):
        GPUMDMSST(atoms, **defaults)

    atoms = make_atoms()
    velocities = atoms.get_velocities()
    velocities[0, 0] = np.inf
    atoms.set_velocities(velocities)
    with np.testing.assert_raises_regex(ValueError, "finite.*veloc|veloc.*finite"):
        GPUMDMSST(atoms, **defaults)

    atoms = make_atoms()
    cell = atoms.cell.array.copy()
    cell[0, 0] = np.nan
    atoms.set_cell(cell)
    with np.testing.assert_raises_regex(ValueError, "finite.*cell|cell.*finite"):
        GPUMDMSST(atoms, **defaults)


def test_step_requests_ase_md_forces_for_both_force_evaluations():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.05 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=1.0e8,
        mu=0.0,
    )
    original_get_forces = atoms.get_forces
    md_arguments = []

    def get_forces(apply_constraint=True, md=False):
        md_arguments.append(md)
        return original_get_forces(apply_constraint=apply_constraint, md=md)

    atoms.get_forces = get_forces
    dynamics.step()

    assert md_arguments == [True, True]


def test_linear_ode_step_matches_independent_scalar_solution():
    value = np.array([0.0, 1.25, -0.4])
    constant = np.array([2.0, -0.3, 0.8])
    linear = np.array([0.7, -0.2, 0.05])
    dt = 0.13

    expected = np.exp(linear * dt) * value + constant * (
        np.expm1(linear * dt) / linear
    )

    with np.errstate(divide="raise", over="raise", invalid="raise"):
        actual = linear_ode_step(value, constant, linear, dt)

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_linear_ode_step_has_continuous_constant_acceleration_limit():
    value = np.array([0.0, 1.0e-300, -2.0])
    acceleration = np.array([3.0, -4.0, 0.25])
    dt = 0.4

    with np.errstate(divide="raise", over="raise", invalid="raise"):
        at_zero = linear_ode_step(value, acceleration, np.zeros(3), dt)
        near_zero = linear_ode_step(
            value, acceleration, np.full(3, 1.0e-14), dt
        )

    expected = value + acceleration * dt
    np.testing.assert_allclose(at_zero, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(near_zero, expected, rtol=2e-14, atol=2e-14)


def test_linear_ode_step_fails_cleanly_on_nonfinite_or_exponential_overflow():
    with np.testing.assert_raises_regex(FloatingPointError, "finite|overflow"):
        linear_ode_step(1.0, 1.0, np.inf, 0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.testing.assert_raises_regex(FloatingPointError, "overflow|finite"):
            linear_ode_step(1.0, 1.0, 1.0e6, 1.0)


def test_shock_direction_must_be_a_cartesian_axis():
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.set_velocities([[0.0, 0.0, 0.0]])

    with np.testing.assert_raises_regex(ValueError, "shock_direction"):
        GPUMDMSST(
            atoms,
            timestep=0.1,
            shock_direction="xy",
            v_shock=5.0,
            qmass=10000.0,
            mu=0.0,
        )


def test_shock_cell_vector_must_be_cartesian_aligned():
    atoms = Atoms(
        "Ar",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.5, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    atoms.set_velocities([[0.0, 0.0, 0.0]])

    with np.testing.assert_raises_regex(ValueError, "aligned"):
        GPUMDMSST(
            atoms,
            timestep=0.1,
            shock_direction="x",
            v_shock=5.0,
            qmass=10000.0,
            mu=0.0,
        )


def test_transverse_shear_is_allowed_when_shock_vector_is_axis_aligned():
    atoms = Atoms(
        "Ar",
        positions=[[1.0, 1.0, 1.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [-1.5, 0.0, 7.0]],
        pbc=True,
    )
    atoms.set_velocities([[0.1, 0.0, 0.0]])
    atoms.calc = HarmonicOriginCalculator()

    dynamics = GPUMDMSST(
        atoms,
        timestep=0.1 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
    )
    cell0 = atoms.cell.array.copy()
    dynamics._remap_axis(0.8)

    expected = cell0.copy()
    expected[0] *= 0.8
    np.testing.assert_allclose(atoms.cell.array, expected)
    np.testing.assert_allclose(atoms.cell.array[2], cell0[2])


def test_initial_state_defaults_are_taken_from_ase_state():
    atoms = make_atoms()
    expected_v0 = atoms.get_volume()
    expected_e0 = atoms.get_total_energy()
    expected_p0 = -atoms.get_stress(
        voigt=False, include_ideal_gas=True
    )[0, 0]

    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
    )

    assert dynamics.v0 == expected_v0
    assert dynamics.e0 == expected_e0
    assert dynamics.p0 == expected_p0


def test_explicit_reference_state_overrides_ase_defaults():
    atoms = make_atoms()

    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="z",
        v_shock=6.0,
        qmass=12000.0,
        mu=2.0,
        v0=321.0,
        e0=-17.5,
        p0=0.125,
    )

    assert dynamics.v0 == 321.0
    assert dynamics.e0 == -17.5
    assert dynamics.p0 == 0.125


def test_tscale_matches_gpumd_startup_scaling_and_reports_energy_offset():
    atoms = make_atoms()
    velocities = atoms.get_velocities().copy()
    kinetic_energy = atoms.get_kinetic_energy()
    total_mass = atoms.get_masses().sum()
    qmass = 10000.0
    tscale = 0.25

    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="y",
        v_shock=5.0,
        qmass=qmass,
        mu=0.0,
        tscale=tscale,
    )

    np.testing.assert_allclose(
        atoms.get_velocities(), velocities * np.sqrt(1.0 - tscale)
    )
    assert dynamics.omega == -np.sqrt(
        tscale * total_mass * kinetic_energy / qmass
    )
    cell_kinetic = qmass * dynamics.omega**2 / (2.0 * total_mass)
    np.testing.assert_allclose(cell_kinetic, 0.5 * tscale * kinetic_energy)
    np.testing.assert_allclose(
        dynamics.tscale_energy_offset_eV, -0.5 * tscale * kinetic_energy
    )


def test_remap_changes_only_shock_axis_cell_position_and_velocity():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="y",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
    )
    cell0 = atoms.cell.array.copy()
    positions0 = atoms.get_positions().copy()
    velocities0 = atoms.get_velocities().copy()

    dynamics._remap_axis(0.8)

    expected_cell = cell0.copy()
    expected_cell[1] *= 0.8
    expected_positions = positions0.copy()
    expected_positions[:, 1] *= 0.8
    expected_velocities = velocities0.copy()
    expected_velocities[:, 1] *= 0.8
    np.testing.assert_allclose(atoms.cell.array, expected_cell)
    np.testing.assert_allclose(atoms.get_positions(), expected_positions)
    np.testing.assert_allclose(atoms.get_velocities(), expected_velocities)


def test_velocity_half_step_is_finite_for_zero_and_tiny_components():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=3.0,
    )
    dynamics.omega = -0.2
    before = atoms.get_velocities().copy()
    forces = np.array([[0.3, -0.2, 0.1], [-0.1, 0.4, -0.3]])
    half_dt = 0.1 * units.fs
    masses = atoms.get_masses()[:, None]
    speed_sum = float(np.sum(before * before))
    rates = np.full_like(before, dynamics.omega**2 * dynamics.mu)
    rates /= speed_sum * masses * atoms.get_volume()
    rates[:, 0] -= 2.0 * dynamics.omega / atoms.get_volume()
    expected = linear_ode_step(before, forces / masses, rates, half_dt)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            dynamics._velocity_step(forces, half_dt, speed_sum=speed_sum)

    np.testing.assert_allclose(atoms.get_velocities(), expected)
    assert np.isfinite(atoms.get_velocities()).all()


def test_omega_half_step_matches_linear_ode_and_zero_viscosity_limit():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=3.0,
    )
    dynamics.omega = -0.2
    half_dt = 0.1 * units.fs
    volume = atoms.get_volume()
    pressure = dynamics._longitudinal_pressure()
    rayleigh = pressure - dynamics.p0 - (
        dynamics.total_mass
        * dynamics.v_shock**2
        / dynamics.v0
        * (1.0 - volume / dynamics.v0)
    )
    constant = dynamics.total_mass * rayleigh / dynamics.qmass
    damping = dynamics.mu * dynamics.total_mass / (dynamics.qmass * volume)
    expected = float(linear_ode_step(dynamics.omega, constant, -damping, half_dt))

    with np.errstate(all="raise"):
        dynamics._omega_step(half_dt)
    assert dynamics.omega == expected

    dynamics.mu = 0.0
    before = dynamics.omega
    pressure = dynamics._longitudinal_pressure()
    rayleigh = pressure - dynamics.p0 - (
        dynamics.total_mass
        * dynamics.v_shock**2
        / dynamics.v0
        * (1.0 - atoms.get_volume() / dynamics.v0)
    )
    constant = dynamics.total_mass * rayleigh / dynamics.qmass
    with np.errstate(all="raise"):
        dynamics._omega_step(half_dt)
    np.testing.assert_allclose(dynamics.omega, before + half_dt * constant)


def test_omega_step_reverses_expansion_side_positive_drive():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
        v0=100.0,
    )
    assert atoms.get_volume() > dynamics.v0
    dynamics.omega = 0.0
    dt = 0.1 * units.fs
    pressure = dynamics._longitudinal_pressure()
    drive = dynamics.total_mass / dynamics.qmass * (
        pressure
        - dynamics.p0
        - dynamics.total_mass
        * dynamics.v_shock**2
        / dynamics.v0
        * (1.0 - atoms.get_volume() / dynamics.v0)
    )
    assert drive > 0.0

    dynamics._omega_step(dt)

    np.testing.assert_allclose(dynamics.omega, -drive * dt)


def test_each_atomic_mass_must_be_finite_and_positive():
    atoms = make_atoms()
    atoms.set_masses([-1.0, 40.0])

    with np.testing.assert_raises_regex(ValueError, "mass"):
        GPUMDMSST(
            atoms,
            timestep=0.2 * units.fs,
            shock_direction="x",
            v_shock=5.0,
            qmass=10000.0,
            mu=0.0,
        )


def test_zero_speed_norm_uses_removable_limit_or_fails_clearly():
    atoms = make_atoms()
    atoms.set_velocities(np.zeros((2, 3)))
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
    )
    dynamics.omega = -0.2
    forces = np.array([[0.3, -0.2, 0.1], [-0.1, 0.4, -0.3]])
    half_dt = 0.1 * units.fs
    rates = np.zeros((2, 3))
    rates[:, 0] = -2.0 * dynamics.omega / atoms.get_volume()
    expected = linear_ode_step(
        np.zeros((2, 3)), forces / atoms.get_masses()[:, None], rates, half_dt
    )

    with np.errstate(all="raise"):
        dynamics._velocity_step(forces, half_dt, speed_sum=0.0)
    np.testing.assert_allclose(atoms.get_velocities(), expected)

    atoms.set_velocities(np.zeros((2, 3)))
    dynamics.mu = 1.0
    with np.testing.assert_raises_regex(FloatingPointError, "speed|kinetic"):
        dynamics._velocity_step(forces, half_dt, speed_sum=0.0)


def test_step_uses_documented_split_order_and_refreshes_forces():
    atoms = make_atoms()
    events = []
    original_calculate = atoms.calc.calculate

    def calculate(*args, **kwargs):
        events.append("force_refresh")
        return original_calculate(*args, **kwargs)

    atoms.calc.calculate = calculate
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
    )
    events.clear()

    dynamics._update_diagnostics = lambda: events.append("diagnostics")
    dynamics._omega_step = lambda dt: events.append("omega")
    dynamics._velocity_step = lambda forces, dt, speed_sum: events.append("velocity")
    dynamics._volume_step = lambda dt: events.append("volume")

    def position_step(dt):
        events.append("position")
        atoms.set_positions(atoms.get_positions() + 0.01)

    dynamics._position_step = position_step
    dynamics._refresh_thermo = lambda: events.append("thermo")

    returned_forces = dynamics.step(forces=np.zeros((2, 3)))

    assert events == [
        "omega",
        "velocity",
        "velocity",
        "volume",
        "position",
        "volume",
        "force_refresh",
        "velocity",
        "thermo",
        "omega",
        "diagnostics",
    ]
    assert returned_forces.shape == (2, 3)


def test_step_reuses_trial_speed_sum_for_both_physical_velocity_kicks():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=1.0,
    )
    base = atoms.get_velocities().copy()
    speed_sum_arguments = []

    dynamics._update_diagnostics = lambda: None
    dynamics._omega_step = lambda dt: None
    dynamics._volume_step = lambda dt: None
    dynamics._position_step = lambda dt: None
    dynamics._refresh_thermo = lambda: None

    def velocity_step(forces, dt, speed_sum):
        speed_sum_arguments.append(speed_sum)
        increment = float(len(speed_sum_arguments))
        atoms.set_velocities(base + increment)

    dynamics._velocity_step = velocity_step
    dynamics.step(forces=np.zeros((2, 3)))

    original_speed_sum = float(np.sum(base * base))
    predicted_speed_sum = float(np.sum((base + 1.0) ** 2))
    assert speed_sum_arguments == [
        original_speed_sum,
        predicted_speed_sum,
        predicted_speed_sum,
    ]


def test_volume_step_advances_dvdt_and_rejects_nonpositive_volume():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="z",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
    )
    volume0 = atoms.get_volume()
    cell0 = atoms.cell.array.copy()
    dt = 0.1 * units.fs
    dynamics.omega = -0.5

    dynamics._volume_step(dt)

    expected_volume = volume0 + dynamics.omega * dt
    np.testing.assert_allclose(atoms.get_volume(), expected_volume)
    np.testing.assert_allclose(atoms.cell.array[:2], cell0[:2])

    dynamics.omega = -2.0 * atoms.get_volume() / dt
    with np.testing.assert_raises_regex(FloatingPointError, "volume"):
        dynamics._volume_step(dt)


def test_position_step_is_cartesian_drift_only():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="y",
        v_shock=5.0,
        qmass=10000.0,
        mu=0.0,
    )
    positions = atoms.get_positions().copy()
    velocities = atoms.get_velocities().copy()
    cell = atoms.cell.array.copy()
    dt = 0.2 * units.fs

    dynamics._position_step(dt)

    np.testing.assert_allclose(atoms.get_positions(), positions + velocities * dt)
    np.testing.assert_allclose(atoms.get_velocities(), velocities)
    np.testing.assert_allclose(atoms.cell.array, cell)


def test_diagnostics_match_hugoniot_rayleigh_and_extended_energy_formulas():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.2 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=10000.0,
        mu=2.0,
        p0=0.015,
        v0=130.0,
        e0=0.25,
    )
    dynamics.omega = -0.3
    volume = atoms.get_volume()
    pressure = dynamics._longitudinal_pressure()
    energy = atoms.get_total_energy()
    compression = 1.0 - volume / dynamics.v0
    expected_hugoniot = (
        0.5 * (pressure + dynamics.p0) * (dynamics.v0 - volume)
        + dynamics.e0
        - energy
    )
    expected_rayleigh = (
        pressure
        - dynamics.p0
        - dynamics.total_mass * dynamics.v_shock**2 / dynamics.v0 * compression
    )
    expected_conserved = (
        energy
        + dynamics.qmass * dynamics.omega**2 / (2.0 * dynamics.total_mass)
        + dynamics.p0 * (volume - dynamics.v0)
        - 0.5 * dynamics.total_mass * dynamics.v_shock**2 * compression**2
    )

    np.testing.assert_allclose(dynamics.dHugoniot, expected_hugoniot)
    np.testing.assert_allclose(dynamics.dRayleigh, expected_rayleigh)
    np.testing.assert_allclose(
        dynamics.extended_conserved_energy, expected_conserved
    )
    diagnostics = dynamics.diagnostics
    np.testing.assert_allclose(diagnostics["dHugoniot_eV"], expected_hugoniot)
    np.testing.assert_allclose(diagnostics["dRayleigh_eV_A3"], expected_rayleigh)
    np.testing.assert_allclose(
        diagnostics["dRayleigh_GPa"], expected_rayleigh / units.GPa
    )
    np.testing.assert_allclose(
        diagnostics["extended_conserved_eV"], expected_conserved
    )
    assert np.isfinite(list(diagnostics.values())).all()


def test_real_step_updates_finite_thermo_without_printing(capsys):
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.05 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=1.0e6,
        mu=0.0,
    )

    forces = dynamics.step()

    assert capsys.readouterr().out == ""
    assert np.isfinite(forces).all()
    assert np.isfinite(atoms.get_positions()).all()
    assert np.isfinite(atoms.get_velocities()).all()
    assert np.isfinite(atoms.cell.array).all()
    assert atoms.get_volume() > 0.0
    assert np.isfinite(dynamics.vsum)
    assert np.isfinite(dynamics.temperature)
    assert np.isfinite(dynamics.pressure)
    assert np.isfinite(list(dynamics.last_diagnostics.values())).all()


def test_completed_step_advances_lagrangian_position_and_reports_it():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.05 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=1.0e8,
        mu=0.0,
    )

    dynamics.step()

    expected = -(
        dynamics.v_shock
        * atoms.get_volume()
        / dynamics.v0
        * dynamics.dt
    )
    np.testing.assert_allclose(dynamics.lagrangian_position, expected)
    np.testing.assert_allclose(
        dynamics.diagnostics["lagrangian_position_A"], expected
    )


def test_last_diagnostics_exists_initially_and_tracks_completed_state():
    atoms = make_atoms()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.05 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=1.0e8,
        mu=0.0,
    )

    initial = dynamics.diagnostics
    assert dynamics.last_diagnostics == initial

    dynamics.step()

    current = dynamics.diagnostics
    assert dynamics.last_diagnostics.keys() == current.keys()
    for key in current:
        np.testing.assert_allclose(dynamics.last_diagnostics[key], current[key])
    assert dynamics.last_diagnostics["lagrangian_position_A"] != initial[
        "lagrangian_position_A"
    ]


def test_analytic_system_stays_finite_for_1000_steps_without_output(capsys):
    atoms = make_atoms()
    atoms.set_velocities([[0.02, 0.01, 0.005], [-0.02, -0.01, -0.005]])
    transverse_cell0 = atoms.cell.array[1:].copy()
    dynamics = GPUMDMSST(
        atoms,
        timestep=0.01 * units.fs,
        shock_direction="x",
        v_shock=5.0,
        qmass=1.0e12,
        mu=1.0e-4,
        tscale=0.01,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(divide="raise", over="raise", invalid="raise"):
            dynamics.run(1000)

    assert capsys.readouterr().out == ""
    assert dynamics.nsteps == 1000
    assert np.isfinite(atoms.get_positions()).all()
    assert np.isfinite(atoms.get_velocities()).all()
    assert np.isfinite(atoms.cell.array).all()
    assert atoms.get_volume() > 0.0
    np.testing.assert_allclose(atoms.cell.array[1:], transverse_cell0)
    assert np.isfinite(list(dynamics.diagnostics.values())).all()
