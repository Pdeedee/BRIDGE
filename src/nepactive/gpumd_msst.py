"""ASE MSST integrator with independently implemented GPUMD-style behavior.

The algorithm behavior in this module is independently implemented from public
MSST equations and documentation; no GPUMD source code was copied.
"""

from __future__ import annotations

import numpy as np
from ase import units
from ase.md.md import MolecularDynamics

__all__ = ["GPUMDMSST", "linear_ode_step"]


class GPUMDMSST(MolecularDynamics):
    """GPUMD-style single-axis MSST dynamics for ASE atoms.

    Parameters use explicit units: ``timestep`` is in ASE internal time units
    (normally supplied as ``value * ase.units.fs``), ``v_shock`` is in km/s,
    ``qmass`` in amu**2/Angstrom**4, ``mu`` in
    sqrt(amu*eV)/Angstrom**2, ``p0`` in eV/Angstrom**3, ``v0`` in
    Angstrom**3, and ``e0`` in eV.  Positive pressure means compression.

    This first implementation requires the shock cell vector to be aligned with
    its selected Cartesian axis. The two transverse cell vectors may retain
    shear components along that axis. ASE constraints are rejected because this
    integrator does not implement constrained position/velocity remapping.
    """

    def __init__(
        self,
        atoms,
        timestep,
        shock_direction,
        v_shock,
        qmass,
        mu,
        tscale=0.0,
        p0=None,
        v0=None,
        e0=None,
        trajectory=None,
        logfile=None,
        loginterval=1,
        append_trajectory=False,
    ):
        if shock_direction not in {"x", "y", "z"}:
            raise ValueError("shock_direction must be 'x', 'y', or 'z'")
        axis = "xyz".index(shock_direction)
        cell = np.asarray(atoms.cell.array, dtype=float)
        if not np.isfinite(cell).all():
            raise ValueError("initial cell must contain only finite values")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError("initial positions must contain only finite values")
        velocities = atoms.get_velocities()
        if velocities is None:
            atoms.set_velocities(np.zeros((len(atoms), 3), dtype=float))
        elif not np.isfinite(np.asarray(velocities, dtype=float)).all():
            raise ValueError("initial velocities must contain only finite values")
        scale = max(1.0, float(np.max(np.abs(cell))))
        tolerance = 1.0e-12 * scale
        shock_off_axis = np.delete(cell[axis], axis)
        if np.any(np.abs(shock_off_axis) > tolerance):
            raise ValueError("shock cell vector must be Cartesian-aligned")
        if not np.all(atoms.get_pbc()):
            raise ValueError("MSST requires periodic boundary conditions in all axes")
        if atoms.constraints:
            raise ValueError(
                "ASE constraints are not supported by GPUMDMSST; "
                "remove constraints before constructing the dynamics"
            )
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("timestep must be finite and positive")
        if not np.isfinite(v_shock) or v_shock <= 0.0:
            raise ValueError("v_shock must be finite and positive in km/s")
        if not np.isfinite(qmass) or qmass <= 0.0:
            raise ValueError("qmass must be finite and positive")
        if not np.isfinite(mu) or mu < 0.0:
            raise ValueError("mu must be finite and non-negative")
        if not np.isfinite(tscale) or not 0.0 <= tscale < 1.0:
            raise ValueError("tscale must satisfy 0 <= tscale < 1")
        if atoms.get_volume() <= 0.0:
            raise ValueError("cell volume must be positive")

        super().__init__(
            atoms,
            timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
        self.axis = axis
        self.shock_direction = shock_direction
        self.v_shock_km_s = float(v_shock)
        self.v_shock = float(v_shock) * 1000.0 * units.m / units.second
        self.qmass = float(qmass)
        self.mu = float(mu)
        self.tscale = float(tscale)
        masses = np.asarray(atoms.get_masses(), dtype=float)
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("each atomic mass must be finite and positive")
        self.total_mass = float(np.sum(masses))
        if not np.isfinite(self.total_mass) or self.total_mass <= 0.0:
            raise ValueError("total atomic mass must be finite and positive")

        initial_volume = float(atoms.get_volume())
        initial_energy = float(atoms.get_total_energy())
        initial_pressure = self._longitudinal_pressure()
        initial_kinetic_energy = float(atoms.get_kinetic_energy())
        if not np.isfinite(
            [initial_volume, initial_energy, initial_pressure, initial_kinetic_energy]
        ).all():
            raise ValueError("initial thermodynamic state must be finite")
        self.v0 = initial_volume if v0 is None else float(v0)
        self.e0 = initial_energy if e0 is None else float(e0)
        self.p0 = initial_pressure if p0 is None else float(p0)
        if not np.isfinite([self.v0, self.e0, self.p0]).all():
            raise ValueError("p0, v0, and e0 must be finite")
        if self.v0 <= 0.0:
            raise ValueError("v0 must be positive")
        self.omega = -np.sqrt(
            self.tscale * self.total_mass * initial_kinetic_energy / self.qmass
        )
        if not np.isfinite(self.omega):
            raise ValueError("initial MSST volume rate must be finite")
        # The documented GPUMD-style startup scales atomic kinetic energy by
        # (1-tscale), while the cell kinetic term receives half of the removed
        # amount under this omega convention.  Record the resulting offset
        # explicitly instead of describing it as exact energy transfer.
        self.tscale_energy_offset_eV = (
            -0.5 * self.tscale * initial_kinetic_energy
        )
        if self.tscale:
            atoms.set_velocities(
                atoms.get_velocities() * np.sqrt(1.0 - self.tscale)
            )
        self.lagrangian_position = 0.0
        self._refresh_thermo()
        self._update_diagnostics()

    def _longitudinal_pressure(self):
        """Return positive-compression longitudinal pressure in eV/A^3."""
        # Squaring physically negligible subnormal momenta inside ASE's ideal-gas
        # stress can underflow.  Keep all scientifically dangerous FP errors live.
        with np.errstate(under="ignore"):
            stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        return float(-stress[self.axis, self.axis])

    @property
    def dHugoniot(self):
        """Energy-form Hugoniot residual in eV."""
        volume = self.atoms.get_volume()
        pressure = self._longitudinal_pressure()
        return float(
            0.5 * (pressure + self.p0) * (self.v0 - volume)
            + self.e0
            - self.atoms.get_total_energy()
        )

    @property
    def dRayleigh(self):
        """Longitudinal Rayleigh-line residual in eV/Angstrom^3."""
        volume = self.atoms.get_volume()
        compression = 1.0 - volume / self.v0
        return float(
            self._longitudinal_pressure()
            - self.p0
            - self.total_mass * self.v_shock**2 / self.v0 * compression
        )

    @property
    def extended_conserved_energy(self):
        """MSST extended conserved-energy diagnostic in eV."""
        volume = self.atoms.get_volume()
        compression = 1.0 - volume / self.v0
        return float(
            self.atoms.get_total_energy()
            + self.qmass * self.omega**2 / (2.0 * self.total_mass)
            + self.p0 * (volume - self.v0)
            - 0.5 * self.total_mass * self.v_shock**2 * compression**2
        )

    @property
    def diagnostics(self):
        """Return a finite ASE-unit diagnostic snapshot without printing."""
        volume = float(self.atoms.get_volume())
        pressure = self._longitudinal_pressure()
        values = {
            "volume_A3": volume,
            "pressure_eV_A3": pressure,
            "pressure_GPa": pressure / units.GPa,
            "temperature_K": float(self.atoms.get_temperature()),
            "omega_A3_fs": float(self.omega * units.fs),
            "lagrangian_position_A": float(self.lagrangian_position),
            "tscale_energy_offset_eV": float(self.tscale_energy_offset_eV),
            "dHugoniot_eV": self.dHugoniot,
            "dRayleigh_eV_A3": self.dRayleigh,
            "dRayleigh_GPa": self.dRayleigh / units.GPa,
            "extended_conserved_eV": self.extended_conserved_energy,
        }
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError("MSST diagnostics contain non-finite values")
        return values

    def _remap_axis(self, dilation):
        """Apply a homogeneous remap along the supported Cartesian axis."""
        if not np.isfinite(dilation) or dilation <= 0.0:
            raise FloatingPointError("MSST remap requires a positive finite dilation")
        cell = self.atoms.cell.array.copy()
        positions = self.atoms.get_positions().copy()
        velocities = self.atoms.get_velocities().copy()
        cell[self.axis] *= dilation
        positions[:, self.axis] *= dilation
        velocities[:, self.axis] *= dilation
        self.atoms.set_cell(cell, scale_atoms=False)
        self.atoms.set_positions(positions)
        self.atoms.set_velocities(velocities)

    def _velocity_step(self, forces, dt, speed_sum):
        """Advance particle velocities for one fixed-coefficient substep."""
        velocities = self.atoms.get_velocities()
        masses = self.atoms.get_masses()[:, None]
        coupling = self.omega**2 * self.mu
        if not np.isfinite(speed_sum) or speed_sum < 0.0:
            raise FloatingPointError("particle speed sum must be finite and non-negative")
        if speed_sum == 0.0:
            if coupling != 0.0:
                raise FloatingPointError(
                    "nonzero viscosity coupling is undefined at zero kinetic speed"
                )
            rates = np.zeros_like(velocities)
        else:
            rates = np.full_like(velocities, coupling)
            rates /= speed_sum * masses * self.atoms.get_volume()
        rates[:, self.axis] -= 2.0 * self.omega / self.atoms.get_volume()
        updated = linear_ode_step(
            velocities, np.asarray(forces) / masses, rates, dt
        )
        if not np.isfinite(updated).all():
            raise FloatingPointError("particle velocity step produced non-finite values")
        self.atoms.set_velocities(updated)

    def _omega_step(self, dt):
        """Advance the cell volume rate for one fixed-state substep."""
        volume = self.atoms.get_volume()
        pressure = self._longitudinal_pressure()
        rayleigh = pressure - self.p0 - (
            self.total_mass
            * self.v_shock**2
            / self.v0
            * (1.0 - volume / self.v0)
        )
        constant = self.total_mass * rayleigh / self.qmass
        if volume > self.v0 and constant > 0.0:
            constant = -constant
        damping = self.mu * self.total_mass / (self.qmass * volume)
        self.omega = float(linear_ode_step(self.omega, constant, -damping, dt))
        if not np.isfinite(self.omega):
            raise FloatingPointError("cell volume-rate step produced a non-finite value")

    def _volume_step(self, dt):
        """Advance volume by ``omega * dt`` and remap the shock axis."""
        volume = float(self.atoms.get_volume())
        new_volume = volume + self.omega * dt
        if not np.isfinite(new_volume) or new_volume <= 0.0:
            raise FloatingPointError("MSST volume step requires positive finite volume")
        self._remap_axis(new_volume / volume)

    def _position_step(self, dt):
        """Drift Cartesian positions at the current remapped velocities."""
        positions = self.atoms.get_positions() + self.atoms.get_velocities() * dt
        if not np.isfinite(positions).all():
            raise FloatingPointError("particle position step produced non-finite values")
        self.atoms.set_positions(positions)

    def _update_diagnostics(self):
        """Store a step-boundary diagnostic snapshot for ASE observers."""
        self.last_diagnostics = self.diagnostics

    def _refresh_thermo(self):
        """Refresh finite thermodynamic scalars after the second velocity kick."""
        velocities = self.atoms.get_velocities()
        self.vsum = float(np.sum(velocities * velocities))
        self.temperature = float(self.atoms.get_temperature())
        self.pressure = self._longitudinal_pressure()
        if not np.isfinite([self.vsum, self.temperature, self.pressure]).all():
            raise FloatingPointError("MSST thermodynamic state contains non-finite values")

    def step(self, forces=None):
        """Advance one MSST timestep and return the refreshed forces."""
        if forces is None:
            forces = self.atoms.get_forces(md=True)
        half_dt = 0.5 * self.dt

        self._omega_step(half_dt)

        velocities = self.atoms.get_velocities().copy()
        speed_sum = float(np.sum(velocities * velocities))
        self._velocity_step(forces, half_dt, speed_sum=speed_sum)
        predicted_speed_sum = float(
            np.sum(self.atoms.get_velocities() * self.atoms.get_velocities())
        )
        self.atoms.set_velocities(velocities)
        self._velocity_step(forces, half_dt, speed_sum=predicted_speed_sum)

        self._volume_step(half_dt)
        self._position_step(self.dt)
        self._volume_step(half_dt)

        forces = self.atoms.get_forces(md=True)
        self._velocity_step(forces, half_dt, speed_sum=predicted_speed_sum)
        self._refresh_thermo()
        self._omega_step(half_dt)
        self.lagrangian_position -= (
            self.v_shock * self.atoms.get_volume() / self.v0 * self.dt
        )
        if not np.isfinite(self.lagrangian_position):
            raise FloatingPointError("MSST Lagrangian position is non-finite")
        self._update_diagnostics()
        return forces


def linear_ode_step(value, constant, linear, dt):
    """Advance ``dy/dt = constant + linear*y`` exactly for fixed coefficients.

    The removable singularity at ``linear == 0`` is evaluated by its analytic
    limit rather than by adding an epsilon to a denominator.
    """
    value = np.asarray(value, dtype=float)
    constant = np.asarray(constant, dtype=float)
    linear = np.asarray(linear, dtype=float)
    dt = float(dt)
    if not (
        np.isfinite(value).all()
        and np.isfinite(constant).all()
        and np.isfinite(linear).all()
        and np.isfinite(dt)
    ):
        raise FloatingPointError("linear ODE inputs must be finite")
    try:
        with np.errstate(divide="raise", over="raise", invalid="raise"):
            z = np.asarray(linear * dt, dtype=float)
            small = np.abs(z) < 1.0e-7
            phi1 = np.empty_like(z)
            np.divide(np.expm1(z), z, out=phi1, where=~small)
            series = 1.0 + z / 2.0 + z * z / 6.0 + z * z * z / 24.0
            np.copyto(phi1, series, where=small)
            result = np.exp(z) * value + dt * phi1 * constant
    except FloatingPointError as exc:
        raise FloatingPointError(
            "linear ODE exponential overflow or invalid operation"
        ) from exc
    if not np.isfinite(result).all():
        raise FloatingPointError("linear ODE result must be finite")
    return result
