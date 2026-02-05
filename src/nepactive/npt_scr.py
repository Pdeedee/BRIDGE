"""
NPT ensemble using Stochastic Cell Rescaling (SCR) barostat
combined with BDP (Bussi-Donadio-Parrinello) thermostat.

References:
[1] Mattia Bernetti and Giovanni Bussi,
    Pressure control using stochastic cell rescaling,
    J. Chem. Phys. 153, 114107 (2020).
    https://doi.org/10.1063/1.5144289

[2] G. Bussi, D. Donadio, and M. Parrinello,
    Canonical sampling through velocity rescaling,
    J. Chem. Phys. 126, 014101 (2007).
    https://doi.org/10.1063/1.2408420
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, IO, List
from ase import Atoms, units
from ase.md.md import MolecularDynamics


# Boltzmann constant in eV/K (ASE uses eV as energy unit)
K_B = units.kB


def gasdev(rng: np.random.Generator) -> float:
    """Generate a random number from standard normal distribution."""
    return rng.standard_normal()


def gamma_deviate(rng: np.random.Generator, alpha: float) -> float:
    """Generate a random number from gamma distribution with shape alpha."""
    return rng.gamma(alpha)


def resamplekin(ke_old: float, sigma: float, ndeg: int, taut: float,
                rng: np.random.Generator) -> float:
    """
    Resample kinetic energy using the BDP algorithm.

    This implements the stochastic velocity rescaling thermostat from
    Bussi, Donadio, and Parrinello, J. Chem. Phys. 126, 014101 (2007).

    Parameters
    ----------
    ke_old : float
        Current kinetic energy
    sigma : float
        Target kinetic energy (0.5 * ndeg * kB * T)
    ndeg : int
        Number of degrees of freedom
    taut : float
        Thermostat coupling parameter (dimensionless)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    float
        New kinetic energy after rescaling
    """
    if taut > 0.1:
        c1 = np.exp(-1.0 / taut)
    else:
        c1 = 0.0
    c2 = (1.0 - c1) * sigma / ndeg

    r1 = gasdev(rng)

    if ndeg == 1:
        # Special case for 1 degree of freedom
        ke_new = c1 * ke_old + c2 * r1 * r1 + 2.0 * np.sqrt(c1 * c2 * ke_old) * r1
    else:
        # General case
        r2_sum = 2.0 * gamma_deviate(rng, (ndeg - 1) / 2.0)
        ke_new = (c1 * ke_old + c2 * (r2_sum + r1 * r1) +
                  2.0 * np.sqrt(c1 * c2 * ke_old) * r1)

    return max(ke_new, 0.0)


class NPT_SCR(MolecularDynamics):
    """
    NPT molecular dynamics using Stochastic Cell Rescaling barostat
    combined with the BDP (velocity rescaling) thermostat.

    This class implements the stochastic cell rescaling method from
    Bernetti and Bussi, J. Chem. Phys. 153, 114107 (2020), which provides
    a simple and robust way to control pressure in molecular dynamics.

    The thermostat uses the Bussi-Donadio-Parrinello velocity rescaling
    method from J. Chem. Phys. 126, 014101 (2007).

    Supports three pressure modes:
    - isotropic: uniform scaling in all directions
    - orthogonal: independent scaling in x, y, z (for orthorhombic cells)
    - triclinic: full 3x3 cell tensor scaling

    Parameters
    ----------
    atoms : Atoms
        The atoms object.
    timestep : float
        Time step in ASE time units (fs).
    temperature : float
        Target temperature in Kelvin.
    pressure : float or array_like
        Target pressure. For isotropic: single value in GPa.
        For orthogonal: [Pxx, Pyy, Pzz] in GPa.
        For triclinic: [Pxx, Pyy, Pzz, Pyz, Pxz, Pxy] in GPa (Voigt notation).
    temperature_coupling : float
        Thermostat coupling time scale in units of timestep.
        Typical value: 100 * timestep.
    pressure_coupling : float or array_like
        Barostat coupling parameter(s). Same shape as pressure.
        Typical value: 1e-4 to 1e-3 per step.
    pmode : str
        Pressure mode: 'iso', 'ortho', or 'tri'.
    deform : array_like, optional
        Deformation rates [rate_x, rate_y, rate_z] in Angstrom/step.
        If provided, overrides pressure control in that direction.
    seed : int, optional
        Random seed for reproducibility.
    trajectory : str, optional
        Trajectory file name.
    logfile : str or file, optional
        Log file for MD output.
    loginterval : int
        Interval for writing to log file.
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: float,
        pressure: Union[float, List[float]],
        temperature_coupling: float = 100.0,
        pressure_coupling: Union[float, List[float]] = 1e-4,
        pmode: str = 'iso',
        deform: Optional[List[float]] = None,
        seed: Optional[int] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[str, IO]] = None,
        loginterval: int = 1,
        **kwargs
    ):
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            **kwargs
        )

        self.temp_target = temperature
        self.temp_coupling = temperature_coupling
        self.pmode = pmode.lower()

        # Initialize random number generator
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Set up pressure targets and coupling based on mode
        self._setup_pressure(pressure, pressure_coupling)

        # Set up deformation if provided
        self._setup_deform(deform)

        # Cache atom properties
        self.natoms = len(atoms)
        assert hasattr(self, 'masses') and self.masses is not None

    def _setup_pressure(self, pressure, pressure_coupling):
        """Set up pressure targets and coupling parameters."""
        # Convert pressure from GPa to ASE internal units (eV/Å³)
        pressure_unit = units.GPa

        if self.pmode == 'iso':
            self.num_pressure_components = 1
            self.target_pressure = np.zeros(6)
            self.pressure_coupling = np.zeros(6)
            p = pressure if np.isscalar(pressure) else pressure[0]
            tau = pressure_coupling if np.isscalar(pressure_coupling) else pressure_coupling[0]
            self.target_pressure[:3] = p * pressure_unit
            self.pressure_coupling[:3] = tau

        elif self.pmode == 'ortho':
            self.num_pressure_components = 3
            self.target_pressure = np.zeros(6)
            self.pressure_coupling = np.zeros(6)
            if np.isscalar(pressure):
                self.target_pressure[:3] = pressure * pressure_unit
            else:
                self.target_pressure[:3] = np.array(pressure[:3]) * pressure_unit
            if np.isscalar(pressure_coupling):
                self.pressure_coupling[:3] = pressure_coupling
            else:
                self.pressure_coupling[:3] = np.array(pressure_coupling[:3])

        elif self.pmode == 'tri':
            self.num_pressure_components = 6
            self.target_pressure = np.zeros(6)
            self.pressure_coupling = np.zeros(6)
            if np.isscalar(pressure):
                self.target_pressure[:] = pressure * pressure_unit
            else:
                self.target_pressure[:] = np.array(pressure) * pressure_unit
            if np.isscalar(pressure_coupling):
                self.pressure_coupling[:] = pressure_coupling
            else:
                self.pressure_coupling[:] = np.array(pressure_coupling)
        else:
            raise ValueError(f"Unknown pressure mode: {self.pmode}. "
                           f"Use 'iso', 'ortho', or 'tri'.")

    def _setup_deform(self, deform):
        """Set up deformation rates."""
        if deform is None:
            self.deform = [False, False, False]
            self.deform_rate = [0.0, 0.0, 0.0]
        else:
            self.deform = [d != 0 for d in deform]
            self.deform_rate = list(deform)

    def get_stress_tensor(self) -> np.ndarray:
        """
        Get current stress tensor in Voigt notation.

        Returns stress as [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]
        in ASE internal units (eV/Å³).
        """
        # ASE returns stress with opposite sign convention, and in Voigt order
        # [xx, yy, zz, yz, xz, xy]
        stress = -self.atoms.get_stress(voigt=True, include_ideal_gas=True)
        return stress

    def get_pressure_diagonal(self) -> np.ndarray:
        """Get diagonal pressure components [Pxx, Pyy, Pzz]."""
        stress = self.get_stress_tensor()
        return stress[:3]

    def _apply_thermostat(self):
        """Apply BDP thermostat (stochastic velocity rescaling)."""
        velocities = self.atoms.get_velocities()
        masses = self.atoms.get_masses()

        # Calculate current kinetic energy
        ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

        # Degrees of freedom (3N for no constraints)
        ndeg = 3 * self.natoms

        # Target kinetic energy
        sigma = 0.5 * ndeg * K_B * self.temp_target

        # Resample kinetic energy
        ke_new = resamplekin(ke, sigma, ndeg, self.temp_coupling, self.rng)

        # Scale velocities
        if ke > 0:
            scale_factor = np.sqrt(ke_new / ke)
            self.atoms.set_velocities(velocities * scale_factor)

    def _apply_barostat_isotropic(self):
        """Apply isotropic SCR barostat."""
        cell = self.atoms.get_cell()
        volume = self.atoms.get_volume()

        # Get current pressure (average of diagonal) in GPa
        p_current = self.get_pressure_diagonal()
        p_instant = np.mean(p_current) / units.GPa  # Convert to GPa
        p_target = self.target_pressure[0] / units.GPa  # Convert to GPa
        tau = self.pressure_coupling[0]

        # Berendsen-like deterministic part (pressure in GPa)
        scale_berendsen = 1.0 - tau * (p_target - p_instant)

        # Stochastic part (SCR correction)
        # Factor 2/3 because 3 directions are coupled
        # Use GPUMD's K_B = 8.617333262e-5 eV/K
        scale_stochastic = (np.sqrt(2.0 / 3.0 * tau * K_B * self.temp_target / volume)
                          * gasdev(self.rng))

        scale_factor = scale_berendsen + scale_stochastic

        # Scale cell and positions
        new_cell = cell * scale_factor
        self.atoms.set_cell(new_cell, scale_atoms=True)

    def _apply_barostat_orthogonal(self):
        """Apply orthogonal SCR barostat (independent x, y, z scaling)."""
        cell = self.atoms.get_cell().array.copy()
        volume = self.atoms.get_volume()
        positions = self.atoms.get_positions()

        p_current = self.get_pressure_diagonal()
        scale_factors = np.ones(3)

        for i in range(3):
            if self.deform[i]:
                # Apply deformation
                old_length = cell[i, i]
                new_length = old_length + self.deform_rate[i]
                scale_factors[i] = new_length / old_length
            else:
                # Apply SCR barostat (pressure in GPa)
                p_target = self.target_pressure[i] / units.GPa
                p_inst = p_current[i] / units.GPa
                tau = self.pressure_coupling[i]

                scale_berendsen = 1.0 - tau * (p_target - p_inst)
                scale_stochastic = (np.sqrt(2.0 * tau * K_B * self.temp_target / volume)
                                  * gasdev(self.rng))
                scale_factors[i] = scale_berendsen + scale_stochastic

        # Scale cell diagonal elements
        for i in range(3):
            cell[i, i] *= scale_factors[i]

        # Scale positions
        positions[:, 0] *= scale_factors[0]
        positions[:, 1] *= scale_factors[1]
        positions[:, 2] *= scale_factors[2]

        self.atoms.set_cell(cell, scale_atoms=False)
        self.atoms.set_positions(positions)

    def _apply_barostat_triclinic(self):
        """Apply triclinic SCR barostat (full cell tensor scaling)."""
        cell = self.atoms.get_cell().array.copy()
        volume = self.atoms.get_volume()
        positions = self.atoms.get_positions()

        # Get full stress tensor and convert to GPa
        stress = self.get_stress_tensor() / units.GPa
        # stress order: [xx, yy, zz, yz, xz, xy]

        # Build mu matrix (deformation gradient)
        # target_pressure order: [xx, yy, zz, yz, xz, xy] (Voigt)
        mu = np.eye(3)

        # Diagonal elements
        for i in range(3):
            p_target = self.target_pressure[i] / units.GPa
            tau = self.pressure_coupling[i]
            mu[i, i] = 1.0 - tau * (p_target - stress[i])
            mu[i, i] += np.sqrt(2.0 * tau * K_B * self.temp_target / volume) * gasdev(self.rng)

        # Off-diagonal elements (symmetric)
        # xy: index 5 in Voigt, position [0,1] and [1,0]
        tau_xy = self.pressure_coupling[5]
        p_target_xy = self.target_pressure[5] / units.GPa
        mu_xy = -tau_xy * (p_target_xy - stress[5])
        mu_xy += np.sqrt(2.0 * tau_xy * K_B * self.temp_target / volume) * gasdev(self.rng)
        mu[0, 1] = mu[1, 0] = mu_xy

        # xz: index 4 in Voigt, position [0,2] and [2,0]
        tau_xz = self.pressure_coupling[4]
        p_target_xz = self.target_pressure[4] / units.GPa
        mu_xz = -tau_xz * (p_target_xz - stress[4])
        mu_xz += np.sqrt(2.0 * tau_xz * K_B * self.temp_target / volume) * gasdev(self.rng)
        mu[0, 2] = mu[2, 0] = mu_xz

        # yz: index 3 in Voigt, position [1,2] and [2,1]
        tau_yz = self.pressure_coupling[3]
        p_target_yz = self.target_pressure[3] / units.GPa
        mu_yz = -tau_yz * (p_target_yz - stress[3])
        mu_yz += np.sqrt(2.0 * tau_yz * K_B * self.temp_target / volume) * gasdev(self.rng)
        mu[1, 2] = mu[2, 1] = mu_yz

        # Apply transformation: h_new = mu @ h_old
        new_cell = mu @ cell

        # Transform positions: r_new = mu @ r_old
        new_positions = (mu @ positions.T).T

        self.atoms.set_cell(new_cell, scale_atoms=False)
        self.atoms.set_positions(new_positions)

    def _apply_barostat(self):
        """Apply the appropriate barostat based on pressure mode."""
        if self.num_pressure_components == 1:
            self._apply_barostat_isotropic()
        elif self.num_pressure_components == 3:
            self._apply_barostat_orthogonal()
        else:
            self._apply_barostat_triclinic()

    def step(self):
        """Perform one MD step using velocity Verlet with SCR barostat."""
        atoms = self.atoms

        # Get current state
        masses = atoms.get_masses()[:, np.newaxis]
        positions = atoms.get_positions()
        velocities = atoms.get_velocities()
        forces = atoms.get_forces()

        # First half of velocity Verlet: v(t + dt/2) = v(t) + a(t) * dt/2
        velocities += 0.5 * self.dt * forces / masses
        atoms.set_velocities(velocities)

        # Update positions: r(t + dt) = r(t) + v(t + dt/2) * dt
        positions += self.dt * velocities
        atoms.set_positions(positions)

        # Calculate new forces at new positions
        forces = atoms.get_forces()

        # Second half of velocity Verlet: v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        velocities = atoms.get_velocities()
        velocities += 0.5 * self.dt * forces / masses
        atoms.set_velocities(velocities)

        # Apply thermostat (BDP velocity rescaling)
        self._apply_thermostat()

        # Apply barostat (SCR)
        self._apply_barostat()

    def run(self, steps: int):
        """
        Run MD for a number of steps.

        Parameters
        ----------
        steps : int
            Number of MD steps to perform.
        """
        for _ in range(steps):
            self.step()
            self.nsteps += 1
            self.call_observers()

    def get_temperature(self) -> float:
        """Get current instantaneous temperature in Kelvin."""
        return self.atoms.get_temperature()

    def get_pressure(self) -> float:
        """Get current instantaneous pressure in GPa."""
        p_diag = self.get_pressure_diagonal()
        return np.mean(p_diag) / units.GPa

    def get_volume(self) -> float:
        """Get current volume in Å³."""
        return self.atoms.get_volume()


class NPT_SCR_Hugo(NPT_SCR):
    """
    NPT_SCR with Hugoniot thermostat for shock wave simulations.

    Inherits SCR barostat from NPT_SCR, but dynamically updates
    target temperature to satisfy the Hugoniot relation:
    E - E0 = 0.5 * (P + P0) * (V0 - V)
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        pressure: float,
        e0: Optional[float] = None,
        p0: Optional[float] = None,
        v0: Optional[float] = None,
        temperature_coupling: float = 100.0,
        pressure_coupling: float = 1e-4,
        pmode: str = 'iso',
        **kwargs
    ):
        # Store pmode for Hugoniot calculation
        self._pmode_hugo = pmode.lower()

        # Set reference state (before parent init to use in temp calculation)
        self.v0 = v0 if v0 is not None else atoms.get_volume()
        self.e0 = e0 if e0 is not None else atoms.get_total_energy()

        if p0 is None:
            # Get pressure from stress tensor (note: stress has opposite sign)
            self.p0 = -atoms.get_stress(voigt=False).trace() / 3
        else:
            self.p0 = p0 * units.GPa

        self.tdof = 3 * len(atoms)
        self.dhugo = 0.0

        # Initialize parent class with initial temperature
        t_init = max(300.0, self._compute_hugoniot_temp(atoms))
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            temperature=t_init,
            pressure=pressure,
            temperature_coupling=temperature_coupling,
            pressure_coupling=pressure_coupling,
            pmode='iso' if pmode in ['iso', 'x', 'y', 'z'] else pmode,
            **kwargs
        )

        self.print_init_parameters()

    def print_init_parameters(self):
        """Print initial parameters for debugging."""
        print(f"NPT_SCR_Hugo: e0={self.e0:.4f} eV, v0={self.v0:.2f} A^3, "
              f"p0={self.p0/units.GPa:.4f} GPa, pmode={self._pmode_hugo}")

    @property
    def volume(self):
        return self.atoms.get_volume()

    def _compute_hugoniot_temp(self, atoms: Atoms) -> float:
        """Compute target temperature from Hugoniot relation."""
        # Get current pressure based on pmode
        stress = -atoms.get_stress(voigt=False)
        if self._pmode_hugo == 'iso':
            p_current = stress.trace() / 3
        elif self._pmode_hugo == 'x':
            p_current = stress[0, 0]
        elif self._pmode_hugo == 'y':
            p_current = stress[1, 1]
        elif self._pmode_hugo == 'z':
            p_current = stress[2, 2]
        else:
            p_current = stress.trace() / 3

        # Hugoniot: dE = 0.5*(P0+P)*(V0-V)
        # dhugo > 0 means we need to heat up
        v_current = atoms.get_volume()
        e_current = atoms.get_total_energy()

        dhugo = (0.5 * (self.p0 + p_current) * (self.v0 - v_current)
                 + self.e0 - e_current)
        self.dhugo = dhugo / (self.tdof * K_B)  # Convert to temperature

        return atoms.get_temperature() + self.dhugo

    def _apply_thermostat(self):
        """Apply thermostat with dynamically updated Hugoniot temperature."""
        self.temp_target = max(300.0, self._compute_hugoniot_temp(self.atoms))
        super()._apply_thermostat()

    def get_hugoniot_deviation(self) -> float:
        """Get current Hugoniot deviation in temperature units."""
        return self.dhugo


class NPH_SCR(NPT_SCR):
    """
    NPH (constant enthalpy, constant pressure) using SCR barostat.

    This is NPT without thermostat - temperature evolves freely while
    pressure is controlled by the SCR barostat.

    Useful for:
    - Studying adiabatic processes
    - Shock wave simulations without artificial temperature control
    - Cases where you want pressure control but natural energy conservation
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        pressure: Union[float, List[float]],
        pressure_coupling: Union[float, List[float]] = 1e-4,
        pmode: str = 'iso',
        deform: Optional[List[float]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        # Initialize parent with dummy temperature (won't be used)
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            temperature=300.0,  # Not used, just for initialization
            pressure=pressure,
            temperature_coupling=100.0,  # Not used
            pressure_coupling=pressure_coupling,
            pmode=pmode,
            deform=deform,
            seed=seed,
            **kwargs
        )

    def _apply_thermostat(self):
        """No thermostat in NPH - temperature evolves freely."""
        pass

    def step(self):
        """Perform one MD step using velocity Verlet with SCR barostat only."""
        atoms = self.atoms

        # Get current state
        masses = atoms.get_masses()[:, np.newaxis]
        positions = atoms.get_positions()
        velocities = atoms.get_velocities()
        forces = atoms.get_forces()

        # First half of velocity Verlet
        velocities += 0.5 * self.dt * forces / masses
        atoms.set_velocities(velocities)

        # Update positions
        positions += self.dt * velocities
        atoms.set_positions(positions)

        # Calculate new forces
        forces = atoms.get_forces()

        # Second half of velocity Verlet
        velocities = atoms.get_velocities()
        velocities += 0.5 * self.dt * forces / masses
        atoms.set_velocities(velocities)

        # Apply barostat only (no thermostat)
        self._apply_barostat()


class NPH_SCR_Hugo(NPH_SCR):
    """
    NPH with Hugoniot constraint using SCR barostat.

    This combines SCR barostat with Hugoniot energy constraint.
    Instead of controlling temperature, it monitors the Hugoniot deviation
    and can optionally apply velocity scaling to stay on the Hugoniot curve.

    The Hugoniot relation: E - E0 = 0.5 * (P + P0) * (V0 - V)
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        pressure: float,
        e0: Optional[float] = None,
        p0: Optional[float] = None,
        v0: Optional[float] = None,
        pressure_coupling: float = 1e-4,
        pmode: str = 'iso',
        hugoniot_correction: bool = True,
        **kwargs
    ):
        self._pmode_hugo = pmode.lower()
        self.hugoniot_correction = hugoniot_correction

        # Set reference state
        self.v0 = v0 if v0 is not None else atoms.get_volume()
        self.e0 = e0 if e0 is not None else atoms.get_total_energy()

        if p0 is None:
            self.p0 = -atoms.get_stress(voigt=False).trace() / 3
        else:
            self.p0 = p0 * units.GPa

        self.tdof = 3 * len(atoms)
        self.dhugo = 0.0

        super().__init__(
            atoms=atoms,
            timestep=timestep,
            pressure=pressure,
            pressure_coupling=pressure_coupling,
            pmode='iso' if pmode in ['iso', 'x', 'y', 'z'] else pmode,
            **kwargs
        )

        self.print_init_parameters()

    def print_init_parameters(self):
        """Print initial parameters."""
        print(f"NPH_SCR_Hugo: e0={self.e0:.4f} eV, v0={self.v0:.2f} A^3, "
              f"p0={self.p0/units.GPa:.4f} GPa, pmode={self._pmode_hugo}")

    @property
    def volume(self):
        return self.atoms.get_volume()

    def compute_hugoniot(self) -> float:
        """
        Compute Hugoniot deviation.

        Returns dhugo in temperature units (K).
        Positive means system needs to heat up to reach Hugoniot.
        """
        stress = -self.atoms.get_stress(voigt=False)
        if self._pmode_hugo == 'iso':
            p_current = stress.trace() / 3
        elif self._pmode_hugo == 'x':
            p_current = stress[0, 0]
        elif self._pmode_hugo == 'y':
            p_current = stress[1, 1]
        elif self._pmode_hugo == 'z':
            p_current = stress[2, 2]
        else:
            p_current = stress.trace() / 3

        v_current = self.atoms.get_volume()
        e_current = self.atoms.get_total_energy()

        # Hugoniot: E - E0 = 0.5*(P0+P)*(V0-V)
        dhugo = (0.5 * (self.p0 + p_current) * (self.v0 - v_current)
                 + self.e0 - e_current)
        self.dhugo = dhugo / (self.tdof * K_B)

        return self.dhugo

    def step(self):
        """Perform one MD step with optional Hugoniot correction."""
        # Standard NPH step
        super().step()

        # Compute Hugoniot deviation
        self.compute_hugoniot()

        # Optional: apply velocity scaling to stay on Hugoniot
        if self.hugoniot_correction and abs(self.dhugo) > 1.0:
            # Scale velocities to correct energy
            ke_current = self.atoms.get_kinetic_energy()
            ke_target = ke_current + self.dhugo * self.tdof * K_B * 0.5
            if ke_current > 0 and ke_target > 0:
                scale = np.sqrt(ke_target / ke_current)
                # Apply gentle correction (max 1% per step)
                scale = np.clip(scale, 0.99, 1.01)
                velocities = self.atoms.get_velocities()
                self.atoms.set_velocities(velocities * scale)

    def get_hugoniot_deviation(self) -> float:
        """Get current Hugoniot deviation in temperature units."""
        return self.dhugo


# Convenience function for creating NPT_SCR dynamics
def create_npt_scr(
    atoms: Atoms,
    timestep: float = 1.0,  # fs
    temperature: float = 300.0,  # K
    pressure: float = 0.0,  # GPa
    temperature_coupling: float = 100.0,
    pressure_coupling: float = 1e-4,
    pmode: str = 'iso',
    **kwargs
) -> NPT_SCR:
    """
    Create an NPT_SCR molecular dynamics object.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    timestep : float
        Time step in fs. Default: 1.0 fs.
    temperature : float
        Target temperature in K. Default: 300 K.
    pressure : float
        Target pressure in GPa. Default: 0 GPa.
    temperature_coupling : float
        Thermostat coupling in units of timestep. Default: 100.
    pressure_coupling : float
        Barostat coupling parameter. Default: 1e-4.
    pmode : str
        Pressure mode: 'iso', 'ortho', or 'tri'. Default: 'iso'.
    **kwargs
        Additional arguments passed to NPT_SCR.

    Returns
    -------
    NPT_SCR
        Configured molecular dynamics object.
    """
    from ase import units as ase_units

    return NPT_SCR(
        atoms=atoms,
        timestep=timestep * ase_units.fs,
        temperature=temperature,
        pressure=pressure,
        temperature_coupling=temperature_coupling,
        pressure_coupling=pressure_coupling,
        pmode=pmode,
        **kwargs
    )
