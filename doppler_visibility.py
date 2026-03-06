"""
doppler_visibility.py
=====================
Core computation module for Doppler Visibility Maps in pulsed radar systems.

A Doppler Visibility Map shows which target radial velocities are detectable
(visible) vs. undetectable (blind) for a given set of Pulse Repetition
Frequencies (PRFs).

Physics Background
------------------
For a pulsed Doppler radar:
  - Wavelength:        λ = c / f_radar
  - Unambiguous vel:   v_ua = λ × PRF / 2
  - Blind speeds occur at integer multiples of v_ua:
        v_blind,n = n × (λ × PRF / 2),  n = ±1, ±2, ...

A target is "blind" if its radial velocity falls within a narrow window
around any blind speed. That window is defined by `blind_fraction` — a
fraction of the unambiguous velocity interval (e.g. 0.05 = 5%).

With multiple staggered PRFs, blind zones from different PRFs rarely
coincide, so the combined visibility (AND of all PRF visibilities) has
far fewer dead zones than any single PRF alone.

Usage
-----
    from doppler_visibility import DopplerVisibility

    dv = DopplerVisibility(
        f_radar=10e9,           # 10 GHz X-band radar
        prfs=[1000, 1250, 1500, 1750, 2000, 2500],  # Hz
        v_range=(-500, 500),    # m/s
        n_points=4000,
        blind_fraction=0.05
    )

    result = dv.compute()
    dv.plot(result)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Physical constants
C_LIGHT = 3e8  # speed of light, m/s


@dataclass
class VisibilityResult:
    """
    Holds the output of a Doppler visibility computation.

    Attributes
    ----------
    velocities : np.ndarray
        Velocity axis (m/s), shape (N,)
    visibility_per_prf : np.ndarray
        Binary visibility matrix, shape (n_prfs, N).
        1 = visible, 0 = blind.
    combined_visibility : np.ndarray
        Element-wise AND across all PRFs, shape (N,).
        1 = detectable by ALL PRFs, 0 = blind in at least one.
    blind_speeds : List[np.ndarray]
        Blind speed positions (m/s) for each PRF within v_range.
    prfs : List[float]
        PRF values used (Hz).
    wavelength : float
        Radar wavelength (m).
    unambiguous_velocities : List[float]
        Unambiguous velocity interval for each PRF (m/s).
    coverage_per_prf : List[float]
        Fraction of velocity space that is visible, per PRF.
    combined_coverage : float
        Fraction of velocity space visible to ALL PRFs combined.
    """
    velocities: np.ndarray
    visibility_per_prf: np.ndarray
    combined_visibility: np.ndarray
    blind_speeds: List[np.ndarray]
    prfs: List[float]
    wavelength: float
    unambiguous_velocities: List[float]
    coverage_per_prf: List[float]
    combined_coverage: float


class DopplerVisibility:
    """
    Computes and plots a Doppler Visibility Map for a pulsed radar system.

    Parameters
    ----------
    f_radar : float
        Radar carrier frequency in Hz (e.g. 10e9 for X-band).
    prfs : List[float]
        List of 2–8 Pulse Repetition Frequencies in Hz.
    v_range : Tuple[float, float]
        (v_min, v_max) velocity range to plot in m/s.
    n_points : int
        Number of velocity sample points. Higher = smoother map.
    blind_fraction : float
        Width of each blind zone as a fraction of the unambiguous velocity
        interval. E.g. 0.05 means ±2.5% around each blind speed is masked.
    """

    def __init__(
        self,
        f_radar: float = 10e9,
        prfs: Optional[List[float]] = None,
        v_range: Tuple[float, float] = (-600, 600),
        n_points: int = 4000,
        blind_fraction: float = 0.05,
    ):
        self.f_radar = f_radar
        self.prfs = prfs or [1000, 1250, 1500, 1750, 2000, 2500]
        self.v_range = v_range
        self.n_points = n_points
        self.blind_fraction = blind_fraction

        self.wavelength = C_LIGHT / f_radar

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _unambiguous_velocity(self, prf: float) -> float:
        """First blind speed = half the unambiguous velocity interval."""
        return self.wavelength * prf / 2.0

    def _visibility_for_prf(self, prf: float, velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute binary visibility array for a single PRF.

        A velocity v is BLIND if:
            |v mod v_ua - 0| < half_blind   OR   |v mod v_ua - v_ua| < half_blind
        where v_ua is the unambiguous velocity interval and half_blind is
        (blind_fraction / 2) × v_ua.

        Returns
        -------
        visibility : np.ndarray  (1 = visible, 0 = blind)
        blind_speed_positions : np.ndarray  (positions of blind speeds in v_range)
        """
        v_ua = self._unambiguous_velocity(prf)
        half_blind = (self.blind_fraction / 2.0) * v_ua

        # Phase of each velocity within the unambiguous interval [0, v_ua)
        phase = np.mod(velocities, v_ua)

        # Blind if near 0 (or equivalently near v_ua) in phase
        blind_mask = (phase < half_blind) | (phase > v_ua - half_blind)
        visibility = (~blind_mask).astype(np.float32)

        # Enumerate blind speed positions within v_range
        v_min, v_max = self.v_range
        n_max = int(np.ceil(abs(v_max) / v_ua)) + 1
        blind_positions = []
        for n in range(-n_max, n_max + 1):
            bp = n * v_ua
            if v_min <= bp <= v_max:
                blind_positions.append(bp)

        return visibility, np.array(blind_positions)

    def compute(self) -> VisibilityResult:
        """
        Run the full Doppler visibility computation.

        Returns a VisibilityResult dataclass with all maps and metrics.
        """
        velocities = np.linspace(self.v_range[0], self.v_range[1], self.n_points)

        visibility_per_prf = []
        blind_speeds = []
        unambiguous_velocities = []
        coverage_per_prf = []

        for prf in self.prfs:
            vis, bsp = self._visibility_for_prf(prf, velocities)
            visibility_per_prf.append(vis)
            blind_speeds.append(bsp)
            unambiguous_velocities.append(self._unambiguous_velocity(prf))
            coverage_per_prf.append(float(np.mean(vis)))

        visibility_matrix = np.array(visibility_per_prf)  # (n_prfs, N)
        combined_visibility = np.prod(visibility_matrix, axis=0)  # AND across PRFs
        combined_coverage = float(np.mean(combined_visibility))

        return VisibilityResult(
            velocities=velocities,
            visibility_per_prf=visibility_matrix,
            combined_visibility=combined_visibility,
            blind_speeds=blind_speeds,
            prfs=self.prfs,
            wavelength=self.wavelength,
            unambiguous_velocities=unambiguous_velocities,
            coverage_per_prf=coverage_per_prf,
            combined_coverage=combined_coverage,
        )

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self, result: VisibilityResult) -> str:
        """Return a formatted text summary of the visibility analysis."""
        lines = [
            "=" * 60,
            "  DOPPLER VISIBILITY MAP — ANALYSIS SUMMARY",
            "=" * 60,
            f"  Radar frequency : {self.f_radar / 1e9:.2f} GHz",
            f"  Wavelength       : {result.wavelength * 100:.2f} cm",
            f"  Velocity range   : {self.v_range[0]} to {self.v_range[1]} m/s",
            f"  Blind zone width : {self.blind_fraction * 100:.1f}% of v_ua",
            "",
            f"  {'PRF (Hz)':>12}  {'v_ua (m/s)':>12}  {'Coverage':>10}",
            "  " + "-" * 40,
        ]
        for i, prf in enumerate(result.prfs):
            lines.append(
                f"  {prf:>12.0f}  {result.unambiguous_velocities[i]:>12.2f}  "
                f"{result.coverage_per_prf[i] * 100:>9.1f}%"
            )
        lines += [
            "  " + "-" * 40,
            f"  {'COMBINED':>12}  {'':>12}  {result.combined_coverage * 100:>9.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------
# Standalone usage example
# ------------------------------------------------------------------

if __name__ == "__main__":
    dv = DopplerVisibility(
        f_radar=10e9,
        prfs=[1000, 1250, 1500, 1750, 2000, 2500],
        v_range=(-600, 600),
        blind_fraction=0.05,
    )
    result = dv.compute()
    print(dv.summary(result))
