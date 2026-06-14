from __future__ import annotations

import math

from .constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM


def rot_x(v: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = v
    c = math.cos(angle)
    s = math.sin(angle)
    return (x, y * c - z * s, y * s + z * c)


def rot_z(v: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = v
    c = math.cos(angle)
    s = math.sin(angle)
    return (x * c - y * s, x * s + y * c, z)


def xy_unit(v: tuple[float, float, float]) -> tuple[float, float] | None:
    x, y, _ = v
    norm = math.hypot(x, y)
    if norm == 0:
        return None
    return (x / norm, y / norm)


def vector_unit(
    v: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    norm = math.sqrt(sum(component * component for component in v))
    if norm == 0:
        return None
    return tuple(component / norm for component in v)


def circular_state(
    radius_km: float,
    inclination_rad: float,
    raan_rad: float,
    argument_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (radius_km**3))
    pos = (
        radius_km * math.cos(argument_rad),
        radius_km * math.sin(argument_rad),
        0.0,
    )
    vel = (
        -radius_km * mean_motion * math.sin(argument_rad),
        radius_km * mean_motion * math.cos(argument_rad),
        0.0,
    )
    return rot_z(rot_x(pos, inclination_rad), raan_rad), rot_z(rot_x(vel, inclination_rad), raan_rad)


def is_sunlit_cylindrical_shadow(
    pos_km: tuple[float, float, float],
    sun_unit: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> bool:
    x, y, z = pos_km
    sx, sy, sz = sun_unit
    along_sun = x * sx + y * sy + z * sz
    if along_sun >= 0:
        return True
    px = x - along_sun * sx
    py = y - along_sun * sy
    pz = z - along_sun * sz
    return math.sqrt(px * px + py * py + pz * pz) >= EARTH_RADIUS_KM
