"""Shared look settings: render parameters, the light rig and the jelly palette."""
import numpy as np

PARAMS = dict(
    EXPOSURE=1.12,
    BLOOM_STRENGTH=0.12,
    BLOOM_THRESHOLD=1.0,
    VIGNETTE=0.45,
    GRAIN=0.010,
    FLOOR_RGB=(0.15, 0.155, 0.17),
    FLOOR_ROUGH=0.22,
    FLOOR_F0=0.04,
    FLOOR_SPEC=0.85,
    AMB_RGB=(0.030, 0.032, 0.042),
    ENV_TOP_RGB=(0.020, 0.023, 0.034),
    ENV_HOR_RGB=(0.003, 0.0035, 0.005),
    FOG=0.07,
    FOG_RGB=(0.0015, 0.0017, 0.0026),
    IOR=1.38,
    AO_DIST=1.0,
    AMBSCAT_RGB=(0.09, 0.09, 0.10),
    LASER_GEL=1.2,
    LOOK_SAT=1.2,
    LOOK_POWER=1.15,
    ADAPT_THRESH=0.07,
    FLOOR_TEX=0.10,
    FIREFLY=30.0,
    MAX_DEPTH=4,
    REFL_DEPTH=1,
    LASER_FLOOR=1.0,
)

PALETTE_KW = dict(gdir=(1.0, 0.22, -0.14), scale=0.42, offset=0.42, sat=1.0, bright=0.97,
                  d0=0.22, sigma_s=2.6, g=0.30)


def rect_light(center, facing, w, h, rgb, spot=0.0, pick=1.0, shadow=True, up=(0.0, 1.0, 0.0)):
    """16-float light record. The emitting side faces `facing`."""
    n = np.asarray(facing, float)
    n /= np.linalg.norm(n)
    up = np.asarray(up, float)
    if abs(np.dot(up, n)) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    ey = up - n * np.dot(up, n)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, n)
    return [*np.asarray(center, float), *(ex * w / 2), *(ey * h / 2), *rgb, spot, pick, 1.0 if shadow else 0.0, 0.0]


def lights_for(center):
    """Softbox rig around an arena centre: key from above, rim strip behind, fill."""
    c = np.asarray(center, float)
    key = rect_light(c + [-0.8, 8.0, 1.6], [0.06, -1, -0.16], 2.6, 2.6, (55.0, 53.0, 50.0), spot=6.0, pick=1.0)
    rim_pos = c + [0.8, 4.2, -6.8]
    rim = rect_light(rim_pos, c + [0, 1.2, 0] - rim_pos, 6.0, 0.9, (34.0, 35.0, 38.0), pick=0.9)
    fill_pos = c + [-7.0, 3.5, 5.0]
    fill = rect_light(fill_pos, c + [0, 1.0, 0] - fill_pos, 4.4, 3.2, (1.3, 1.4, 1.6), pick=0.2, shadow=False)
    return [key, rim, fill]
