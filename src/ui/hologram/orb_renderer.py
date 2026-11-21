"""Genuine 3D holographic orb renderer for the VoiceOS HUD.

A :class:`QOpenGLWidget` that draws a real 3D energy core: a shaded
sphere with Fresnel rim lighting and travelling energy pulses, several
concentric torus rings, and a particle field — all GPU-rendered through
custom GLSL shaders.

Geometry is generated once in ``initializeGL`` and reused every frame;
per-frame cost is a handful of draw calls plus uniform updates, which
keeps the Intel Iris Xe comfortably at 30+ FPS.

Visual state is driven entirely through plain Python setters
(``set_level``, ``set_mode``, ``set_speaking`` ...), so the voice-facing
bridge never touches OpenGL objects.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QMatrix4x4, QPainter, QRadialGradient, QVector3D
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFunctions_2_0,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLSL sources
# ---------------------------------------------------------------------------

_SPHERE_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform vec3 uCameraPos;

out vec3 vNormal;
out vec3 vViewDir;
out vec3 vObjPos;
out vec2 vUv;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vObjPos = aPos;
    vNormal = normalize(mat3(uModel) * aNormal);
    vViewDir = normalize(uCameraPos - world.xyz);
    vUv = aUv;
    gl_Position = uMVP * world;
}
"""

_SPHERE_FRAG = """
#version 330 core
precision highp float;

in vec3 vNormal;
in vec3 vViewDir;
in vec3 vObjPos;
in vec2 vUv;

uniform float uTime;
uniform float uAudio;        // 0..1 live input level
uniform float uEnergy;       // 0..1 base energy (state driven)
uniform float uPulseAngle;   // travelling energy pulse position (radians)
uniform float uPulseStrength;
uniform float uScan;         // 0..1 scanning sweep progress, <0 = off
uniform float uError;        // amber error tint amount
uniform float uBrightness;   // appearance: overall luminosity 0..1
uniform float uGlow;         // appearance: sphere emissive rim 0..1
uniform float uOpacity;      // appearance: artwork alpha 0..1
uniform vec3 uTint;

out vec4 fragColor;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(vViewDir);
    float ndv = max(dot(N, V), 0.0);
    float fresnel = pow(1.0 - ndv, 2.2);

    // Base body: deep electric blue core with cyan banding.
    float band = smoothstep(0.15, 0.55, abs(vObjPos.y)) * 0.5 + 0.5;
    vec3 deep = vec3(0.008, 0.030, 0.090);
    vec3 core = vec3(0.020, 0.220, 0.420);
    vec3 hot  = vec3(0.120, 0.750, 0.950);
    vec3 col = mix(deep, core, band);
    col = mix(col, hot, fresnel * 0.85);

    // Travelling energy pulse around the equator.
    float ang = atan(vObjPos.z, vObjPos.x);
    float d = abs(mod(ang - uPulseAngle + PI, TAU) - PI);
    float pulse = exp(-d * d * 26.0) * uPulseStrength;
    col += vec3(0.35, 0.90, 1.00) * pulse * 1.6;

    // Scanning sweep: bright band sweeping latitude while analysing.
    float lat = vObjPos.y;
    float scanPos = mix(-1.15, 1.15, uScan);
    float scanD = abs(lat - scanPos);
    float scan = exp(-scanD * scanD * 60.0) * step(0.001, uScan)
               * step(uScan, 0.999);
    col += vec3(0.25, 0.70, 1.00) * scan * 1.2;

    // Audio-reactive shimmer.
    col += vec3(0.10, 0.45, 0.60) * uAudio * 0.9;

    // Faint latitude/longitude grid for a technical hologram feel.
    float grid = smoothstep(0.985, 1.0, abs(sin(vUv.x * 24.0)));
    float grid2 = smoothstep(0.985, 1.0, abs(sin(vUv.y * 12.0)));
    col += vec3(0.05, 0.30, 0.40) * (grid * 0.25 + grid2 * 0.15);

    // Restrained amber error tint.
    col = mix(col, vec3(0.55, 0.28, 0.05), uError * 0.35);

    // Appearance controls: controllable emissive rim (glow), overall
    // luminosity (brightness) and artwork alpha (opacity). Fully
    // transparent pixels stay transparent — opacity only scales the
    // alpha the shader already produced.
    col += vec3(0.10, 0.55, 0.75) * fresnel * uGlow * 0.9;
    col *= uBrightness;
    float alpha = clamp(0.35 + fresnel * 0.65 + uEnergy * 0.25
                        + uAudio * 0.2, 0.0, 1.0) * uOpacity;
    fragColor = vec4(col * uTint * (0.55 + uEnergy * 0.9
                                    + pulse * 0.8 + uAudio * 0.8), alpha);
}
"""

_RING_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform vec3 uCameraPos;

out vec3 vNormal;
out vec3 vViewDir;
out vec3 vObjPos;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vObjPos = aPos;
    vNormal = normalize(mat3(uModel) * aNormal);
    vViewDir = normalize(uCameraPos - world.xyz);
    gl_Position = uMVP * world;
}
"""

_RING_FRAG = """
#version 330 core
precision highp float;

in vec3 vNormal;
in vec3 vViewDir;
in vec3 vObjPos;

uniform float uTime;
uniform float uAudio;
uniform float uEnergy;
uniform float uSpeed;      // energy packet speed multiplier
uniform float uFlow;       // packet visibility 0..1
uniform float uError;
uniform float uBrightness; // appearance gain (overall x ring pref)
uniform float uOpacity;    // appearance artwork alpha 0..1
uniform vec3 uColor;

out vec4 fragColor;

const float TAU = 6.28318530718;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(vViewDir);
    float ndv = abs(dot(N, V));
    float rim = pow(1.0 - ndv, 1.6);

    // Travelling energy packets around the ring.
    float ang = atan(vObjPos.z, vObjPos.x);
    float packets = 0.0;
    for (int i = 0; i < 3; i++) {
        float p = fract(ang / TAU * 3.0 + uTime * uSpeed * 0.35
                        + float(i) * 0.3333);
        float d = abs(p - 0.5) * 2.0;
        packets += exp(-d * d * 18.0) * 0.9;
    }
    packets *= uFlow;

    float glow = 0.25 + uEnergy * 0.55 + uAudio * 0.8 + packets * 0.9;
    vec3 col = uColor * glow + vec3(0.9, 0.98, 1.0) * packets * 0.8;
    col = mix(col, vec3(0.9, 0.45, 0.08), uError * 0.6);
    col *= uBrightness;
    float alpha = clamp(0.15 + rim * 0.85 + uEnergy * 0.35,
                        0.0, 1.0) * uOpacity;
    fragColor = vec4(col, alpha);
}
"""

_POINT_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aPhase;
layout(location = 2) in float aSize;

uniform mat4 uMVP;
uniform float uTime;
uniform float uAudio;
uniform float uPixelRatio;

out float vTwinkle;
out float vFade;

void main() {
    vec3 p = aPos;
    // Gentle orbital drift + audio-reactive jitter.
    float t = uTime * 0.15 + aPhase * 6.28318;
    p.x += sin(t * 1.7 + aPhase * 9.0) * 0.06;
    p.y += cos(t * 1.3 + aPhase * 7.0) * 0.06;
    p.z += sin(t * 1.1 + aPhase * 5.0) * 0.06;
    float jitter = uAudio * 0.05;
    p += vec3(sin(aPhase * 40.0 + uTime * 3.0),
              cos(aPhase * 31.0 + uTime * 2.6),
              sin(aPhase * 23.0 + uTime * 2.2)) * jitter;
    vTwinkle = 0.55 + 0.45 * sin(uTime * (1.5 + aPhase) + aPhase * 20.0);
    vFade = 0.35 + uAudio * 0.65;
    vec4 clip = uMVP * vec4(p, 1.0);
    gl_Position = clip;
    float dist = max(0.1, -clip.z);
    gl_PointSize = aSize * uPixelRatio * 220.0 / dist;
}
"""

_POINT_FRAG = """
#version 330 core
precision highp float;

in float vTwinkle;
in float vFade;

uniform float uBrightness; // appearance gain (overall x particle pref)
uniform float uOpacity;    // appearance artwork alpha 0..1

out vec4 fragColor;

void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c) * 2.0;
    float a = smoothstep(1.0, 0.1, d);
    vec3 col = vec3(0.45, 0.85, 1.0);
    fragColor = vec4(col * (0.6 + vTwinkle * 0.8) * uBrightness,
                     a * vFade * 0.8 * uOpacity);
}
"""

# ---------------------------------------------------------------------------
# Geometry builders (run once on the GL thread)
# ---------------------------------------------------------------------------

def _sphere_mesh(radius: float, lat: int = 40, lon: int = 72):
    """UV sphere -> (positions, normals, uvs, indices) float32/uint32."""
    verts, norms, uvs = [], [], []
    for i in range(lat + 1):
        theta = math.pi * i / lat
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(lon + 1):
            phi = 2.0 * math.pi * j / lon
            x = radius * st * math.cos(phi)
            y = radius * ct
            z = radius * st * math.sin(phi)
            verts.append((x, y, z))
            norms.append((st * math.cos(phi), ct, st * math.sin(phi)))
            uvs.append((j / lon, i / lat))
    idx = []
    for i in range(lat):
        for j in range(lon):
            a = i * (lon + 1) + j
            b = a + lon + 1
            idx += [a, b, a + 1, b, b + 1, a + 1]
    return (
        np.asarray(verts, np.float32),
        np.asarray(norms, np.float32),
        np.asarray(uvs, np.float32),
        np.asarray(idx, np.uint32),
    )


def _torus_mesh(major: float, minor: float, seg_u: int = 160,
                seg_v: int = 10):
    """Torus in the XZ plane (axis = Y) -> interleaved pos+normal."""
    data = np.zeros(((seg_u + 1) * (seg_v + 1), 6), np.float32)
    k = 0
    for i in range(seg_u + 1):
        u = 2.0 * math.pi * i / seg_u
        cu, su = math.cos(u), math.sin(u)
        for j in range(seg_v + 1):
            v = 2.0 * math.pi * j / seg_v
            cv, sv = math.cos(v), math.sin(v)
            data[k, 0] = (major + minor * cv) * cu
            data[k, 1] = minor * sv
            data[k, 2] = (major + minor * cv) * su
            data[k, 3] = cv * cu
            data[k, 4] = sv
            data[k, 5] = cv * su
            k += 1
    idx = []
    row = seg_v + 1
    for i in range(seg_u):
        for j in range(seg_v):
            a = i * row + j
            b = a + row
            idx += [a, b, a + 1, b, b + 1, a + 1]
    return data, np.asarray(idx, np.uint32)


def _particle_shell(count: int = 420, seed: int = 1337):
    """Random points on a spherical shell -> interleaved (pos,phase,size)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, count)
    theta = rng.uniform(0.0, 2.0 * math.pi, count)
    r = rng.uniform(2.1, 3.4, count)
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos = np.stack([r * s * np.cos(theta), r * z, r * s * np.sin(theta)],
                   axis=1)
    data = np.zeros((count, 5), np.float32)
    data[:, 0:3] = pos
    data[:, 3] = rng.uniform(0.0, 1.0, count)
    data[:, 4] = rng.uniform(1.4, 3.4, count)
    return data


# ---------------------------------------------------------------------------
# Orb widget
# ---------------------------------------------------------------------------

class OrbGLWidget(QOpenGLWidget):
    """3D reactive orb. All visual parameters are plain Python values.

    Modes: ``standby``, ``activated``, ``listening``, ``processing``,
    ``speaking``, ``error``. ``level`` is a smoothed 0..1 audio-reactive
    value supplied by the HUD bridge.
    """

    # Ring definitions: (major radius, minor radius, tilt euler XYZ deg,
    # base rotation speed, color RGB 0..1, base energy)
    _RINGS = (
        (1.55, 0.030, (20.0, 0.0, 6.0), 0.55, (0.15, 0.75, 1.00), 0.55),
        (1.95, 0.020, (-31.0, 14.0, 0.0), -0.38, (0.05, 0.55, 0.95), 0.40),
        (2.45, 0.014, (66.0, -17.0, 26.0), 0.28, (0.35, 0.85, 1.00), 0.30),
    )
    _PARTICLE_COUNT = 420

    _MODE_ENERGY = {
        "standby": 0.35,
        "activated": 0.85,
        "listening": 0.60,
        "processing": 0.75,
        "speaking": 0.90,
        "error": 0.45,
    }
    _MODE_SPEED = {
        "standby": 0.35, "activated": 1.6, "listening": 0.9,
        "processing": 2.6, "speaking": 1.8, "error": 0.5,
    }
    _MODE_FLOW = {
        "standby": 0.15, "activated": 1.0, "listening": 0.7,
        "processing": 1.6, "speaking": 1.4, "error": 0.3,
    }

    # Balanced appearance defaults (V10.8). ``speed`` is a 0..1
    # multiplier on the animation clock (1.0 = 100%). ``waveform`` is
    # consumed by the WaveformWidget, not the orb.
    _APPEARANCE_DEFAULTS = {
        "brightness": 0.65,
        "glow": 0.55,
        "opacity": 0.95,
        "ring": 0.75,
        "particle": 0.60,
        "waveform": 0.70,
        "speed": 1.0,
    }

    # Voice-state visual modulation: (brightness multiplier, glow
    # multiplier) applied on top of the saved appearance. These are
    # render-time only — the persisted appearance is never touched —
    # and they ease toward their targets so state changes never flash.
    _MODE_GAIN = {
        "standby": (1.00, 1.00),
        "activated": (1.10, 1.15),
        "listening": (1.12, 1.10),
        "processing": (1.06, 1.22),
        "speaking": (1.15, 1.10),
        "error": (0.95, 0.90),
    }
    _GAIN_LERP = 4.0  # per-second easing toward the mode target

    def __init__(self, parent=None, target_fps: int = 30) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Visual state (GUI thread only).
        self._mode = "standby"
        self._level = 0.0
        self._speaking = False
        self._pulse = 0.0
        self._pulse_angle = 0.0
        self._time = 0.0
        self._gl_failed = False
        self._initialized = False

        # Appearance settings (0..1 fractions) + smoothed voice-state
        # gain multipliers. GUI thread only.
        self._appearance = dict(self._APPEARANCE_DEFAULTS)
        self._gain_b = 1.0
        self._gain_g = 1.0

        # GL objects (created in initializeGL).
        self._gl: Optional[QOpenGLFunctions_2_0] = None
        self._prog_sphere = None
        self._prog_ring = None
        self._prog_point = None
        self._vao_sphere = None
        self._vao_rings = []
        self._vao_points = None
        self._buffers = []
        self._sphere_vert_count = 0
        self._ring_vert_counts = []

        interval = max(8, int(1000 / max(1, target_fps)))
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    @property
    def gl_failed(self) -> bool:
        return self._gl_failed

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode in self._MODE_ENERGY:
            self._mode = mode

    def trigger_pulse(self) -> None:
        """Fire the wake-word energy pulse across the sphere."""
        self._pulse = 1.0

    def set_level(self, level: float) -> None:
        """Set the smoothed audio-reactive level (0..1)."""
        self._level = max(0.0, min(1.0, float(level)))

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = bool(speaking)

    def set_appearance(self, **params) -> None:
        """Update appearance uniforms (0..1 fractions).

        Accepted keys: ``brightness``, ``glow``, ``opacity``, ``ring``,
        ``particle``, ``speed`` (``waveform`` is handled by the HUD's
        waveform widget). Values are clamped; unknown keys ignored.
        Only uniforms change — geometry and GL objects are untouched.
        """
        changed = False
        for key, value in params.items():
            if key not in self._appearance or key == "waveform":
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v != v:  # NaN
                continue
            v = max(0.0, min(1.0, v))
            if v != self._appearance[key]:
                self._appearance[key] = v
                changed = True
        if changed:
            self.update()

    def set_rendering_active(self, active: bool) -> None:
        """Pause/resume the render timer (hidden/minimized HUD)."""
        if active:
            if not self._timer.isActive():
                self._timer.start()
        elif self._timer.isActive():
            self._timer.stop()

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        dt = self._timer.interval() / 1000.0
        # Animation-speed control scales the shared animation clock;
        # pulse decay stays real-time so pulses still feel responsive.
        self._time += dt * self._appearance["speed"]
        # Ease the voice-state gain multipliers toward the mode target.
        target_b, target_g = self._MODE_GAIN.get(
            self._mode, (1.0, 1.0))
        k = min(1.0, dt * self._GAIN_LERP)
        self._gain_b += (target_b - self._gain_b) * k
        self._gain_g += (target_g - self._gain_g) * k
        if self._pulse > 0.0:
            self._pulse = max(0.0, self._pulse - 0.75 * dt)
            self._pulse_angle += 0.11
        self.update()

    # ------------------------------------------------------------------
    # OpenGL lifecycle
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        try:
            self._gl = QOpenGLFunctions_2_0()
            self._gl.initializeOpenGLFunctions()

            self._prog_sphere = self._make_program(
                _SPHERE_VERT, _SPHERE_FRAG, "sphere")
            self._prog_ring = self._make_program(
                _RING_VERT, _RING_FRAG, "ring")
            self._prog_point = self._make_program(
                _POINT_VERT, _POINT_FRAG, "point")
            self._build_geometry()
            self._initialized = True
            logger.info("OrbGLWidget: 3D context initialized")
        except Exception:  # noqa: BLE001
            self._gl_failed = True
            logger.exception(
                "OrbGLWidget: 3D initialization failed — HUD will use "
                "the 2D fallback presentation")

    @staticmethod
    def _make_program(vert: str, frag: str, name: str) -> QOpenGLShaderProgram:
        """Compile a shader program, trying desktop GLSL then GLSL ES.

        Qt may give us a desktop GL context or ANGLE (OpenGL ES); the
        shaders are written to compile under both by swapping only the
        version directive.
        """
        last_err = ""
        # (version directive, strip precision qualifiers?) — desktop
        # GLSL on the Intel driver rejects `precision` below 400, while
        # GLSL ES requires it in fragment shaders.
        for version, strip_precision in (
            ("#version 400 core", True),
            ("#version 300 es", False),
        ):
            prog = QOpenGLShaderProgram()
            v = re.sub(r"^#version.*$", version, vert,
                       count=1, flags=re.MULTILINE)
            f = re.sub(r"^#version.*$", version, frag,
                       count=1, flags=re.MULTILINE)
            if strip_precision:
                f = re.sub(r"^precision[^\n]*\n", "", f,
                           flags=re.MULTILINE)
                v = re.sub(r"^precision[^\n]*\n", "", v,
                           flags=re.MULTILINE)
            if not prog.addShaderFromSourceCode(
                    QOpenGLShader.ShaderTypeBit.Vertex, v):
                last_err = f"{name} vertex shader failed"
                continue
            if not prog.addShaderFromSourceCode(
                    QOpenGLShader.ShaderTypeBit.Fragment, f):
                last_err = (f"{name} fragment shader failed: "
                            f"{prog.log()}")
                continue
            if prog.link():
                return prog
            last_err = f"{name} program link failed: {prog.log()}"
        raise RuntimeError(last_err or f"{name} shader failed")

    def _vbo(self, data: np.ndarray):
        buf = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        buf.create()
        buf.bind()
        buf.allocate(data.tobytes(), data.nbytes)
        buf.release()
        self._buffers.append(buf)
        return buf

    @staticmethod
    def _unindex(data: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Expand an indexed mesh into a triangle soup (done once)."""
        return data[idx.astype(np.intp)]

    def _build_geometry(self) -> None:
        # --- core sphere (interleaved pos+normal+uv, stride 32) ---------
        pos, nrm, uv, idx = _sphere_mesh(1.0)
        sdata = np.zeros((pos.shape[0], 8), np.float32)
        sdata[:, 0:3] = pos
        sdata[:, 3:6] = nrm
        sdata[:, 6:8] = uv
        sdata = self._unindex(sdata, idx)
        prog = self._prog_sphere
        prog.bind()
        loc_pos = prog.attributeLocation(b"aPos")
        loc_nrm = prog.attributeLocation(b"aNormal")
        loc_uv = prog.attributeLocation(b"aUv")
        prog.release()

        self._vao_sphere = QOpenGLVertexArrayObject()
        self._vao_sphere.create()
        self._vao_sphere.bind()
        buf = self._vbo(sdata)
        buf.bind()
        prog.enableAttributeArray(loc_pos)
        prog.setAttributeBuffer(loc_pos, 0x1406, 0, 3, 32)
        prog.enableAttributeArray(loc_nrm)
        prog.setAttributeBuffer(loc_nrm, 0x1406, 12, 3, 32)
        prog.enableAttributeArray(loc_uv)
        prog.setAttributeBuffer(loc_uv, 0x1406, 24, 2, 32)
        buf.release()
        self._sphere_vert_count = int(sdata.shape[0])
        self._vao_sphere.release()

        # --- rings (interleaved pos+normal, stride 24) -------------------
        prog = self._prog_ring
        prog.bind()
        rloc_pos = prog.attributeLocation(b"aPos")
        rloc_nrm = prog.attributeLocation(b"aNormal")
        prog.release()
        for major, minor, _tilt, _speed, _color, _e in self._RINGS:
            data, ridx = _torus_mesh(major, minor)
            data = self._unindex(data, ridx)
            vao = QOpenGLVertexArrayObject()
            vao.create()
            vao.bind()
            buf = self._vbo(data)
            buf.bind()
            prog.enableAttributeArray(rloc_pos)
            prog.setAttributeBuffer(rloc_pos, 0x1406, 0, 3, 24)
            prog.enableAttributeArray(rloc_nrm)
            prog.setAttributeBuffer(rloc_nrm, 0x1406, 12, 3, 24)
            buf.release()
            vao.release()
            self._vao_rings.append(vao)
            self._ring_vert_counts.append(int(data.shape[0]))

        # --- particles (interleaved pos+phase+size, stride 20) -----------
        prog = self._prog_point
        prog.bind()
        ploc_pos = prog.attributeLocation(b"aPos")
        ploc_phase = prog.attributeLocation(b"aPhase")
        ploc_size = prog.attributeLocation(b"aSize")
        prog.release()
        pdata = _particle_shell(self._PARTICLE_COUNT)
        self._vao_points = QOpenGLVertexArrayObject()
        self._vao_points.create()
        self._vao_points.bind()
        buf = self._vbo(pdata)
        buf.bind()
        prog.enableAttributeArray(ploc_pos)
        prog.setAttributeBuffer(ploc_pos, 0x1406,
                                0, 3, 20)
        prog.enableAttributeArray(ploc_phase)
        prog.setAttributeBuffer(ploc_phase,
                                0x1406, 12, 1, 20)
        prog.enableAttributeArray(ploc_size)
        prog.setAttributeBuffer(ploc_size,
                                0x1406, 16, 1, 20)
        buf.release()
        self._vao_points.release()

    def resizeGL(self, w: int, h: int) -> None:
        if self._gl is not None:
            self._gl.glViewport(0, 0, w, h)

    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        if self._gl_failed or not self._initialized:
            self._paint_fallback()
            return
        try:
            self._render()
        except Exception:  # noqa: BLE001
            self._gl_failed = True
            logger.exception(
                "OrbGLWidget: render failed; fallback active")
            self._paint_fallback()

    def _paint_fallback(self) -> None:
        """2D placeholder used only when the GL context is unavailable.

        Keeps the HUD usable; the failure is logged so this is never
        mistaken for the real 3D orb.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        cx, cy = rect.center().x(), rect.center().y()
        r = min(rect.width(), rect.height()) * 0.44
        grad = QRadialGradient(cx, cy, r)
        grad.setColorAt(0.0, QColor(120, 230, 255, 200))
        grad.setColorAt(0.4, QColor(20, 110, 200, 120))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(rect, QColor(0, 0, 0, 0))
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        painter.setPen(QColor(120, 230, 255, 160))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         "3D RENDERER UNAVAILABLE")
        painter.end()

    # ------------------------------------------------------------------
    @staticmethod
    def _perspective(fov_deg: float, aspect: float,
                     znear: float, zfar: float) -> QMatrix4x4:
        m = QMatrix4x4()
        m.perspective(fov_deg, aspect, znear, zfar)
        return m

    @staticmethod
    def _look_at(eye, center, up) -> QMatrix4x4:
        m = QMatrix4x4()
        m.lookAt(QVector3D(*eye), QVector3D(*center), QVector3D(*up))
        return m

    @staticmethod
    def _rot(euler_deg, extra_y_deg: float = 0.0) -> QMatrix4x4:
        m = QMatrix4x4()
        m.rotate(euler_deg[0], 1.0, 0.0, 0.0)
        m.rotate(euler_deg[1], 0.0, 1.0, 0.0)
        m.rotate(euler_deg[2], 0.0, 0.0, 1.0)
        if extra_y_deg:
            m.rotate(extra_y_deg, 0.0, 1.0, 0.0)
        return m

    def _set_f(self, prog, name: str, value: float) -> None:
        """Set a float uniform via glUniform1f (PySide6's
        setUniformValue float overload is unreliable for integral
        values like 0.0 / -1.0)."""
        loc = prog.uniformLocation(name)
        if loc >= 0 and self._gl is not None:
            self._gl.glUniform1f(loc, float(value))

    # GL constants (PySide6's QOpenGLFunctions does not export them).
    _GL_DEPTH_TEST = 0x0B71
    _GL_BLEND = 0x0BE2
    _GL_SRC_ALPHA = 0x0302
    _GL_ONE = 0x0001
    _GL_COLOR_BUFFER_BIT = 0x00004000
    _GL_DEPTH_BUFFER_BIT = 0x00000100
    _GL_POINTS = 0x0000
    _GL_TRIANGLES = 0x0004
    _GL_UNSIGNED_INT = 0x1405

    def _render(self) -> None:
        gl = self._gl
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClearDepth(1.0)
        gl.glEnable(self._GL_DEPTH_TEST)
        gl.glEnable(self._GL_BLEND)
        gl.glBlendFunc(self._GL_SRC_ALPHA, self._GL_ONE)  # additive glow
        gl.glClear(self._GL_COLOR_BUFFER_BIT | self._GL_DEPTH_BUFFER_BIT)

        w, h = self.width(), max(1, self.height())
        proj = self._perspective(50.0, w / h, 0.1, 100.0)

        # Gentle cinematic camera sway around the orb.
        t = self._time
        cam_pos = (
            math.sin(t * 0.11) * 0.35,
            math.sin(t * 0.07) * 0.18,
            5.2 + math.sin(t * 0.09) * 0.25,
        )
        view = self._look_at(cam_pos, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        mvp_base = proj * view

        energy = self._MODE_ENERGY[self._mode]
        if self._speaking:
            energy = min(1.0, energy + 0.15)
        level = self._level
        pixel_ratio = float(self.devicePixelRatioF())
        mode_speed = self._MODE_SPEED[self._mode]
        ring_flow = self._MODE_FLOW[self._mode] * (0.6 + level)
        error = 1.0 if self._mode == "error" else 0.0

        # Appearance uniforms: saved values x smoothed voice-state gain.
        app = self._appearance
        gain_b = app["brightness"] * self._gain_b
        gain_g = app["glow"] * self._gain_g
        opacity = app["opacity"]

        # --- particles ---------------------------------------------------
        prog = self._prog_point
        prog.bind()
        self._vao_points.bind()
        prog.setUniformValue("uMVP", mvp_base)
        self._set_f(prog, "uTime", t)
        self._set_f(prog, "uAudio", level)
        self._set_f(prog, "uPixelRatio", pixel_ratio)
        self._set_f(prog, "uBrightness", gain_b * app["particle"])
        self._set_f(prog, "uOpacity", opacity)
        gl.glDrawArrays(self._GL_POINTS, 0, self._PARTICLE_COUNT)
        self._vao_points.release()
        prog.release()

        # --- rings -------------------------------------------------------
        prog = self._prog_ring
        prog.bind()
        for i, ((major, minor, tilt, r_speed, color, base_e),
                vao, vert_count) in enumerate(
            zip(self._RINGS, self._vao_rings, self._ring_vert_counts)
        ):
            vao.bind()
            rot_deg = t * r_speed * mode_speed * 21.0
            model = self._rot(tilt, rot_deg)
            mvp = mvp_base * model
            prog.setUniformValue("uMVP", mvp)
            prog.setUniformValue("uModel", model)
            prog.setUniformValue("uCameraPos", QVector3D(*cam_pos))
            self._set_f(prog, "uTime", t + i * 1.7)
            self._set_f(prog, "uAudio", level)
            self._set_f(
                prog, "uEnergy",
                min(1.0, base_e * energy / 0.6 + level * 0.3))
            self._set_f(prog, "uSpeed", mode_speed)
            self._set_f(prog, "uFlow", ring_flow)
            self._set_f(prog, "uError", error)
            self._set_f(prog, "uBrightness", gain_b * app["ring"])
            self._set_f(prog, "uOpacity", opacity)
            prog.setUniformValue(
                "uColor", QVector3D(*color))
            gl.glDrawArrays(self._GL_TRIANGLES, 0, vert_count)
            vao.release()
        prog.release()

        # --- core sphere -------------------------------------------------
        prog = self._prog_sphere
        prog.bind()
        self._vao_sphere.bind()
        breathe = 1.0 + level * 0.10 + math.sin(t * 1.4) * 0.012
        model = QMatrix4x4()
        model.scale(breathe * (1.0 + self._pulse * 0.18),
                    breathe * (1.0 + self._pulse * 0.18),
                    breathe * (1.0 + self._pulse * 0.18))
        mvp = mvp_base * model
        prog.setUniformValue("uMVP", mvp)
        prog.setUniformValue("uModel", model)
        prog.setUniformValue("uCameraPos", QVector3D(*cam_pos))
        self._set_f(prog, "uTime", t)
        self._set_f(prog, "uAudio", level)
        self._set_f(prog, "uEnergy", energy)
        self._set_f(prog, "uPulseAngle", self._pulse_angle)
        self._set_f(prog, "uPulseStrength", self._pulse)
        self._set_f(
            prog, "uScan",
            (t * 0.55) % 1.0 if self._mode == "processing" else -1.0)
        self._set_f(prog, "uError", error)
        self._set_f(prog, "uBrightness", gain_b)
        self._set_f(prog, "uGlow", gain_g)
        self._set_f(prog, "uOpacity", opacity)
        prog.setUniformValue("uTint", QVector3D(0.55, 0.82, 1.0))
        gl.glDrawArrays(self._GL_TRIANGLES, 0, self._sphere_vert_count)
        self._vao_sphere.release()
        prog.release()

        gl.glDisable(self._GL_BLEND)
        gl.glDisable(self._GL_DEPTH_TEST)
