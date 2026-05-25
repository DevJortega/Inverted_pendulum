"""
graficos.py
-----------
Funciones de cálculo y visualización para el dashboard de control del péndulo invertido.
Planta (FT para análisis): G(s) = 0.01209 / (0.002846·s² − 0.09678)
Simulación: espacio de estados completo de 4 estados [x, ẋ, θ, θ̇]
"""

import math
import base64
import io
import numpy as np
import control as ct
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image as PILImage, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constantes de estilo (gráficas Plotly)
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_white"
COLOR_MAIN      = "#2563eb"
COLOR_REF       = "#dc2626"
COLORS_OVERLAY  = ["#16a34a", "#ea580c", "#9333ea", "#0891b2", "#ca8a04", "#db2777"]

# ---------------------------------------------------------------------------
# Parámetros físicos reales del sistema
# ---------------------------------------------------------------------------
M_CART = 0.723        # masa del carro   [kg]
M_PEND = 0.093        # masa del péndulo [kg]
L_PEND = 0.26         # longitud varilla  [m]
G_GRAV = 9.81         # gravedad          [m/s²]
I_PEND = (1/3) * M_PEND * L_PEND**2   # inercia respecto al pivote [kg·m²]

# Delta del sistema (denominador del desacoplamiento)
_DELTA = (M_CART + M_PEND) * (I_PEND + M_PEND * L_PEND**2) - (M_PEND * L_PEND)**2

# ---------------------------------------------------------------------------
# Matrices de espacio de estados completo [x, ẋ, θ, θ̇]
# Derivadas de las ecs. linealizadas del libro (ecs. 3-16 y 3-17)
# ---------------------------------------------------------------------------
A_SS = np.array([
    [0,  1,                                                    0,  0],
    [0,  0,  -(M_PEND**2 * G_GRAV * L_PEND**2) / _DELTA,     0],
    [0,  0,                                                    0,  1],
    [0,  0,   (M_CART + M_PEND) * M_PEND * G_GRAV * L_PEND / _DELTA, 0],
])

B_SS = np.array([
    [0],
    [(I_PEND + M_PEND * L_PEND**2) / _DELTA],
    [0],
    [-(M_PEND * L_PEND) / _DELTA],
])

# C para medir θ (índice 2 del estado)
C_SS = np.array([[0, 0, 1, 0]])
D_SS = np.array([[0]])

# ---------------------------------------------------------------------------
# Planta (FT de θ respecto a u) — para análisis en frecuencia
# ---------------------------------------------------------------------------
NUM   = [0.01209]
DEN   = [0.002846, 0.0, -0.09678]
PLANT = ct.tf(NUM, DEN)


# ===========================================================================
# 1. CONSTRUCCIÓN DEL SISTEMA (análisis)
# ===========================================================================

def construir_controlador(tipo, Kp, Ki, Kd):
    """Devuelve la FT del controlador C(s)."""
    if tipo == "Lazo Abierto":
        return ct.tf([1], [1])
    if tipo == "P":
        return ct.tf([Kp], [1])
    if tipo == "PI":
        return ct.tf([Kp, Ki], [1, 0])
    if tipo == "PD":
        N = 100
        return ct.tf([Kd * N + Kp, Kp * N], [1, N])
    if tipo == "PID":
        N = 100
        num = [Kp + Kd * N, Kp * N + Ki, Ki * N]
        den = [1, N, 0]
        return ct.tf(num, den)
    return ct.tf([1], [1])


def sistema_lazo_cerrado(tipo, Kp, Ki, Kd):
    """Devuelve (T, C): FT lazo cerrado y controlador."""
    C = construir_controlador(tipo, Kp, Ki, Kd)
    if tipo == "Lazo Abierto":
        return PLANT, C
    T = ct.feedback(C * PLANT, 1)
    return T, C


# ===========================================================================
# 2. MÉTRICAS
# ===========================================================================

def calcular_metricas_impulso(T, t, y):
    """Métricas de la respuesta al impulso."""
    try:
        estable = bool(np.all(np.real(ct.poles(T)) < 0))
    except Exception:
        estable = False
    if not estable or not np.all(np.isfinite(y)):
        return {"estable": False, "pico": None, "t_pico": None,
                "Ts": None, "y_final": None}
    idx_pico = np.argmax(np.abs(y))
    pico     = y[idx_pico]
    t_pico   = t[idx_pico]
    banda    = 0.02 * abs(pico)
    fuera    = np.where(np.abs(y) > banda)[0]
    Ts       = t[fuera[-1]] if len(fuera) > 0 else 0.0
    return {"estable": True, "pico": pico, "t_pico": t_pico,
            "Ts": Ts, "y_final": y[-1]}


def calcular_margenes(L):
    """Devuelve (GM_dB, PM, wcg, wcp)."""
    try:
        gm, pm, wcg, wcp = ct.margin(L)
        gm_db = 20 * np.log10(gm) if (gm is not None and gm > 0 and np.isfinite(gm)) else None
        pm_v  = pm  if (pm  is not None and np.isfinite(pm))  else None
        wcg_v = wcg if (wcg is not None and np.isfinite(wcg)) else None
        wcp_v = wcp if (wcp is not None and np.isfinite(wcp)) else None
        return gm_db, pm_v, wcg_v, wcp_v
    except Exception:
        return None, None, None, None


# ===========================================================================
# 3. CÁLCULO DE RESPUESTAS
# ===========================================================================

def respuesta_condicion_inicial_completa(tipo, Kp, Ki, Kd,
                                         theta0_deg=15.0, t_final=5.0, n_pts=600):
    """
    Simula θ(t) desde θ₀=theta0_deg usando el modelo completo de 4 estados
    [x, ẋ, θ, θ̇] con RK4 y el controlador tipo/Kp/Ki/Kd en lazo cerrado.
    Idéntico al loop de simulación del carro — permite comparar directamente.
    Devuelve (t, theta_grados).
    """
    dt = t_final / n_pts
    x  = np.array([0.0, 0.0, np.radians(theta0_deg), 0.0])
    integral   = 0.0
    error_prev = 0.0

    t_hist     = []
    theta_hist = []

    for i in range(n_pts):
        theta = x[2]
        error = theta   # igual que el loop del carro (u>0 estabiliza theta>0)
        P = Kp * error
        if tipo in ("PI", "PID"):
            integral = float(np.clip(integral + error * dt, -50.0, 50.0))
            I = Ki * integral
        else:
            I = 0.0
        D = Kd * (error - error_prev) / dt if tipo in ("PD", "PID") and dt > 0 else 0.0
        u = float(np.clip(P + I + D, -50.0, 50.0))
        error_prev = error

        def f(xx):
            return (A_SS @ xx.reshape(-1, 1) + B_SS * u).flatten()
        k1 = f(x)
        k2 = f(x + 0.5*dt*k1)
        k3 = f(x + 0.5*dt*k2)
        k4 = f(x + dt*k3)
        x  = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        t_hist.append((i + 1) * dt)
        theta_hist.append(np.degrees(x[2]))

        if abs(np.degrees(x[2])) > 89.0:
            for j in range(i + 1, n_pts):
                t_hist.append((j + 1) * dt)
                theta_hist.append(np.nan)
            break

    return np.array(t_hist), np.array(theta_hist)


def respuesta_condicion_inicial(T, t_final=5.0, n_pts=600, theta0_deg=15.0):
    """
    Respuesta a condición inicial θ₀=theta0_deg con el sistema en lazo cerrado T(s).
    Usa ct.initial_response() — equivale a lo que se ve en la simulación del carro.
    """
    try:
        t = np.linspace(0, t_final, n_pts)
        T_ss = ct.tf2ss(T)
        n_states = T_ss.A.shape[0]
        C = np.asarray(T_ss.C).flatten()
        A = np.asarray(T_ss.A)

        # Construye la matriz de observabilidad [C; CA; CA²; ...] y resuelve
        # para x0 tal que y(0)=theta0_rad y todas las derivadas iniciales = 0.
        # Necesario porque los estados canónicos de ct.tf2ss no son θ directamente.
        rows = [C.copy()]
        for _ in range(n_states - 1):
            rows.append(rows[-1] @ A)
        M = np.vstack(rows)
        rhs = np.zeros(n_states)
        rhs[0] = np.radians(theta0_deg)
        x0, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        t_out, y_out = ct.initial_response(T_ss, T=t, X0=x0)
        y_out = np.degrees(np.asarray(y_out).flatten())
        y_out = np.clip(y_out, -1e6, 1e6)
        return t_out, y_out
    except Exception:
        return np.linspace(0, t_final, n_pts), np.full(n_pts, np.nan)


# ===========================================================================
# 4. FIGURAS DE ANÁLISIS (Plotly)
# ===========================================================================

def fig_condicion_inicial(curvas, t_final=5.0, height=300):
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=COLOR_REF, width=1.5, dash="dot"),
                  annotation_text="Ref", annotation_position="top right")
    for c in curvas:
        fig.add_trace(go.Scatter(
            x=c["t"], y=c["y"], mode="lines", name=c["nombre"],
            line=dict(color=c["color"], width=2.2, dash=c.get("dash", "solid")),
            hovertemplate="t=%{x:.3f}s<br>θ=%{y:.2f}°<extra></extra>",
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Respuesta a Condición Inicial — Modelo Completo (θ₀=15°)", font=dict(size=14)),
        xaxis_title="Tiempo (s)", yaxis_title="θ (°)",
        height=height, margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45,
                    xanchor="center", x=0.5, font=dict(size=10)),
    )
    fig.update_xaxes(range=[0, t_final])
    return fig


def fig_root_locus(sistemas, height=300):
    fig = go.Figure()
    all_re, all_im = [], []
    for s in sistemas:
        try:
            L = s["C"] * PLANT
            try:
                rldata = ct.root_locus_map(L)
                roots  = rldata.loci
            except AttributeError:
                roots, _ = ct.root_locus(L, plot=False)
            for i in range(roots.shape[1]):
                fig.add_trace(go.Scatter(
                    x=np.real(roots[:, i]), y=np.imag(roots[:, i]),
                    mode="lines",
                    line=dict(color=s["color"], width=1.8, dash=s.get("dash","solid")),
                    name=s["nombre"], showlegend=(i == 0),
                    legendgroup=s["nombre"],
                ))
                all_re.extend(np.real(roots[:, i]).tolist())
                all_im.extend(np.imag(roots[:, i]).tolist())
            try:
                polos_lc = ct.poles(ct.feedback(L, 1))
                fig.add_trace(go.Scatter(
                    x=np.real(polos_lc), y=np.imag(polos_lc), mode="markers",
                    marker=dict(symbol="square", size=10, color=s["color"],
                                line=dict(color="white", width=1)),
                    name=f"Polos LC {s['nombre']}",
                    legendgroup=s["nombre"], showlegend=False,
                ))
            except Exception:
                pass
        except Exception:
            continue
    try:
        pg = ct.poles(PLANT)
        fig.add_trace(go.Scatter(
            x=np.real(pg), y=np.imag(pg), mode="markers",
            marker=dict(symbol="x", size=14, color="#1a2332", line=dict(width=3)),
            name="Polos planta",
        ))
    except Exception:
        pass
    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    if all_re and all_im:
        re_sp = max(max(all_re)-min(all_re), 1.0)
        im_sp = max(max(all_im)-min(all_im), 1.0)
        fig.update_xaxes(range=[min(all_re)-0.15*re_sp, max(all_re)+0.15*re_sp])
        fig.update_yaxes(range=[min(all_im)-0.15*im_sp, max(all_im)+0.15*im_sp])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Lugar Geométrico de las Raíces", font=dict(size=14)),
        xaxis_title="Re", yaxis_title="Im",
        height=height, margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45,
                    xanchor="center", x=0.5, font=dict(size=10)),
    )
    return fig


def fig_bode(sistemas, height=300):
    omega = np.logspace(0, 10, 2000)
    fig   = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                          subplot_titles=("Magnitud (dB)", "Fase (°)"))
    margenes_actual = (None, None, None, None)
    for idx, s in enumerate(sistemas):
        try:
            L = s["C"] * PLANT
            mag, phase, w = ct.frequency_response(L, omega)
            mag_db    = 20 * np.log10(np.maximum(np.asarray(mag).flatten(), 1e-12))
            phase_deg = np.degrees(np.asarray(phase).flatten())
            fig.add_trace(go.Scatter(x=w, y=mag_db, mode="lines",
                line=dict(color=s["color"], width=2, dash=s.get("dash","solid")),
                name=s["nombre"], legendgroup=s["nombre"],
                hovertemplate="ω=%{x:.4g} rad/s<br>%{y:.2f} dB<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=w, y=phase_deg, mode="lines",
                line=dict(color=s["color"], width=2, dash=s.get("dash","solid")),
                name=s["nombre"], legendgroup=s["nombre"], showlegend=False,
                hovertemplate="ω=%{x:.4g} rad/s<br>%{y:.2f}°<extra></extra>",
            ), row=2, col=1)
            if idx == 0:
                margenes_actual = calcular_margenes(L)
        except Exception:
            continue
    fig.add_hline(y=0,    line=dict(color="#94a3b8", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=-180, line=dict(color="#94a3b8", dash="dash", width=1), row=2, col=1)
    gm_db, pm_val, wcg, wcp = margenes_actual
    if wcp is not None and pm_val is not None:
        fig.add_vline(x=wcp, line=dict(color=COLOR_REF, dash="dot", width=1.3), row=1, col=1)
        fig.add_vline(x=wcp, line=dict(color=COLOR_REF, dash="dot", width=1.3), row=2, col=1)
        fig.add_annotation(x=np.log10(wcp), y=-180+pm_val,
            text=f"<b>MF={pm_val:.1f}°</b>", showarrow=True, arrowhead=2, ax=35, ay=-25,
            bgcolor="rgba(220,38,38,0.9)", font=dict(color="white", size=10),
            xref="x2", yref="y2")
    if wcg is not None and gm_db is not None:
        fig.add_vline(x=wcg, line=dict(color="#16a34a", dash="dot", width=1.3), row=1, col=1)
        fig.add_vline(x=wcg, line=dict(color="#16a34a", dash="dot", width=1.3), row=2, col=1)
        fig.add_annotation(x=np.log10(wcg), y=-gm_db,
            text=f"<b>MG={gm_db:.1f} dB</b>", showarrow=True, arrowhead=2, ax=35, ay=25,
            bgcolor="rgba(22,163,74,0.9)", font=dict(color="white", size=10),
            xref="x", yref="y")
    fig.update_xaxes(type="log", range=[0, 10], autorange=False, row=1, col=1)
    fig.update_xaxes(type="log", range=[0, 10], autorange=False,
                     title_text="ω (rad/s)", row=2, col=1)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30,
                    xanchor="center", x=0.5, font=dict(size=10)))
    return fig, gm_db, pm_val, wcg, wcp


def fig_polos_lc(sistemas, height=300):
    fig = go.Figure()
    all_re, all_im = [0], [0]
    for s in sistemas:
        try:
            T     = ct.feedback(s["C"] * PLANT, 1)
            polos = ct.poles(T)
            ceros = ct.zeros(T)
            fig.add_trace(go.Scatter(
                x=np.real(polos), y=np.imag(polos), mode="markers",
                marker=dict(symbol="x", size=14, color=s["color"], line=dict(width=3)),
                name=f"Polos {s['nombre']}", legendgroup=s["nombre"],
            ))
            if len(ceros) > 0:
                fig.add_trace(go.Scatter(
                    x=np.real(ceros), y=np.imag(ceros), mode="markers",
                    marker=dict(symbol="circle-open", size=12, color=s["color"],
                                line=dict(width=2.5)),
                    name=f"Ceros {s['nombre']}",
                    legendgroup=s["nombre"], showlegend=False,
                ))
            all_re.extend(np.real(polos).tolist())
            all_im.extend(np.imag(polos).tolist())
        except Exception:
            continue
    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    if all_re:
        re_sp = max(max(all_re)-min(all_re), 1.0)
        im_sp = max(max(all_im)-min(all_im), 1.0)
        fig.update_xaxes(range=[min(all_re)-0.2*re_sp, max(all_re)+0.2*re_sp])
        fig.update_yaxes(range=[min(all_im)-0.2*im_sp, max(all_im)+0.2*im_sp])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Polos y Ceros de Lazo Cerrado", font=dict(size=14)),
        xaxis_title="Re", yaxis_title="Im",
        height=height, margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45,
                    xanchor="center", x=0.5, font=dict(size=10)),
    )
    return fig


# ===========================================================================
# 5. ANIMACIÓN DEL CARRO (estilo CartPole con PIL → base64 → HTML)
# ===========================================================================

# Dimensiones del canvas de animación
CANVAS_W = 700
CANVAS_H = 320
_SCALE   = 130   # píxeles por metro (para la posición x del carro)

# Colores estilo CartPole profesional
_C_BG       = (248, 250, 252)   # fondo
_C_TRACK    = (51,  65,  85)    # pista
_C_TRACK_L  = (148, 163, 184)   # hash pista
_C_CART     = (37,  99,  235)   # carro azul
_C_CART_OUT = (30,  58,  138)   # borde carro
_C_WHEEL    = (30,  41,  59)    # ruedas
_C_POLE     = (220, 38,  38)    # varilla roja
_C_BALL     = (153, 27,  27)    # bola
_C_PIVOT    = (251, 191, 70)    # pivote amarillo
_C_REF      = (203, 213, 225)   # línea de referencia


def _pil_a_base64(img: PILImage.Image) -> str:
    """Convierte imagen PIL a base64 PNG para incrustar en HTML."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def dibujar_frame(x_carro: float, theta_rad: float,
                  ancho: int = CANVAS_W, alto: int = CANVAS_H) -> PILImage.Image:
    """
    Dibuja un frame PIL del péndulo invertido estilo CartPole.

    Parámetros
    ----------
    x_carro   : posición del carro en metros (0 = centro)
    theta_rad : ángulo del péndulo en radianes (0 = vertical arriba)
    """
    img  = PILImage.new("RGB", (ancho, alto), _C_BG)
    draw = ImageDraw.Draw(img)

    # --- Geometría ---
    cart_w  = 90          # px ancho carro
    cart_h  = 40          # px alto carro
    wheel_r = 10          # px radio rueda
    pole_px = int(L_PEND * _SCALE * 2.2)   # longitud visual varilla en px

    # Centro vertical de la pista
    track_y = int(alto * 0.68)

    # El carro siempre en el centro del canvas; el mundo se desplaza con él
    cart_cx = ancho // 2
    cart_top    = track_y - cart_h
    cart_bottom = track_y

    # Cargar fuentes aquí (necesarias para las marcas de pista)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 15)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # --- Pista ---
    draw.line([(0, track_y + wheel_r + 2), (ancho, track_y + wheel_r + 2)],
              fill=_C_TRACK, width=4)
    for xi in range(0, ancho, 28):
        draw.line([(xi, track_y + wheel_r + 2), (xi - 10, track_y + wheel_r + 14)],
                  fill=_C_TRACK_L, width=2)

    # Línea de referencia x=0 — se desplaza relativamente al carro
    ref_cx = int(ancho // 2 - x_carro * _SCALE)
    draw.line([(ref_cx, track_y - cart_h - 8), (ref_cx, track_y + wheel_r + 20)],
              fill=_C_REF, width=2)

    # Marcas de posición cada 0.5 m (se desplazan con la cámara)
    for metro in range(-10, 11):
        marca_x = int(ancho // 2 - x_carro * _SCALE + metro * _SCALE * 0.5)
        if 0 < marca_x < ancho:
            draw.line([(marca_x, track_y + wheel_r + 2),
                       (marca_x, track_y + wheel_r + 10)],
                      fill=_C_TRACK, width=2)
            if metro % 2 == 0:
                draw.text((marca_x - 8, track_y + wheel_r + 12),
                          f"{metro * 0.5:.1f}m", fill=_C_TRACK, font=font_sm)

    # --- Carro (rectángulo redondeado aproximado) ---
    l = cart_cx - cart_w // 2
    r = cart_cx + cart_w // 2
    t = cart_top
    b = cart_bottom
    draw.rectangle([l, t, r, b], fill=_C_CART, outline=_C_CART_OUT, width=2)

    # Detalle: franja horizontal en el carro
    mid_y = (t + b) // 2
    draw.line([(l + 4, mid_y), (r - 4, mid_y)], fill=_C_CART_OUT, width=1)

    # --- Ruedas ---
    for wx in [l + 18, r - 18]:
        wy = cart_bottom
        draw.ellipse([wx - wheel_r, wy, wx + wheel_r, wy + wheel_r * 2],
                     fill=_C_WHEEL, outline=(20, 20, 20), width=1)
        # Eje de la rueda
        draw.ellipse([wx - 3, wy + wheel_r - 3, wx + 3, wy + wheel_r + 3],
                     fill=(150, 150, 150))

    # --- Pivote del péndulo (centro superior del carro) ---
    pivot_px = cart_cx
    pivot_py = cart_top

    # --- Punta de la varilla ---
    tip_px = int(pivot_px + pole_px * math.sin(theta_rad))
    tip_py = int(pivot_py - pole_px * math.cos(theta_rad))

    # Sombra suave de la varilla
    draw.line([(pivot_px + 2, pivot_py + 2), (tip_px + 2, tip_py + 2)],
              fill=(200, 150, 150), width=8)
    # Varilla principal
    draw.line([(pivot_px, pivot_py), (tip_px, tip_py)],
              fill=_C_POLE, width=9)

    # --- Extremo superior de la varilla (redondeado) ---
    draw.ellipse([tip_px - 5, tip_py - 5,
                  tip_px + 5, tip_py + 5],
                 fill=_C_POLE, outline=_C_POLE)

    # --- Pivote ---
    draw.ellipse([pivot_px - 7, pivot_py - 7,
                  pivot_px + 7, pivot_py + 7],
                 fill=_C_PIVOT, outline=(120, 80, 10), width=1)

    # --- Indicador de ángulo (arco pequeño) ---
    if abs(theta_rad) > 0.01:
        for ang in np.linspace(math.pi/2, math.pi/2 - theta_rad, 20):
            ax = int(pivot_px + 28 * math.cos(ang))
            ay = int(pivot_py - 28 * math.sin(ang))
            draw.ellipse([ax-2, ay-2, ax+2, ay+2], fill=(100, 150, 255))

    # --- HUD: ángulo y posición ---
    theta_deg = math.degrees(theta_rad)
    hud_lines = [
        f"θ = {theta_deg:+.2f}°",
        f"x = {x_carro:+.3f} m",
    ]
    hud_x, hud_y = 10, 10
    for line in hud_lines:
        draw.text((hud_x + 1, hud_y + 1), line, fill=(180, 180, 180), font=font)
        draw.text((hud_x, hud_y), line, fill=(30, 41, 59), font=font)
        hud_y += 20

    # Aviso zona lineal
    if abs(theta_deg) > 15:
        warn = "⚠ Fuera zona lineal" if abs(theta_deg) <= 30 else "⚠ CAÍDA"
        draw.text((ancho // 2 - 70, 8), warn,
                  fill=(220, 38, 38) if abs(theta_deg) > 30 else (180, 100, 0),
                  font=font)

    return img


def frame_a_html(x_carro: float, theta_rad: float) -> str:
    """
    Genera HTML con el frame PIL incrustado como base64.
    Permite actualizar la imagen con st.empty() sin recrear el componente.
    """
    img = dibujar_frame(x_carro, theta_rad)
    b64 = _pil_a_base64(img)
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;border-radius:8px;display:block;" />'
    )


# ===========================================================================
# 6. GRÁFICA θ(t) en tiempo real (Plotly — igual que antes)
# ===========================================================================

def figura_theta_tiempo(t_hist, theta_hist, x_hist=None, ventana=10.0, height=260):
    """
    Gráfica θ(t) y opcionalmente x(t) durante la simulación.
    x_hist: lista de posiciones del carro [m], opcional.
    """
    if x_hist is not None and len(x_hist) > 0:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("θ (°)", "x carro (m)"))
    else:
        fig = go.Figure()

    # Líneas de referencia
    if x_hist is not None:
        fig.add_hline(y=0,   line=dict(color=COLOR_REF,   width=1.5, dash="dot"), row=1, col=1)
        fig.add_hline(y=15,  line=dict(color="#f59e0b",   width=1,   dash="dash"), row=1, col=1)
        fig.add_hline(y=-15, line=dict(color="#f59e0b",   width=1,   dash="dash"), row=1, col=1)
        fig.add_hline(y=0,   line=dict(color="#64748b",   width=1,   dash="dot"),  row=2, col=1)
    else:
        fig.add_hline(y=0,   line=dict(color=COLOR_REF,   width=1.5, dash="dot"))
        fig.add_hline(y=15,  line=dict(color="#f59e0b",   width=1,   dash="dash"))
        fig.add_hline(y=-15, line=dict(color="#f59e0b",   width=1,   dash="dash"))

    if len(t_hist) > 0:
        t_max = max(t_hist[-1], ventana)
        x_min = max(0, t_max - ventana)
        x_max = t_max + 0.5

        if x_hist is not None:
            fig.add_trace(go.Scatter(
                x=t_hist, y=theta_hist, mode="lines",
                line=dict(color=COLOR_MAIN, width=2.2), name="θ(t)",
                hovertemplate="t=%{x:.2f}s<br>θ=%{y:.2f}°<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=t_hist, y=x_hist, mode="lines",
                line=dict(color="#16a34a", width=2.0), name="x(t)",
                hovertemplate="t=%{x:.2f}s<br>x=%{y:.3f}m<extra></extra>",
            ), row=2, col=1)
            fig.update_xaxes(range=[x_min, x_max], row=1, col=1)
            fig.update_xaxes(range=[x_min, x_max], row=2, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=t_hist, y=theta_hist, mode="lines",
                line=dict(color=COLOR_MAIN, width=2.2), name="θ(t)",
                hovertemplate="t=%{x:.2f}s<br>θ=%{y:.2f}°<extra></extra>",
            ))
            fig.update_xaxes(range=[x_min, x_max])
    else:
        x_min, x_max = 0, ventana
        if x_hist is None:
            fig.update_xaxes(range=[x_min, x_max])

    if x_hist is not None:
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=height,
            margin=dict(l=50, r=20, t=35, b=40),
            showlegend=False,
        )
    else:
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=dict(text="Ángulo del péndulo θ(t)", font=dict(size=13)),
            xaxis_title="Tiempo (s)", yaxis_title="θ (°)",
            height=height, margin=dict(l=50, r=20, t=35, b=40),
            showlegend=False,
        )
    return fig


# ===========================================================================
# 7. LÓGICA PID DISCRETO
# ===========================================================================

def pid_step(error, dt, Kp, Ki, Kd, tipo, integral, error_prev):
    """
    PID discreto con anti-windup.
    Devuelve (u, integral_actualizado, error_prev_no_actualizado).
    El caller debe hacer: error_prev = error después de la llamada.
    """
    P = Kp * error if tipo != "Lazo Abierto" else 0.0

    if tipo in ("PI", "PID"):
        integral = float(np.clip(integral + error * dt, -50.0, 50.0))
        I = Ki * integral
    else:
        integral = 0.0
        I = 0.0

    D = Kd * (error - error_prev) / dt if (tipo in ("PD", "PID") and dt > 0) else 0.0

    u = float(np.clip(P + I + D, -50.0, 50.0))
    return u, integral, error_prev


# ===========================================================================
# 8. AUTO-SINTONIZACIÓN (scipy differential_evolution)
# ===========================================================================

def auto_sintonizar(tipo, A, B, theta0_deg=15.0, t_final=5.0, saturacion=50.0):
    """
    Encuentra las ganancias óptimas (Kp, Ki, Kd) para el controlador `tipo`
    minimizando el tiempo de asentamiento al 2% desde θ₀=theta0_deg,
    usando el modelo lineal completo de 4 estados simulado con RK4.

    Devuelve dict con keys: 'Kp', 'Ki', 'Kd', 'Ts', 'exito'
    """
    from scipy.optimize import differential_evolution

    dt = 0.005
    n_steps = int(t_final / dt)
    theta0 = np.radians(theta0_deg)
    x0 = np.array([0.0, 0.0, theta0, 0.0])

    if tipo == "P":
        bounds = [(0.1, 200.0)]
    elif tipo == "PI":
        bounds = [(0.1, 200.0), (0.0, 200.0)]
    elif tipo == "PD":
        bounds = [(0.1, 200.0), (0.0, 30.0)]
    elif tipo == "PID":
        bounds = [(0.1, 200.0), (0.0, 200.0), (0.0, 30.0)]
    else:
        return {"Kp": 0.0, "Ki": 0.0, "Kd": 0.0, "Ts": None, "exito": False}

    def simular(params):
        if tipo == "P":
            Kp, Ki, Kd = params[0], 0.0, 0.0
        elif tipo == "PI":
            Kp, Ki, Kd = params[0], params[1], 0.0
        elif tipo == "PD":
            Kp, Ki, Kd = params[0], 0.0, params[1]
        elif tipo == "PID":
            Kp, Ki, Kd = params[0], params[1], params[2]

        x = x0.copy()
        integral = 0.0
        error_prev = 0.0
        theta_hist = []

        for _ in range(n_steps):
            theta = x[2]
            error = theta   # u>0 estabiliza theta>0 (B[3,0]<0)
            P = Kp * error
            integral = np.clip(integral + error * dt, -50.0, 50.0)
            I = Ki * integral
            D = Kd * (error - error_prev) / dt
            u = np.clip(P + I + D, -saturacion, saturacion)
            error_prev = error

            def f(xx):
                return (A @ xx.reshape(-1, 1) + B * u).flatten()
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            theta_hist.append(x[2])

            if abs(np.degrees(x[2])) > 89.0:
                return 1e6

        theta_arr = np.array(theta_hist)
        banda = 0.02 * abs(theta0)
        fuera = np.where(np.abs(theta_arr) > banda)[0]
        Ts = fuera[-1] * dt if len(fuera) > 0 else 0.0
        return Ts

    resultado = differential_evolution(
        simular, bounds=bounds,
        maxiter=80, popsize=10, tol=1e-3,
        seed=42, workers=1, polish=True
    )

    if tipo == "P":
        Kp, Ki, Kd = resultado.x[0], 0.0, 0.0
    elif tipo == "PI":
        Kp, Ki, Kd = resultado.x[0], resultado.x[1], 0.0
    elif tipo == "PD":
        Kp, Ki, Kd = resultado.x[0], 0.0, resultado.x[1]
    elif tipo == "PID":
        Kp, Ki, Kd = resultado.x[0], resultado.x[1], resultado.x[2]

    Ts_final = resultado.fun
    exito = Ts_final < 1e5

    return {"Kp": round(Kp, 3), "Ki": round(Ki, 3), "Kd": round(Kd, 3),
            "Ts": round(Ts_final, 4) if exito else None, "exito": exito}