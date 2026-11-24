import io
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pillow_avif
from PIL import Image

# =============================================================================
# Configuración de página
# =============================================================================
st.set_page_config(
    page_title="Compresión RGB con SVD - El Comercio",
    page_icon="🖼️",
    layout="wide",
)

# Parámetros globales alineados al proyecto (Álgebra Lineal Computacional - USIL 2026-02)
RANGOS_K_DEFAULT = [10, 25, 50, 100]
UMBRAL_PSNR_DEFAULT = 30.0
LIMITE_TIEMPO_S = 3.0

FOTOS_PRECARGADAS = {
    "Foto 1 - Política (foto1_politica.jpg)": "foto1_politica.jpg",
    "Foto 2 - Deportes (foto2_deportes.jpg)": "foto2_deportes.jpg",
    "Foto 3 - Infografía (foto3_infografia.jpg)": "foto3_infografia.jpg",
}


# =============================================================================
# Núcleo Matemático Modular (SVD Truncada por Canales RGB)
# =============================================================================
@st.cache_data(show_spinner=False)
def cargar_imagen_rgb(file_bytes: bytes, max_dim: int | None = 800) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Carga una fotografía JPG/PNG en RGB, opcionalmente redimensiona y normaliza a [0, 1] en float64."""
    peso_kb = len(file_bytes) / 1024.0
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    dims_orig = img.size  # (ancho, alto)

    if max_dim is not None and max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    A_rgb = np.array(img, dtype=np.float64) / 255.0
    return A_rgb, peso_kb, dims_orig


@st.cache_data(show_spinner=False)
def calcular_svd_rgb(A_rgb: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], float]:
    """Calcula la SVD delgada A = U * Sigma * V^T una sola vez por canal (R, G, B)."""
    t0 = time.perf_counter()
    canales_svd = []
    for c in range(3):
        U, s, Vt = np.linalg.svd(A_rgb[:, :, c], full_matrices=False)
        canales_svd.append((U, s, Vt))
    t_svd_ms = (time.perf_counter() - t0) * 1000.0
    return canales_svd, t_svd_ms


def truncar_svd_rgb(canales_svd: list[tuple], k: int) -> list[tuple]:
    """Conserva las primeras k columnas de U, valores singulares y filas de V^T."""
    rango_max = len(canales_svd[0][1])
    k = max(1, min(int(k), rango_max))
    return [(U[:, :k], s[:k], Vt[:k, :]) for (U, s, Vt) in canales_svd]


def reconstruir_rgb(canales_trunc: list[tuple]) -> np.ndarray:
    """Reconstruye Ak^(c) = (Uk * sk) @ Vtk por canal y aplica clipping a [0, 1]."""
    canales_rec = [np.clip((Uk * sk) @ Vtk, 0.0, 1.0) for (Uk, sk, Vtk) in canales_trunc]
    return np.stack(canales_rec, axis=2)


def imagen_a_jpeg_bytes(Ak_rgb: np.ndarray, quality: int = 85) -> tuple[bytes, float]:
    """Codifica la imagen reconstruida a JPEG en memoria y devuelve (bytes, peso_kb)."""
    img_out = Image.fromarray((Ak_rgb * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img_out.save(buffer, format="JPEG", quality=quality)
    jpeg_bytes = buffer.getvalue()
    return jpeg_bytes, len(jpeg_bytes) / 1024.0


def calcular_metricas_rgb(
    A_rgb: np.ndarray,
    Ak_rgb: np.ndarray,
    canales_svd: list[tuple],
    k: int,
    calcular_peso_jpeg: bool = False,
) -> tuple[dict, bytes | None]:
    """Calcula MSE, PSNR (dB), energía espectral (%), parámetros 3k(m+n+1) y peso JPEG (KB)."""
    m, n, _ = A_rgb.shape
    mse = float(np.mean((A_rgb - Ak_rgb) ** 2))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    energia = float(
        np.mean([np.sum(s[:k] ** 2) / np.sum(s ** 2) * 100.0 for (_, s, _) in canales_svd])
    )
    datos_orig = 3 * m * n
    datos_comp = 3 * k * (m + n + 1)
    compresion_param = float((1.0 - datos_comp / datos_orig) * 100.0)

    jpeg_bytes = None
    peso_rec_kb = None
    if calcular_peso_jpeg:
        jpeg_bytes, peso_rec_kb = imagen_a_jpeg_bytes(Ak_rgb, quality=85)

    metricas = {
        "k": k,
        "MSE": round(mse, 6),
        "PSNR (dB)": round(psnr, 2),
        "Energía (%)": round(energia, 2),
        "Parámetros": datos_comp,
        "Compresión Param. (%)": round(compresion_param, 2),
        "Peso Archivo (KB)": round(peso_rec_kb, 1) if peso_rec_kb is not None else "-",
    }
    return metricas, jpeg_bytes


@st.cache_data(show_spinner=False)
def buscar_k_minimo_rgb(
    A_rgb: np.ndarray,
    _canales_svd: list[tuple],
    psnr_min: float = UMBRAL_PSNR_DEFAULT,
) -> dict | None:
    """Usa búsqueda binaria para hallar el menor rango k que alcanza el umbral PSNR."""
    rango_max = len(_canales_svd[0][1])
    izq, der = 1, rango_max
    mejor_k = None

    while izq <= der:
        mid = (izq + der) // 2
        Ak_mid = reconstruir_rgb(truncar_svd_rgb(_canales_svd, mid))
        mse = float(np.mean((A_rgb - Ak_mid) ** 2))
        psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
        if psnr >= psnr_min:
            mejor_k = mid
            der = mid - 1
        else:
            izq = mid + 1

    if mejor_k is None:
        return None

    Ak_opt = reconstruir_rgb(truncar_svd_rgb(_canales_svd, mejor_k))
    metricas, _ = calcular_metricas_rgb(A_rgb, Ak_opt, _canales_svd, mejor_k, calcular_peso_jpeg=True)
    return metricas


@st.cache_data(show_spinner=False)
def calcular_curvas_psnr_energia(
    A_rgb: np.ndarray, _canales_svd: list[tuple], max_k_curva: int = 200
) -> tuple[list[int], list[float], list[float]]:
    """Calcula las series de PSNR y Energía retenida cada 5 valores de k."""
    rango_max = min(max_k_curva, len(_canales_svd[0][1]))
    ks = list(range(5, rango_max + 1, 5))
    if not ks:
        ks = [1]
    psnrs, energias = [], []
    for k in ks:
        Ak = reconstruir_rgb(truncar_svd_rgb(_canales_svd, k))
        m_dict, _ = calcular_metricas_rgb(A_rgb, Ak, _canales_svd, k, calcular_peso_jpeg=False)
        psnrs.append(m_dict["PSNR (dB)"])
        energias.append(m_dict["Energía (%)"])
    return ks, psnrs, energias


# =============================================================================
# Interfaz de Usuario (Streamlit)
# =============================================================================
st.title("Compresión de Imágenes Digitales a Color (RGB) mediante SVD Truncada")
st.markdown(
    """
    **Curso:** Álgebra Lineal Computacional (USIL - 2026-02) | **Caso de Estudio:** Portal Digital de *Empresa Editora El Comercio S.A.*  
    **Grupo 8:** Agurto Moscaiza, Wagner Paolo · Gamarra Alvarado, Mauricio Rodrigo · Llacza Arnao, Joaquín Alexander · Pacotaipe García, Roy Diego
    """
)
st.divider()

# --- Barra lateral ---
with st.sidebar:
    st.header("Configuración")

    # Detectar si el repositorio incluye las fotos locales de El Comercio
    fotos_disponibles = {
        etiqueta: ruta for etiqueta, ruta in FOTOS_PRECARGADAS.items() if os.path.exists(ruta)
    }

    fuente_opciones = ["Subir imagen (.jpg / .png)"]
    if fotos_disponibles:
        fuente_opciones.insert(0, "Usar fotografía precargada de El Comercio")

    fuente = st.radio("Fuente de imagen:", fuente_opciones)

    file_bytes = None
    nombre_imagen = ""

    if fuente == "Usar fotografía precargada de El Comercio":
        seleccion = st.selectbox("Selecciona una fotografía:", list(fotos_disponibles.keys()))
        ruta_sel = fotos_disponibles[seleccion]
        with open(ruta_sel, "rb") as f:
            file_bytes = f.read()
        nombre_imagen = os.path.basename(ruta_sel)
    else:
        archivo = st.file_uploader("Sube una fotografía (.jpg, .png, .avif):", type=["jpg", "jpeg", "png", "avif"])
        if archivo is not None:
            file_bytes = archivo.getvalue()
            nombre_imagen = archivo.name

    st.divider()
    max_dim_opcion = st.selectbox(
        "Resolución máxima de procesamiento:",
        options=[600, 800, 1000, None],
        index=1,
        format_func=lambda x: f"{x} px (recomendado)" if x == 800 else ("Original (más lento)" if x is None else f"{x} px"),
        help="Limitar el lado mayor acelera la SVD y evita exceso de memoria en la nube.",
    )

    umbral_psnr = st.slider(
        "Umbral editorial PSNR (dB):",
        min_value=20.0,
        max_value=45.0,
        value=UMBRAL_PSNR_DEFAULT,
        step=0.5,
    )

# --- Validación de imagen ---
if file_bytes is None:
    st.info(
        "Sube una fotografía desde el panel izquierdo para iniciar la descomposición SVD "
        "(o agrega `foto1_politica.jpg`, `foto2_deportes.jpg` y `foto3_infografia.jpg` al repositorio para tenerlas precargadas)."
    )
    st.stop()

# --- Carga y cálculo SVD (cacheado) ---
A_rgb, peso_orig_kb, dims_orig = cargar_imagen_rgb(file_bytes, max_dim=max_dim_opcion)
m, n, _ = A_rgb.shape
rango_max = min(m, n)

with st.spinner("Calculando SVD delgada por canales RGB..."):
    canales_svd, t_svd_ms = calcular_svd_rgb(A_rgb)

st.caption(
    f"**Archivo:** `{nombre_imagen}` | **Original:** {dims_orig[0]}×{dims_orig[1]} px ({peso_orig_kb:.1f} KB) | "
    f"**Procesada:** {n}×{m} px | **Rango máximo:** {rango_max} | **Tiempo SVD:** {t_svd_ms:.1f} ms"
)

# --- Pestañas principales ---
tab1, tab2, tab3 = st.tabs([
    "Compresión Interactiva",
    "Evaluación Experimental (Rangos k)",
    "Curvas PSNR y Energía",
])

# =============================================================================
# TAB 1: Compresión interactiva
# =============================================================================
with tab1:
    k_sel = st.slider(
        "Selecciona el rango truncado (k):",
        min_value=1,
        max_value=rango_max,
        value=min(50, rango_max),
        step=1,
    )

    t0 = time.perf_counter()
    Ak_rgb = reconstruir_rgb(truncar_svd_rgb(canales_svd, k_sel))
    t_rec_ms = (time.perf_counter() - t0) * 1000.0
    metricas_k, jpeg_bytes = calcular_metricas_rgb(A_rgb, Ak_rgb, canales_svd, k_sel, calcular_peso_jpeg=True)

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    delta_psnr = round(metricas_k["PSNR (dB)"] - umbral_psnr, 2)
    col_m1.metric("PSNR (dB)", f'{metricas_k["PSNR (dB)"]:.2f} dB', delta=f"{delta_psnr:+.2f} dB vs umbral")
    col_m2.metric("Energía Espectral", f'{metricas_k["Energía (%)"]:.2f}%')
    col_m3.metric("Compresión Param.", f'{metricas_k["Compresión Param. (%)"]:.2f}%')
    col_m4.metric("Peso JPEG (q=85)", f'{metricas_k["Peso Archivo (KB)"]} KB')
    col_m5.metric("Tiempo Rec.", f"{t_rec_ms:.1f} ms")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("Original (100%)")
        st.image(A_rgb, use_container_width=True)
    with col_img2:
        st.subheader(f"Reconstruida (k = {k_sel})")
        st.image(Ak_rgb, use_container_width=True)

    nombre_base = os.path.splitext(nombre_imagen)[0]
    st.download_button(
        label=f"Descargar imagen comprimida ({nombre_base}_k{k_sel}.jpg)",
        data=jpeg_bytes,
        file_name=f"{nombre_base}_k{k_sel}.jpg",
        mime="image/jpeg",
    )

    # Cálculo de k mínimo para el umbral editorial
    res_kmin = buscar_k_minimo_rgb(A_rgb, canales_svd, psnr_min=umbral_psnr)
    if res_kmin:
        st.success(
            f"**Rango mínimo editorial ($\text{{PSNR}} \ge {umbral_psnr:.1f}\text{{ dB}}$):** "
            f"**k = {res_kmin['k']}** | PSNR = **{res_kmin['PSNR (dB)']} dB** | "
            f"Compresión paramétrica = **{res_kmin['Compresión Param. (%)']}%** | "
            f"Peso JPEG = **{res_kmin['Peso Archivo (KB)']} KB**"
        )
    else:
        st.warning(f"No se alcanzó el umbral de {umbral_psnr:.1f} dB dentro del rango máximo ({rango_max}).")

# =============================================================================
# TAB 2: Evaluación experimental (Tabla 7 y tira comparativa)
# =============================================================================
with tab2:
    rangos_validos = [k for k in RANGOS_K_DEFAULT if k <= rango_max]
    if not rangos_validos:
        rangos_validos = [rango_max]

    filas = []
    reconstrucciones = []
    for k in rangos_validos:
        t0 = time.perf_counter()
        Ak_tmp = reconstruir_rgb(truncar_svd_rgb(canales_svd, k))
        t_rec = (time.perf_counter() - t0) * 1000.0
        fila, _ = calcular_metricas_rgb(A_rgb, Ak_tmp, canales_svd, k, calcular_peso_jpeg=True)
        fila["Tiempo Total (ms)"] = round(t_svd_ms + t_rec, 1)
        filas.append(fila)
        reconstrucciones.append((k, Ak_tmp, fila["PSNR (dB)"]))

    df_res = pd.DataFrame(filas)[
        ["k", "MSE", "PSNR (dB)", "Energía (%)", "Parámetros", "Compresión Param. (%)", "Peso Archivo (KB)", "Tiempo Total (ms)"]
    ]

    st.subheader("Métricas Cuantitativas por Rango k")
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar tabla de métricas (.csv)",
        data=csv_bytes,
        file_name=f"metricas_svd_{nombre_base}.csv",
        mime="text/csv",
    )

    st.subheader("Tira Comparativa Visual")
    fig, axes = plt.subplots(1, len(reconstrucciones) + 1, figsize=(16, 4.2))
    axes[0].imshow(A_rgb)
    axes[0].set_title("Original (100%)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for idx, (k, Ak_tmp, psnr_val) in enumerate(reconstrucciones, 1):
        axes[idx].imshow(Ak_tmp)
        axes[idx].set_title(f"k = {k}\nPSNR = {psnr_val:.2f} dB", fontsize=10)
        axes[idx].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =============================================================================
# TAB 3: Curvas PSNR y Energía
# =============================================================================
with tab3:
    st.subheader("Curvas de Calidad Visual (PSNR) y Energía Espectral Retenida")
    with st.spinner("Generando curvas de desempeño..."):
        ks, psnrs, energias = calcular_curvas_psnr_energia(A_rgb, canales_svd, max_k_curva=200)

    fig_c, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(ks, psnrs, marker="o", markersize=3, color="#1B365D", label="PSNR (dB)")
    ax1.axhline(umbral_psnr, color="red", linestyle="--", label=f"Umbral {umbral_psnr:.1f} dB")
    ax1.set_title("Calidad Visual (PSNR) vs. Rango k", fontweight="bold")
    ax1.set_xlabel("Rango k")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ks, energias, marker="s", markersize=3, color="#16A34A", label="Energía retenida (%)")
    ax2.set_title("Energía Espectral Retenida vs. Rango k", fontweight="bold")
    ax2.set_xlabel("Rango k")
    ax2.set_ylabel("Energía (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig_c)
    plt.close(fig_c)
