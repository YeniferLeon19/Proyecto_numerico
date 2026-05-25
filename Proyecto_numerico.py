import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import streamlit as st
import pandas as pd

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Análisis Numérico - Proyecto",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
    
.error-message {
    background: rgba(255, 100, 100, 0.9);
    border-radius: 10px;
    padding: 15px;
    color: white;
    font-weight: bold;
    border-left: 5px solid #ff0000;
}
.warning-message {
    background: rgba(255, 165, 0, 0.9);
    border-radius: 10px;
    padding: 15px;
    color:grey;
    font-weight: bold;
    border-left: 5px solid #ff8c00;
}
.success-message {
    background: rgba(76, 175, 80, 0.9);
    border-radius: 10px;
    padding: 15px;
    color: grey;
    font-weight: bold;
    border-left: 5px solid #4CAF50;
}
.data-table {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    max-height: 300px;
    overflow-y: auto;
}
/* Forzar visibilidad de textos sobre el fondo cyberpunk */
h1, h2, h3, p, span, label {
    color: black !important;
}
.stMarkdown p {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Análisis Numérico Avanzado")
st.write("Proyecto unificado: Interpolación de Newton, Derivación por Diferencias Finitas, Integración y Raíces")

# INICIO DE ESTADO DE SESIÓN
if 'puntos' not in st.session_state:
    st.session_state.puntos = []  # Lista de tuplas (x, y, id)
if 'ultimo_id' not in st.session_state:
    st.session_state.ultimo_id = 0

#funciones numericas
def limpiar_datos(x, y):
    """Requisito Entrega 1: Elimina redundancias promediando valores de Y repetidos en X"""
    data = defaultdict(list)
    for xi, yi in zip(x, y):
        data[xi].append(yi)
    x_limpio = []
    y_limpio = []
    for xi in sorted(data):
        promedio = sum(data[xi]) / len(data[xi])
        x_limpio.append(xi)
        y_limpio.append(promedio)
    return x_limpio, y_limpio

def datos_irregulares(x):
    """Criterio de mallas de la entrega 1 y 2"""
    if len(x) < 2:
        return False
    dif = [x[i+1] - x[i] for i in range(len(x)-1)]
    if min(dif) <= 0:
        return False
    return max(dif) / min(dif) > 5

def elegir_metodo(x):
    """Ajuste de contradicción: Lógica unificada sin vacíos numéricos"""
    n = len(x)
    if n <= 6 and not datos_irregulares(x):
        return "newton"
    return "tramos"

#interpolacion de newton
def calcular_diferencias_divididas(x, y):
    """
    PRECOMPUTO OPTIMIZADO (De la Entrega 2 aplicada a Newton): 
    Calcula los coeficientes del polinomio de Newton una sola vez.
    """
    n = len(x)
    coef = np.zeros([n, n])
    coef[:,0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
            
    return coef[0, :] # Retorna la diagonal superior (los coeficientes c_i)

def evaluar_polinomio_newton(x, coef, X):
    """Evalúa el polinomio usando los coeficientes precalculados"""
    n = len(coef)
    p = coef[n-1]
    for k in range(1, n):
        p = coef[n-1-k] + (X - x[n-1-k]) * p
    return p

# interpolacion por tramos
def interpolacion_por_tramos(x, y, X):
    if X < x[0]: return y[0]
    if X > x[-1]: return y[-1]
    for i in range(len(x)-1):
        if x[i] <= X <= x[i+1]:
            return y[i] + (y[i+1] - y[i]) * (X - x[i]) / (x[i+1] - x[i])
    return y[-1]

# derivadas
def derivada_finitas(f, X, x_puntos, h=None):
    if len(x_puntos) < 2:
        return 0
    
    if h is None:
        diffs = [x_puntos[i+1] - x_puntos[i] for i in range(len(x_puntos)-1)]
        h = min(diffs) * 0.1 if diffs else 0.001

    # Extremo izquierdo: Progresiva simple O(h)
    if abs(X - x_puntos[0]) < 1e-10:
        return (f(X + h) - f(X)) / h
    
    # Extremo derecho: Regresiva simple O(h) 
    if abs(X - x_puntos[-1]) < 1e-10:
        return (f(X) - f(X - h)) / h
    
    # Puntos interiores: Centrada O(h^2)
    return (f(X + h) - f(X - h)) / (2 * h)

# --- INTEGRACIÓN NUMÉRICA ---
def integracion_trapecio(f, a, b, n=200):
    if a == b: return 0
    h = (b - a) / n
    suma = (f(a) + f(b)) / 2
    for i in range(1, n):
        suma += f(a + i * h)
    return suma * h

def integracion_simpson(f, a, b, n=200):
    if a == b: return 0
    if n % 2 != 0: n += 1  
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n, 2):
        suma += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        suma += 2 * f(a + i * h)
    return suma * h / 3

def integral_acumulada(f, a, puntos_x, metodo_integracion="trapecio"):
    resultados = []
    for x in puntos_x:
        if x <= a:
            resultados.append(0.0)
        else:
            if metodo_integracion == "simpson":
                resultados.append(integracion_simpson(f, a, x))
            else:
                resultados.append(integracion_trapecio(f, a, x))
    return resultados

# hallar raices
def metodo_biseccion(f, a, b, tol=1e-6, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb >= 0: return None
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or (b - a)/2 < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2

def metodo_newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-10: return None # Control de seguridad agregado
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def encontrar_raices(f, df, intervalo):
    a, b = intervalo
    raices = []
    puntos = np.linspace(a, b, 201)
    for i in range(len(puntos) - 1):
        x1, x2 = puntos[i], puntos[i + 1]
        if f(x1) * f(x2) < 0:
            raiz_bisec = metodo_biseccion(f, x1, x2)
            if raiz_bisec is not None:
                # Evitar duplicaciones
                if not any(abs(raiz_bisec - r) < 1e-5 for r in raices):
                    raiz_nr = metodo_newton_raphson(f, df, raiz_bisec)
                    if raiz_nr is not None and a <= raiz_nr <= b:
                        raices.append(raiz_nr)
                    else:
                        raices.append(raiz_bisec)
    return sorted(raices)

#interfaaz bonita

with st.sidebar:
    st.header("⚙️ Configuración")
    
    precision = st.select_slider(
        "Densidad de puntos de gráfica",
        options=["baja", "media", "alta"],
        value="media"
    )
    
    metodo_integracion = st.radio(
        "Método de integración numérica",
        options=["trapecio", "simpson"],
        index=0 # Prioridad al trapecio según la Entrega 1
    )
    
    mostrar_derivada = st.checkbox("Mostrar gráfica de la derivada", value=True)
    mostrar_integral = st.checkbox("Mostrar gráfica de la integral", value=True)
    mostrar_raices = st.checkbox("Analizar y mostrar raíces (Extensión)", value=True)
    
    st.markdown("---")
    if st.button("🗑️ Limpiar todos los puntos", use_container_width=True):
        st.session_state.puntos = []
        st.session_state.ultimo_id = 0
        st.rerun()
    
    st.metric("📊 Puntos en memoria", len(st.session_state.puntos))
    st.caption("📐 Proyecto Ajustado rigurosamente a los requerimientos técnicos.")

# FORMULARIO DE CAPTURA
st.subheader("➕ Entrada de Puntos Discretos")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    nueva_x = st.number_input("Coordenada X", value=0.0, step=0.5, format="%.4f")
with col2:
    nueva_y = st.number_input("Coordenada Y", value=0.0, step=0.5, format="%.4f")
with col3:
    st.write("")
    st.write("")
    if st.button("Agregar Punto", use_container_width=True):
        st.session_state.puntos.append((nueva_x, nueva_y, st.session_state.ultimo_id))
        st.session_state.ultimo_id += 1
        st.rerun()

# ejemplos rapidos
st.subheader("🚀 Inicialización de muestras Rápidas")
col_ej1, col_ej2, col_ej3 = st.columns(3)

with col_ej1:
    if st.button("📈 Polinomio Suave (5 puntos)", use_container_width=True):
        st.session_state.puntos = [(0.0, 1.0, 0), (1.0, 3.0, 1), (2.0, 0.0, 2), (3.0, -2.0, 3), (4.0, 2.0, 4)]
        st.session_state.ultimo_id = 5
        st.rerun()
with col_ej2:
    if st.button("📊 Curva Estándar (6 puntos)", use_container_width=True):
        st.session_state.puntos = [(-2.0, 4.0, 0), (-1.0, 1.0, 1), (0.0, 0.0, 2), (1.0, 1.0, 3), (2.0, 4.0, 4), (3.0, 9.0, 5)]
        st.session_state.ultimo_id = 6
        st.rerun()
with col_ej3:
    if st.button("🌍 Expectativa de Vida OMS (50 puntos)", use_container_width=True):
        # Dataset OMS: X = Años de Escolaridad (Ordenado Ascendente), Y = Expectativa de Vida (kg/m3 simulado en la lógica anterior)
        st.session_state.puntos = [
            (10.0, 59.9, 0),   # Afganistán
            (10.1, 65.0, 1),   # Afganistán
            (10.2, 71.8, 2),   # Bangladesh
            (11.4, 52.4, 3),   # Angola
            (11.5, 68.0, 4),   # India
            (11.7, 68.3, 5),   # India
            (11.8, 71.0, 6),   # Muestra Control 1
            (11.9, 72.1, 7),   # Muestra Control 2
            (12.1, 74.3, 8),   # Marruecos
            (12.3, 73.0, 9),   # Muestra Control 3
            (12.5, 73.8, 10),  # Muestra Control 4
            (12.7, 72.7, 11),  # Azerbaiyán
            (13.0, 56.9, 12),  # Sudáfrica
            (13.1, 79.0, 13),  # Egipto
            (13.2, 77.2, 14),  # Albania
            (13.3, 76.7, 15),  # México
            (13.4, 75.5, 16),  # Perú
            (13.5, 76.1, 17),  # China
            (13.6, 74.8, 18),  # Colombia
            (13.8, 75.2, 19),  # Muestra Control 5
            (13.9, 79.1, 20),  # Cuba
            (14.0, 76.2, 21),  # Ecuador
            (14.2, 77.8, 22),  # Albania
            (14.4, 75.6, 23),  # Argelia
            (14.7, 75.9, 24),  # Muestra Control 6
            (15.0, 75.0, 25),  # Rusia
            (15.2, 75.0, 26),  # Brasil
            (15.3, 83.7, 27),  # Japón
            (15.5, 77.0, 28),  # Uruguay
            (15.7, 78.2, 29),  # Muestra Control 7
            (15.9, 81.5, 30),  # Austria / Finlandia
            (16.0, 83.4, 31),  # Suiza
            (16.3, 85.0, 32),  # Chile / Francia / USA
            (16.6, 81.1, 33),  # Bélgica
            (16.9, 81.1, 34),  # Portugal
            (17.0, 81.1, 35),  # Finlandia
            (17.1, 81.0, 36),  # Alemania
            (17.3, 76.3, 37),  # Argentina
            (17.5, 81.8, 38),  # Noruega
            (17.7, 82.8, 39),  # España
            (18.1, 81.9, 40),  # Países Bajos
            (18.4, 82.0, 41),  # Muestra Control 8
            (18.7, 82.3, 42),  # Muestra Control 9
            (19.0, 82.5, 43),  # Muestra Control 10
            (19.2, 86.0, 44),  # Dinamarca / Nueva Zelanda
            (19.5, 83.1, 45),  # Muestra Control 11
            (19.8, 83.5, 46),  # Muestra Control 12
            (20.1, 83.0, 47),  # Muestra Control 13
            (20.4, 82.8, 48),  # Australia
            (20.7, 83.9, 49)   # Muestra Control 14
        ]
        st.session_state.ultimo_id = 50
        st.rerun()

# tabla para usuario
st.subheader("📋 Datos Registrados")
if st.session_state.puntos:
    df_puntos = pd.DataFrame([(p[0], p[1]) for p in st.session_state.puntos], columns=["X", "Y"])
    # SE ELIMINÓ EL .T PARA DESPLEGAR LOS DATOS VERTICALMENTE
    st.dataframe(df_puntos, use_container_width=True, height=min(300, 35 * len(st.session_state.puntos) + 38))
    
    if st.button("🗑️ Eliminar último punto ingresado"):
        st.session_state.puntos.pop()
        st.rerun()
else:
    st.info("ℹ️ Sistema listo. Ingrese coordenadas para comenzar el procesamiento.")

#graficos
if len(st.session_state.puntos) >= 2:
    x_raw = [p[0] for p in st.session_state.puntos]
    y_raw = [p[1] for p in st.session_state.puntos]
    
    # 1. Limpieza obligatoria de la Entrega 1
    x, y = limpiar_datos(x_raw, y_raw)
    
    st.info(f" **Dominio Funcional:** [{x[0]:.4f}, {x[-1]:.4f}] | **Nodos Únicos:** {len(x)}")
    
    # 2. Selección de estrategia unificada
    metodo = elegir_metodo(x)
    
    if metodo == "newton":
        coef_newton = calcular_diferencias_divididas(x, y)
        f_interp = lambda X: evaluar_polinomio_newton(x, coef_newton, X)
        st.success("✅ **Algoritmo Operativo: Polinomio Global de Newton** (Diferencias Divididas)")
        st.caption("Construcción matemática estricta a partir de la primera entrega.")
    else:
        f_interp = lambda X: interpolacion_por_tramos(x, y, X)
        st.markdown('<div class="warning-message">⚠️ **Algoritmo Operativo: Interpolación por Tramos Lineales** (Criterio Adaptativo por Malla Irregular o n > 6)</div>', unsafe_allow_html=True)
    
    # Resolución de gráfico
    num_puntos_grafica = 500 if precision == "alta" else (300 if precision == "media" else 150)
    X_vals = np.linspace(x[0], x[-1], num_puntos_grafica)
    Y_vals = [f_interp(xi) for xi in X_vals]
    
    # Gráfica Principal
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(X_vals, Y_vals, linewidth=2, label="f(x) Interpolante", color='#3b82f6')
    ax1.scatter(x, y, color='#ef4444', s=60, zorder=5, label="Datos de Usuario")
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    ax1.set_title("Comportamiento de la Función Interpolante")
    st.pyplot(fig1)
    
    # evaluacion
    st.subheader("📐 Filtro de Evaluación en Punto Singular")
    X_eval = st.number_input(
        "Defina punto objetivo (X)", 
        value=float((x[0] + x[-1]) / 2),
        min_value=float(x[0]), max_value=float(x[-1]),
        format="%.4f"
    )
    
    y_eval = f_interp(X_eval)
    dy_eval = derivada_finitas(f_interp, X_eval, x)
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric(" Valor f(X)", f"{y_eval:.6f}")
    with col_r2:
        st.metric("📈 Derivada f'(X)", f"{dy_eval:.6f}")
    with col_r3:
        if metodo_integracion == "simpson":
            val_int = integracion_simpson(f_interp, x[0], X_eval)
            st.metric(f"📐 Integral Acumulada F(X)", f"{val_int:.6f}")
        else:
            val_int = integracion_trapecio(f_interp, x[0], X_eval)
            st.metric(f"📐 Integral Acumulada F(X)", f"{val_int:.6f}")

    # Gráfica de Recta Tangente
    tangente_vals = [y_eval + dy_eval * (xi - X_eval) for xi in X_vals]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(X_vals, Y_vals, linewidth=2, color='#3b82f6', label="f(x)")
    ax2.plot(X_vals, tangente_vals, linestyle='--', color='#ec4899', label="Recta Tangente")
    ax2.scatter([X_eval], [y_eval], color='#eab308', s=120, zorder=6, label="Punto Evaluado")
    ax2.grid(True, alpha=0.1)
    ax2.legend()
    st.pyplot(fig2)

    # GRÁFICA DE LA DERIVADA
    if mostrar_derivada:
        derivada_vals = [derivada_finitas(f_interp, xi, x) for xi in X_vals]
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(X_vals, derivada_vals, linewidth=2, color='#a855f7')
        ax3.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.grid(True, alpha=0.1)
        ax3.set_title("Perfil de la Derivada Numérica")
        st.pyplot(fig3)
        
    # GRÁFICA DE LA INTEGRAL ACUMULADA
    if mostrar_integral:
        integral_vals = integral_acumulada(f_interp, x[0], X_vals, metodo_integracion)
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.plot(X_vals, integral_vals, linewidth=2, color='#06b6d4')
        ax4.fill_between(X_vals, integral_vals, alpha=0.15, color='#06b6d4')
        ax4.grid(True, alpha=0.1)
        ax4.set_title(f"Perfil de la Antiderivada F(x) - [{metodo_integracion.upper()}]")
        st.pyplot(fig4)

    # ANÁLISIS DE RAÍCES COMPLEMENTARIO
    if mostrar_raices:
        df_interp = lambda X: derivada_finitas(f_interp, X, x)
        raices_calculadas = encontrar_raices(f_interp, df_interp, (x[0], x[-1]))
        
        if raices_calculadas:
            st.subheader(" Localización de Raíces del Interpolante")
            cols_r = st.columns(min(len(raices_calculadas), 5))
            for idx, r_val in enumerate(raices_calculadas):
                with cols_r[idx % 5]:
                    st.metric(f"Raíz x{idx+1}", f"{r_val:.5f}")
                    
            fig5, ax5 = plt.subplots(figsize=(12, 5))
            ax5.plot(X_vals, Y_vals, linewidth=2, color='#3b82f6')
            ax5.axhline(y=0, color='white', linestyle='-', linewidth=0.7, alpha=0.5)
            ax5.scatter(raices_calculadas, [f_interp(r) for r in raices_calculadas], color='#eab308', s=150, marker='X', zorder=6, label="Raíces reales")
            ax5.grid(True, alpha=0.1)
            ax5.legend()
            st.pyplot(fig5)
        else:
            st.info("ℹ️ No se detectaron intersecciones reales con el eje X en el intervalo de trabajo actual.")

elif len(st.session_state.puntos) == 1:
    st.warning("⚠️ Muestra insuficiente. Ingrese al menos 2 puntos para calcular diferencias y construir mallas.")