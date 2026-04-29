import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import numpy as np
from motor_mef import resolver_mef_presa

st.set_page_config(page_title="MEF Grilla Pro", layout="wide")
st.title("🛡️ Modelación MEF Estructurada (Grilla Cuadriculada)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración Geométrica")
    lx = st.slider("Longitud (Lx)", 50.0, 180.0, 135.0, 5.0)
    prof = st.slider("Prof. Muro (m)", 0.0, 20.0, 10.0, 1.0)
    pos_muro = st.slider("Posición Muro (m)", 0.0, 15.0, 5.0, 0.5)
    
    st.subheader("💧 Hidráulica")
    h1 = st.number_input("H Aguas Arriba", value=50.0)
    h2 = st.number_input("H Aguas Abajo", value=5.0)
    k = st.number_input("Permeabilidad (k)", value=1e-5, format="%.1e")
    
    st.subheader("🧪 Suelo y Estabilidad")
    gs, e_v = st.number_input("Gs", value=2.65), st.number_input("e", value=0.65)

# Cálculo
with st.spinner("Ensamblando Grilla Estructurada y Resolviendo MEF..."):
    res = resolver_mef_presa(lx, prof, pos_muro, h1, h2, k, gs, e_v)

# Métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Caudal (Q)", f"{res['Q']*1000:.3f} L/s/m")
col2.metric("FS Sifonamiento", f"{res['fs']:.2f}")

# --- PESTAÑAS ---
tab1, tab2 = st.tabs(["📊 Red de Flujo (Grilla visible)", "🛡️ Seguridad"])
Lx, x_i, x_f, x_m, prof_muro = res['params']

with tab1:
    st.subheader("Visualización de la Malla Cuadriculada y Carga Hidráulica")
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 1. Mapa de calor de Carga Hidráulica (h)
    # Usamos mtri solo para el contourf (relleno de color)
    triang = mtri.Triangulation(res['mesh'].p[0], res['mesh'].p[1])
    cf = ax1.tricontourf(triang, res['h'], levels=40, cmap='Blues', alpha=0.8)
    plt.colorbar(cf, ax=ax1, label='Carga Hidráulica (h)')
    
    # 2. DIBUJO DE LA GRILLA MEF (CUADRICULADA) Impecable
    # Obtenemos las coordenadas de los vértices de cada cuadrilátero
    verts = res['mesh'].p[:, res['mesh'].t].T
    # Creamos una colección de polígonos para dibujarlos eficientemente
    grid_coll = PolyCollection(verts, edgecolors='black', facecolors='none', linewidths=0.2, alpha=0.3)
    ax1.add_collection(grid_coll)
    
    # 3. Geometría de la Estructura (Sobre la grilla)
    ax1.fill([x_i, x_f, x_f, x_i], [30, 30, 32, 32], color='#444444', zorder=10) # Presa
    if prof_muro > 0:
        ax1.plot([x_m, x_m], [30, 30-prof_muro], color='black', linewidth=4, zorder=10) # Muro

    ax1.set_aspect('equal')
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, 32)
    ax1.grid(False) # Apagamos la grilla de fondo de matplotlib, ya tenemos la nuestra
    ax1.set_title('Malla MEF de Cuadriláteros Estructurada y Equipotenciales')
    
    st.pyplot(fig)

with tab2:
    st.header("Análisis de Estabilidad")
    st.markdown(f"FS = {res['fs']:.2f} (Gradiente Crítico = {res['ic']:.2f}, Máx Salida = {res['i_exit']:.3f})")
