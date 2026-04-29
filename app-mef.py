import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from motor_mef import resolver_mef_presa

st.set_page_config(page_title="MEF Geotecnia Pro", layout="wide")

st.title("🛡️ Modelación por Elementos Finitos (MEF)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    lx = st.slider("Longitud (Lx)", 50.0, 180.0, 135.0, 5.0)
    prof = st.slider("Prof. Muro (m)", 0.0, 20.0, 10.0, 1.0)
    pos_muro = st.slider("Posición Muro (m)", 0.0, 15.0, 5.0, 0.5)
    
    st.subheader("💧 Hidráulica")
    h1 = st.number_input("H Aguas Arriba", value=50.0)
    h2 = st.number_input("H Aguas Abajo", value=5.0)
    k = st.number_input("Permeabilidad (k)", value=1e-5, format="%.1e")
    
    st.subheader("🧪 Suelo")
    gs = st.number_input("Gs", value=2.65)
    e_v = st.number_input("e", value=0.65)
    
    st.subheader("📐 Malla MEF")
    d_glob = st.slider("Tamaño Global", 1.0, 5.0, 2.5)
    d_crit = st.slider("Tamaño Crítico", 0.05, 0.5, 0.15)

# --- EJECUCIÓN DEL MOTOR ---
with st.spinner("Ensamblando y Resolviendo Sistema de Elementos Finitos..."):
    # ¡Aquí está la corrección! Se envían todos los parámetros, incluyendo gs y e_v
    res = resolver_mef_presa(lx, prof, pos_muro, h1, h2, k, d_glob, d_crit, gs, e_v)

# --- MÉTRICAS PRINCIPALES ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Caudal (Q)", f"{res['Q']*1000:.3f} L/s/m")
col2.metric("Gradiente Crítico (ic)", f"{res['ic']:.2f}")
col3.metric("i Salida Máx", f"{res['i_exit']:.3f}")
fs_val = res['fs']
col4.metric("FS Sifonamiento", f"{fs_val:.2f}", delta=f"{fs_val-1.5:.2f}", delta_color="normal" if fs_val > 1.5 else "inverse")

# --- PESTAÑAS DEL DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📊 Análisis 2D (Estándar de Ingeniería)", "🧊 Modelo 3D Interactivo", "🛡️ Seguridad"])

# Variables geométricas extraídas del motor
Lx, x_i, x_f, x_m, prof_muro = res['params']

with tab1:
    st.markdown("### Resultados Bidimensionales (Interpolación de Malla No Estructurada)")
    
    # Creamos la triangulación base para Matplotlib a partir de la malla FEM
    triang = mtri.Triangulation(res['mesh'].p[0], res['mesh'].p[1])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- Gráfico 1: Red de Flujo (Carga Hidráulica) ---
    cf1 = ax1.tricontourf(triang, res['h'], levels=50, cmap='Blues', alpha=0.9)
    ax1.tricontour(triang, res['h'], levels=15, colors='black', linewidths=0.5)
    
    # Dibujo de Estructura
    ax1.fill([x_i, x_f, x_f, x_i], [30, 30, 32, 32], color='#333333', zorder=10) # Presa
    if prof_muro > 0:
        ax1.plot([x_m, x_m], [30, 30-prof_muro], color='#333333', linewidth=4, zorder=10) # Tablestaca
        
    ax1.set_title(f'Distribución de Carga Hidráulica (h) - Malla MEF')
    ax1.set_ylabel('Elevación (m)')
    
    # --- Gráfico 2: Mapa de Calor de Gradientes ---
    cf2 = ax2.tricontourf(triang, res['imag'], levels=np.linspace(0, 1.2, 50), cmap='turbo', extend='max')
    plt.colorbar(cf2, ax=ax2, label='Gradiente Hidráulico (i)')
    
    # Dibujo de Estructura
    ax2.fill([x_i, x_f, x_f, x_i], [30, 30, 32, 32], color='#333333', zorder=10)
    if prof_muro > 0:
        ax2.plot([x_m, x_m], [30, 30-prof_muro], color='#333333', linewidth=4, zorder=10)
        
    ax2.tricontour(triang, res['imag'], levels=[res['ic']], colors='red', linewidths=2.5, linestyles='dashed')
    ax2.set_title('Mapa de Calor del Gradiente (i)')
    ax2.set_xlabel('Posición X (m)')
    ax2.set_ylabel('Elevación (m)')

    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, 32)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.markdown("### Superficie de Presiones Tridimensional")
    
    # 3D Mejorado y Proporcionado
    fig3d = go.Figure(data=[go.Mesh3d(
        x=res['mesh'].p[0], 
        y=res['mesh'].p[1], 
        z=res['h'], 
        intensity=res['h'], 
        colorscale='Turbo',
        opacity=0.95
    )])
    
    fig3d.update_layout(
        scene=dict(
            xaxis_title='Longitud (m)', 
            yaxis_title='Profundidad (m)', 
            zaxis_title='Carga (h)',
            # Fix intuitivo: Obliga a que el modelo respete una proporción física 3:1
            aspectratio=dict(x=3.0, y=1.0, z=1.2) 
        ),
        height=700, 
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption("🔍 Gira el modelo con el ratón. Nota cómo la 'caída' de presión es más abrupta justo en la barrera de la tablestaca.")

with tab3:
    st.header("Evaluación de Estabilidad MEF")
    if fs_val > 1.5:
        st.success(f"✅ ESTABLE: El factor de seguridad ({fs_val:.2f}) cumple con la normativa (>1.5).")
    elif fs_val > 1.0:
        st.warning(f"⚠️ CRÍTICO: El factor de seguridad ({fs_val:.2f}) es superior a 1 pero no tiene margen normativo suficiente.")
    else:
        st.error(f"🚨 FALLA: Sifonamiento detectado (FS = {fs_val:.2f}). La estructura es inestable.")
