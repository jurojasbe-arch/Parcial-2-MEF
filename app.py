import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from motor_mef import resolver_mef_presa

st.set_page_config(page_title="MEF Geotecnia Pro", layout="wide")

st.title("🛡️ Modelación por Elementos Finitos (MEF)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    lx = st.slider("Longitud (Lx)", 50.0, 180.0, 135.0)
    prof = st.slider("Prof. Muro (m)", 0.0, 20.0, 10.0)
    pos_muro = st.slider("Posición Muro (m)", 0.0, 15.0, 5.0)
    
    st.subheader("💧 Hidráulica")
    h1, h2 = st.number_input("H Aguas Arriba", value=50.0), st.number_input("H Aguas Abajo", value=5.0)
    k = st.number_input("Permeabilidad (k)", value=1e-5, format="%.1e")
    
    st.subheader("🧪 Suelo")
    gs, e_v = st.number_input("Gs", value=2.65), st.number_input("e", value=0.65)
    
    st.subheader("📐 Malla")
    d_glob, d_crit = st.slider("Tamaño Global", 1.0, 5.0, 2.5), st.slider("Tamaño Crítico", 0.05, 0.5, 0.15)

# Cálculo
with st.spinner("Resolviendo Sistema de Elementos Finitos..."):
    res = resolver_mef_presa(lx, prof, pos_muro, h1, h2, k, d_glob, d_crit, gs, e_v)

# --- DASHBOARD DE RESULTADOS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Caudal (Q)", f"{res['Q']*1000:.3f} L/s/m")
col2.metric("Gradiente Crítico (ic)", f"{res['ic']:.2f}")
col3.metric("i Salida Máx", f"{res['i_exit']:.3f}")
fs_val = res['fs']
col4.metric("FS Sifonamiento", f"{fs_val:.2f}", delta=f"{fs_val-1.5:.2f}", delta_color="normal" if fs_val > 1.5 else "inverse")

tab1, tab2, tab3 = st.tabs(["📊 Análisis 2D (Interactiva)", "🧊 Modelo 3D", "🛡️ Seguridad"])

with tab1:
    st.subheader("Mapa de Calor de Carga Hidráulica (h) y Red de Flujo")
    
    # Crear gráfica 2D con Plotly usando Contour
    # Plotly Contour funciona mejor con mallas no estructuradas si pasamos los puntos
    fig2d = go.Figure()
    
    # Mapa de Calor (h)
    fig2d.add_trace(go.Contour(
        x=res['mesh'].p[0], y=res['mesh'].p[1], z=res['h'],
        colorscale='Turbo', contours_coloring='heatmap',
        line_width=0.5, name="Carga (h)"
    ))

    # Dibujar la estructura (Presa)
    lx, x_i, x_f, x_m, p_m = res['params']
    fig2d.add_shape(type="rect", x0=x_i, y0=30, x1=x_f, y1=35, fillcolor="gray", opacity=0.8)
    if prof > 0:
        fig2d.add_shape(type="line", x0=x_m, y0=30, x1=x_m, y1=30-prof, line=dict(color="black", width=5))

    fig2d.update_layout(xaxis_title="Distancia (m)", yaxis_title="Profundidad (m)", 
                        height=600, template="plotly_white", yaxis=dict(range=[0, 32]))
    st.plotly_chart(fig2d, use_container_width=True)

with tab2:
    st.subheader("Superficie de Presiones")
    fig3d = go.Figure(data=[go.Mesh3d(x=res['mesh'].p[0], y=res['mesh'].p[1], z=res['h'], 
                                     intensity=res['h'], colorscale='Viridis')])
    fig3d.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='h'), height=700, template="plotly_dark")
    st.plotly_chart(fig3d, use_container_width=True)

with tab3:
    st.header("Evaluación de Estabilidad MEF")
    if fs_val > 1.5:
        st.success(f"✅ ESTABLE: El factor de seguridad ({fs_val:.2f}) cumple con la normativa (>1.5).")
    elif fs_val > 1.0:
        st.warning(f"⚠️ CRÍTICO: El factor de seguridad ({fs_val:.2f}) es superior a 1 pero no tiene margen de seguridad suficiente.")
    else:
        st.error(f"🚨 FALLA: Sifonamiento detectado (FS = {fs_val:.2f}). La estructura colapsará.")
