import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
from motor_mef import resolver_mef_presa

st.set_page_config(page_title="MEF Geotecnia Pro", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ Sistema de Elementos Finitos - Análisis de Presas")

with st.sidebar:
    st.header("🎛️ Control de Malla No Estructurada")
    lx = st.slider("Longitud (Lx)", 50.0, 180.0, 135.0)
    prof = st.slider("Muro (m)", 0.0, 20.0, 10.0)
    
    st.subheader("📐 Refinamiento")
    d_glob = st.slider("Tamaño Global", 1.0, 5.0, 2.5)
    d_crit = st.slider("Tamaño en Puntos Críticos", 0.05, 0.5, 0.15)

# Cálculo
with st.spinner("Generando Malla y Resolviendo MEF..."):
    mesh, h_sol = resolver_mef_presa(lx, prof, 5.0, 50, 5, 1e-5, d_glob, d_crit)

# Visualización Sofisticada con Plotly
st.subheader("🔍 Visualización Interactiva del Campo de Presiones")

# Creamos un Contour plot para la malla no estructurada
fig = go.Figure(data=[
    go.Mesh3d(
        x=mesh.p[0], y=mesh.p[1], z=h_sol,
        intensity=h_sol,
        colorscale='Viridis',
        showscale=True,
        flatshading=True
    )
])

# Ajustamos para que parezca un mapa 2D interactivo
fig.update_layout(
    scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Carga (h)',
               camera_eye=dict(x=0, y=0, z=2.5)), # Vista cenital
    height=700, template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

st.info("💡 Tip: Usa el ratón para rotar en 3D o hacer zoom en la tablestaca para ver el refinamiento de la malla.")