import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import numpy as np
from motor_mef import resolver_mef_presa

st.set_page_config(page_title="MEF Geotecnia Pro", layout="wide")
st.title("🛡️ Modelación por Elementos Finitos (Grilla Estructurada)")

with st.sidebar:
    st.header("⚙️ Configuración Geométrica")
    lx = st.slider("Longitud X (m)", 50.0, 180.0, 135.0, 5.0)
    prof = st.slider("Prof. Muro (m)", 0.0, 20.0, 16.0, 1.0)
    pos_muro = st.slider("Posición Muro (m)", 0.0, 15.0, 13.0, 0.5)
    ancho_3d = st.slider("Ancho Z (Modelo 3D)", 10.0, 100.0, 30.0, 5.0)
    
    st.subheader("💧 Hidráulica")
    h1 = st.number_input("H Aguas Arriba", value=40.0)
    h2 = st.number_input("H Aguas Abajo", value=5.0)
    k = st.number_input("Permeabilidad (k)", value=1e-5, format="%.1e")
    
    st.subheader("🧪 Suelo")
    gs, e_v = st.number_input("Gs", value=2.65), st.number_input("e", value=0.65)

with st.spinner("Fabricando Malla Tensorial y Resolviendo Sistema..."):
    res = resolver_mef_presa(lx, prof, pos_muro, h1, h2, k, gs, e_v)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Caudal (Q)", f"{res['Q']*1000:.3f} L/s/m")
col2.metric("Gradiente Crítico (ic)", f"{res['ic']:.2f}")
col3.metric("i Salida Máx", f"{res['i_exit']:.3f}")
fs_val = res['fs']
col4.metric("FS Sifonamiento", f"{fs_val:.2f}", delta=f"{fs_val-1.5:.2f}", delta_color="normal" if fs_val > 1.5 else "inverse")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Red de Flujo y Malla", "🧊 Modelo 3D Físico", "🛡️ Seguridad", "📄 Memoria de Cálculo"])
Lx, x_i, x_f, x_m, prof_muro, grosor = res['params']

with tab1:
    triang = mtri.Triangulation(res['mesh'].p[0], res['mesh'].p[1])
    x_tri = res['mesh'].p[0, triang.triangles].mean(axis=1)
    y_tri = res['mesh'].p[1, triang.triangles].mean(axis=1)
    
    # Máscara estricta booleana para los gráficos
    mask_presa = (x_tri >= x_i) & (x_tri <= x_f) & (y_tri >= 25.0)
    mask_muro = (x_tri >= x_m - grosor/2) & (x_tri <= x_m + grosor/2) & (y_tri >= 25.0 - prof_muro)
    triang.set_mask(mask_presa | mask_muro)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico A: Red de Flujo
    cf1 = ax1.tricontourf(triang, res['h'], levels=40, cmap='Blues', alpha=0.9)
    ax1.tricontour(triang, res['h'], levels=15, colors='black', linewidths=0.6)
    
    xi, yi = np.meshgrid(np.linspace(0, Lx, 250), np.linspace(0, 30, 120))
    interp_ix = mtri.LinearTriInterpolator(triang, res['ix'])
    interp_iy = mtri.LinearTriInterpolator(triang, res['iy'])
    
    ix_grid = interp_ix(xi, yi)
    iy_grid = interp_iy(xi, yi)
    
    # Prohibimos a las flechas dibujarse sobre el concreto/acero
    mask_grid_obstaculo = ((xi >= x_i) & (xi <= x_f) & (yi >= 25.0)) | ((xi >= x_m - grosor/2) & (xi <= x_m + grosor/2) & (yi >= 25.0 - prof_muro))
    ix_grid = np.ma.masked_where(mask_grid_obstaculo, ix_grid)
    iy_grid = np.ma.masked_where(mask_grid_obstaculo, iy_grid)
    
    ax1.streamplot(xi, yi, ix_grid, iy_grid, color='darkblue', linewidth=1.0, density=1.8)
    
    verts = res['mesh'].p[:, res['mesh'].t].T
    ax1.add_collection(PolyCollection(verts, edgecolors='black', facecolors='none', linewidths=0.2, alpha=0.4))
    
    ax1.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0: ax1.fill([x_m-grosor/2, x_m+grosor/2, x_m+grosor/2, x_m-grosor/2], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
    ax1.set_title('Malla Estructurada Cuadriculada con Equipotenciales y Líneas de Flujo')
    
    # Gráfico B: Mapa de Gradientes Exacto usando Triangulación
    # Esto garantiza que el calor nazca EXACTAMENTE en la punta del muro
    cf2 = ax2.tricontourf(triang, res['imag'], levels=np.linspace(0, res['ic']*1.5, 50), cmap='turbo', extend='max')
    plt.colorbar(cf2, ax=ax2, label='Gradiente Hidráulico (i)')
    ax2.tricontour(triang, res['imag'], levels=[res['ic']], colors='red', linewidths=2.5, linestyles='dashed')
    
    ax2.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0: ax2.fill([x_m-grosor/2, x_m+grosor/2, x_m+grosor/2, x_m-grosor/2], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
    ax2.set_title('Concentración Vectorial de Gradientes (Foco Singular Físico en la Punta)')

    for ax in [ax1, ax2]: ax.set_aspect('equal'); ax.set_xlim(0, Lx); ax.set_ylim(0, 30)
    st.pyplot(fig)

with tab2:
    st.subheader("Modelación CAD Física en 3D con Flujo Mapeado")
    fig3d = go.Figure()

    # Mapeo del calor (presiones) directamente sobre las caras usando la triangulación física
    x_nodos = res['mesh'].p[0]
    y_nodos = res['mesh'].p[1]
    h_nodos = res['h']
    tris = triang.triangles

    # Cara Frontal (Z = 0) dibujando la presión
    fig3d.add_trace(go.Mesh3d(
        x=x_nodos, y=np.zeros_like(x_nodos), z=y_nodos,
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        intensity=h_nodos, colorscale='Blues', showscale=True, name='Frente (Flujo)'
    ))
    
    # Cara Trasera (Z = Ancho)
    fig3d.add_trace(go.Mesh3d(
        x=x_nodos, y=np.full_like(x_nodos, ancho_3d), z=y_nodos,
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        intensity=h_nodos, colorscale='Blues', showscale=False, name='Fondo (Flujo)'
    ))

    # Creador de bloques de concreto/acero 3D
    def crear_bloque(x0, x1, y0, y1, z0, z1, color, nombre):
        return go.Mesh3d(
            x=[x0, x0, x1, x1, x0, x0, x1, x1], y=[y0, y1, y1, y0, y0, y1, y1, y0], z=[z0, z0, z0, z0, z1, z1, z1, z1],
            alphahull=0, color=color, flatshading=True, name=nombre
        )

    # Estructuras
    fig3d.add_trace(crear_bloque(x_i, x_f, 0, ancho_3d, 25, 30, '#888888', 'Presa de Concreto'))
    if prof_muro > 0: fig3d.add_trace(crear_bloque(x_m-grosor/2, x_m+grosor/2, 0, ancho_3d, 25-prof_muro, 25, '#222222', 'Tablestaca de Acero'))

    fig3d.update_layout(
        scene=dict(
            xaxis_title='Longitud X (m)', 
            yaxis_title='Ancho Z (m)', 
            zaxis_title='Elevación Y (m)', 
            aspectmode='manual', aspectratio=dict(x=3, y=1.5, z=1)
        ), 
        height=750, template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig3d, use_container_width=True)

with tab3:
    st.header("Evaluación Estructural de Sifonamiento")
    if fs_val > 1.5: st.success(f"✅ ESTABLE (FS = {fs_val:.2f})")
    elif fs_val > 1.0: st.warning(f"⚠️ CRÍTICO (FS = {fs_val:.2f})")
    else: st.error(f"🚨 FALLA INMINENTE POR TUBIFICACIÓN (FS = {fs_val:.2f})")

with tab4:
    st.header("📄 Memoria de Cálculo y Marco Teórico MEF")
    st.markdown(r"El método desarrollado resuelve el campo bidimensional de presiones intersticiales empleando el **Método de los Elementos Finitos (MEF)**...")
    # ... (El texto teórico se mantiene idéntico al anterior)
