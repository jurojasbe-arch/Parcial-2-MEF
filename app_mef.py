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
    st.header("⚙️ Configuración")
    lx = st.slider("Longitud (Lx)", 50.0, 180.0, 135.0, 5.0)
    prof = st.slider("Prof. Muro (m)", 0.0, 20.0, 10.0, 1.0)
    pos_muro = st.slider("Posición Muro (m)", 0.0, 15.0, 5.0, 0.5)
    
    st.subheader("💧 Hidráulica")
    h1 = st.number_input("H Aguas Arriba", value=50.0)
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

# ¡Tus 4 pestañas de vuelta!
tab1, tab2, tab3, tab4 = st.tabs(["📊 Red de Flujo y Malla", "🧊 Modelo 3D", "🛡️ Seguridad", "📄 Memoria de Cálculo"])
Lx, x_i, x_f, x_m, prof_muro = res['params']

with tab1:
    triang = mtri.Triangulation(res['mesh'].p[0], res['mesh'].p[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Mapa de Calor y Líneas
    cf1 = ax1.tricontourf(triang, res['h'], levels=40, cmap='Blues', alpha=0.9)
    ax1.tricontour(triang, res['h'], levels=15, colors='black', linewidths=0.6)
    
    # Líneas de Flujo Vectoriales
    xi, yi = np.meshgrid(np.linspace(0, Lx, 180), np.linspace(0, 30, 90))
    interp_ix = mtri.LinearTriInterpolator(triang, res['ix'])
    interp_iy = mtri.LinearTriInterpolator(triang, res['iy'])
    
    # Enmascarar presa para que el agua la rodee
    ix_grid, iy_grid = interp_ix(xi, yi), interp_iy(xi, yi)
    mask = (xi >= x_i) & (xi <= x_f) & (yi >= 25) | ((xi >= x_m - 0.2) & (xi <= x_m + 0.2) & (yi >= 25 - prof_muro))
    ix_grid, iy_grid = np.ma.masked_where(mask, ix_grid), np.ma.masked_where(mask, iy_grid)
    ax1.streamplot(xi, yi, ix_grid, iy_grid, color='darkblue', linewidth=1.0, density=1.5)
    
    # DIBUJO DE LA MALLA CUADRICULADA PURA
    verts = res['mesh'].p[:, res['mesh'].t].T
    grid_coll = PolyCollection(verts, edgecolors='black', facecolors='none', linewidths=0.2, alpha=0.4)
    ax1.add_collection(grid_coll)
    
    # Geometría Física
    ax1.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0: ax1.fill([x_m-0.2, x_m+0.2, x_m+0.2, x_m-0.2], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
    ax1.set_title('Malla Estructurada Cuadriculada con Equipotenciales y Líneas de Flujo')
    
    # 2. Mapa de Calor de Gradientes
    interp_imag = mtri.LinearTriInterpolator(triang, res['imag'])
    imag_grid = np.ma.masked_where(mask, interp_imag(xi, yi))
    cf2 = ax2.contourf(xi, yi, imag_grid, levels=np.linspace(0, res['ic']*1.2, 50), cmap='turbo', extend='max')
    plt.colorbar(cf2, ax=ax2, label='Gradiente Hidráulico (i)')
    ax2.contour(xi, yi, imag_grid, levels=[res['ic']], colors='red', linewidths=2.5, linestyles='dashed')
    ax2.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0: ax2.fill([x_m-0.2, x_m+0.2, x_m+0.2, x_m-0.2], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
    ax2.set_title('Concentración Vectorial de Gradientes (Suavizado Tensorial)')

    for ax in [ax1, ax2]: ax.set_aspect('equal'); ax.set_xlim(0, Lx); ax.set_ylim(0, 30)
    st.pyplot(fig)

with tab2:
    fig3d = go.Figure(data=[go.Mesh3d(x=res['mesh'].p[0], y=res['mesh'].p[1], z=res['h'], intensity=res['h'], colorscale='Turbo')])
    fig3d.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Carga', aspectratio=dict(x=3.0, y=1.0, z=1.2)), height=700, template="plotly_dark")
    st.plotly_chart(fig3d, use_container_width=True)

with tab3:
    if fs_val > 1.5: st.success(f"✅ ESTABLE (FS = {fs_val:.2f})")
    elif fs_val > 1.0: st.warning(f"⚠️ CRÍTICO (FS = {fs_val:.2f})")
    else: st.error(f"🚨 FALLA POR SIFONAMIENTO (FS = {fs_val:.2f})")

with tab4:
    st.header("📄 Memoria de Cálculo MEF Estructurado")
    st.markdown("""
    Este modelo utiliza una discretización espacial basada en una **Grilla Tensorial Estructurada** de elementos cuadriláteros bilineales (Quad-4). La topología geométrica de la presa y la tablestaca se incorpora mediante operaciones de excavación booleana sobre la malla generada, garantizando un cálculo riguroso del flujo alrededor de la estructura.
    
    ### Formulación Variacional de Galerkin
    El problema de flujo en medios porosos se describe mediante la ecuación de Laplace. Su forma débil se expresa como:
    """)
    st.latex(r"\int_{\Omega} \nabla v \cdot (k \nabla h) \, d\Omega = 0")
    st.markdown("""
    Donde $h$ es la carga hidráulica y $v$ es la función de prueba.
    
    ### Ensamblaje y Caudal Exacto
    La matriz de permeabilidad global $[K]$ se ensambla proyectando las matrices jacobianas locales. El caudal ($Q$) se recupera extrayendo las reacciones nodales en las fronteras de Dirichlet, garantizando la conservación de masa:
    """)
    st.latex(r"\{R\} = [K]\{H\} \quad \rightarrow \quad Q = \sum |R_{\text{entrada}}|")
