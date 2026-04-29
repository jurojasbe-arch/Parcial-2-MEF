import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from motor_mef import resolver_mef_presa

st.set_page_config(page_title="MEF Geotecnia Pro", layout="wide")
st.title("🛡️ Modelación por Elementos Finitos (MEF)")

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

with st.spinner("Ensamblando Geometría Sólida y Resolviendo MEF..."):
    res = resolver_mef_presa(lx, prof, pos_muro, h1, h2, k, d_glob, d_crit, gs, e_v)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Caudal (Q)", f"{res['Q']*1000:.3f} L/s/m")
col2.metric("Gradiente Crítico (ic)", f"{res['ic']:.2f}")
col3.metric("i Salida Máx", f"{res['i_exit']:.3f}")
fs_val = res['fs']
col4.metric("FS Sifonamiento", f"{fs_val:.2f}", delta=f"{fs_val-1.5:.2f}", delta_color="normal" if fs_val > 1.5 else "inverse")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis 2D", "🧊 Modelo 3D", "🛡️ Seguridad", "📄 Memoria de Cálculo"])
Lx, x_i, x_f, x_m, prof_muro = res['params']

with tab1:
    triang = mtri.Triangulation(res['mesh'].p[0], res['mesh'].p[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- GRÁFICO 1: Red de Flujo y Malla ---
    cf1 = ax1.tricontourf(triang, res['h'], levels=40, cmap='Blues', alpha=0.9)
    ax1.tricontour(triang, res['h'], levels=15, colors='black', linewidths=0.6) # Equipotenciales
    
    # Dibujo de la Malla (Líneas negras semi-transparentes para que resalte)
    ax1.triplot(triang, color='black', linewidth=0.3, alpha=0.4) 
    
    # Líneas de Flujo Vectoriales
    xi, yi = np.meshgrid(np.linspace(0, Lx, 180), np.linspace(0, 30, 90))
    interp_ix = mtri.LinearTriInterpolator(triang, res['ix'])
    interp_iy = mtri.LinearTriInterpolator(triang, res['iy'])
    
    # Enmascarar las líneas de flujo para que NO atraviesen el muro
    ix_grid, iy_grid = interp_ix(xi, yi), interp_iy(xi, yi)
    mask_presa = (xi >= x_i) & (xi <= x_f) & (yi >= 25)
    mask_muro = (xi >= x_m - 0.2) & (xi <= x_m + 0.2) & (yi >= 25 - prof_muro)
    ix_grid = np.ma.masked_where(mask_presa | mask_muro, ix_grid)
    iy_grid = np.ma.masked_where(mask_presa | mask_muro, iy_grid)
    
    ax1.streamplot(xi, yi, ix_grid, iy_grid, color='darkblue', linewidth=1.0, density=1.5, arrowsize=1.2)
    
    # Geometría Física (Tapando los huecos vacíos de la malla)
    ax1.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0:
        ax1.fill([x_m-0.1, x_m+0.1, x_m+0.1, x_m-0.1], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
        
    ax1.set_title('Red de Flujo: Malla MEF, Equipotenciales y Líneas de Corriente')
    ax1.set_ylabel('Elevación (m)')
    
    # --- GRÁFICO 2: Mapa de Calor Vectorial ---
    cf2 = ax2.tricontourf(triang, res['imag'], levels=np.linspace(0, res['ic']*1.2, 50), cmap='turbo', extend='max')
    plt.colorbar(cf2, ax=ax2, label='Gradiente Hidráulico (i)')
    
    ax2.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0:
        ax2.fill([x_m-0.1, x_m+0.1, x_m+0.1, x_m-0.1], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
        
    ax2.tricontour(triang, res['imag'], levels=[res['ic']], colors='red', linewidths=2.5, linestyles='dashed')
    ax2.set_title('Concentración de Gradientes: El Rojo Indica Peligro de Tubificación')
    ax2.set_xlabel('Posición X (m)')
    ax2.set_ylabel('Elevación (m)')

    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, 30)

    st.pyplot(fig)

with tab2:
    fig3d = go.Figure(data=[go.Mesh3d(x=res['mesh'].p[0], y=res['mesh'].p[1], z=res['h'], intensity=res['h'], colorscale='Turbo', opacity=0.95)])
    fig3d.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Carga (h)', aspectratio=dict(x=3.0, y=1.0, z=1.2)), height=700, template="plotly_dark")
    st.plotly_chart(fig3d, use_container_width=True)

with tab3:
    if fs_val > 1.5: st.success(f"✅ ESTABLE (FS = {fs_val:.2f})")
    elif fs_val > 1.0: st.warning(f"⚠️ CRÍTICO (FS = {fs_val:.2f})")
    else: st.error(f"🚨 FALLA POR SIFONAMIENTO (FS = {fs_val:.2f})")

with tab4:
    st.header("📄 Memoria de Cálculo y Formulación MEF")
    st.markdown("""
    Este modelo utiliza el **Método de Elementos Finitos (MEF)** acoplado al motor geométrico de sólidos **OpenCASCADE (OCC)** para garantizar una simulación hiperrealista. A diferencia de las Diferencias Finitas (que asumen un medio continuo en grilla), el MEF permite generar una *excavación booleana real* para el empotramiento de la presa ($5\text{ m}$) y el grosor físico de la tablestaca, obligando al agua a modificar su trayectoria.

    ### 1. Formulación Variacional de Galerkin
    Partiendo de la Ecuación de Laplace para flujo en medios porosos $\\nabla \\cdot (k \\nabla h) = 0$, multiplicamos por una función de prueba $v$ e integramos por partes (Teorema de Green) para obtener la forma débil:
    """)
    st.latex(r"\int_{\Omega} \nabla v \cdot (k \nabla h) \, d\Omega = 0")
    st.markdown("""
    ### 2. Discretización de Malla No Estructurada (Quad-4)
    El dominio $\Omega$ se divide en elementos cuadrangulares isoparamétricos y triangulares en zonas de transición. Las coordenadas $(x, y)$ y la carga $h$ se interpolan usando funciones de forma $N_i$:
    """)
    st.latex(r"h \approx \sum N_i h_i \quad ; \quad \nabla h = [B] \{H\}")
    st.markdown("""
    El algoritmo detecta la presencia de la punta del muro e incrementa exponencialmente la densidad de nodos en esa zona geométrica, minimizando el error de truncamiento numérico donde el gradiente es más agresivo.

    ### 3. Matriz de Rigidez y Caudal Exacto ($Q$)
    La matriz de permeabilidad global se ensambla sumando las contribuciones locales:
    """)
    st.latex(r"[K] = \sum_{e} \int_{\Omega_e} [B]^T k [B] \, d\Omega")
    st.markdown("""
    En lugar de aproximar el caudal mediante derivadas simples, este modelo recupera el **Caudal Exacto** mediante las fuerzas de reacción nodales en la frontera de Dirichlet aguas arriba. Esto garantiza la conservación total de masa:
    """)
    st.latex(r"\{R\} = [K]\{H\} \quad \rightarrow \quad Q = \sum |R_{\text{entrada}}|")
    st.markdown("Los gradientes espaciales se obtienen proyectando las derivadas de los elementos a los nodos vecinos: $i_x = -\\frac{\partial h}{\partial x}$ e $i_y = -\\frac{\partial h}{\partial y}$.")
