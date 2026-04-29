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

with st.spinner("Resolviendo Sistema y Generando Vectores..."):
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
    
    # 1. Red de Flujo con Malla y Líneas de Corriente
    cf1 = ax1.tricontourf(triang, res['h'], levels=40, cmap='Blues', alpha=0.85)
    ax1.tricontour(triang, res['h'], levels=15, colors='black', linewidths=0.5) # Equipotenciales
    ax1.triplot(triang, color='white', linewidth=0.2, alpha=0.4) # VISUALIZACIÓN DE LA MALLA MEF
    
    # Generación de Líneas de Flujo (Interpolando tensores a una grilla regular)
    xi, yi = np.meshgrid(np.linspace(0, Lx, 150), np.linspace(0, 32, 80))
    interp_ix = mtri.LinearTriInterpolator(triang, res['ix'])
    interp_iy = mtri.LinearTriInterpolator(triang, res['iy'])
    ax1.streamplot(xi, yi, interp_ix(xi, yi), interp_iy(xi, yi), color='darkblue', linewidth=0.8, density=1.2, arrowsize=1.0)
    
    # Geometría Empotrada de la Presa (y=25 a y=32)
    ax1.fill([x_i, x_f, x_f, x_i], [25, 25, 32, 32], color='#333333', zorder=10)
    if prof_muro > 0: ax1.plot([x_m, x_m], [25, 25-prof_muro], color='#333333', linewidth=4, zorder=10)
        
    ax1.set_title('Red de Flujo: Equipotenciales, Líneas de Corriente y Malla Discretizada')
    
    # 2. Mapa de Gradientes
    cf2 = ax2.tricontourf(triang, res['imag'], levels=np.linspace(0, 1.2, 50), cmap='turbo', extend='max')
    plt.colorbar(cf2, ax=ax2, label='Gradiente Hidráulico (i)')
    ax2.fill([x_i, x_f, x_f, x_i], [25, 25, 32, 32], color='#333333', zorder=10)
    if prof_muro > 0: ax2.plot([x_m, x_m], [25, 25-prof_muro], color='#333333', linewidth=4, zorder=10)
    ax2.tricontour(triang, res['imag'], levels=[res['ic']], colors='red', linewidths=2.5, linestyles='dashed')
    ax2.set_title('Mapa de Calor del Gradiente Vectorial')

    for ax in [ax1, ax2]:
        ax.set_aspect('equal'); ax.set_xlim(0, Lx); ax.set_ylim(0, 32)
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
    st.header("📄 Memoria de Cálculo y Justificación Matemática")
    st.markdown("""
    El presente modelo abandona la aproximación clásica de Diferencias Finitas para implementar una solución rigurosa del campo de flujo mediante el **Método de los Elementos Finitos (MEF)**, permitiendo adaptarse a geometrías complejas (como el empotramiento de la presa) y refinar la malla en singularidades.
    
    ### 1. Ecuación Gobernante
    Asumiendo flujo bidimensional, incompresible y en régimen permanente a través de un medio poroso isotrópico, el fenómeno obedece a la ecuación de Laplace (derivada de la Ley de Darcy y la continuidad):
    """)
    st.latex(r"\nabla \cdot (k \nabla h) = k \left( \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} \right) = 0")
    
    st.markdown("""
    ### 2. Formulación Débil (Galerkin)
    Para resolver el sistema computacionalmente, se multiplica la ecuación diferencial por una función de prueba $v$ y se integra sobre el dominio $\Omega$. Aplicando el teorema de la divergencia (identidad de Green), obtenemos la forma bilineal:
    """)
    st.latex(r"\int_{\Omega} \nabla v \cdot (k \nabla h) \, d\Omega = \int_{\partial \Omega} v (k \nabla h \cdot \mathbf{n}) \, dS")
    
    st.markdown("""
    Dado que las fronteras impermeables (fondo y contornos laterales) cumplen que el flujo normal es cero $(k \nabla h \cdot \mathbf{n} = 0)$, el lado derecho de la ecuación se anula para las fronteras de Neumann naturales.

    ### 3. Discretización del Dominio
    El dominio continuo se discretiza utilizando elementos **Cuadriláteros Bilineales de 4 nodos (Quad-4)**, generados mediante el algoritmo *Frontal-Quad*. La carga hidráulica $h$ dentro de cada elemento se interpola a partir de sus valores nodales usando funciones de forma $N_i$:
    """)
    st.latex(r"h(x,y) \approx \sum_{i=1}^{4} N_i(\xi, \eta) h_i")
    
    st.markdown("""
    ### 4. Matriz de Rigidez Local y Ensamblaje
    Sustituyendo la interpolación en la formulación de Galerkin, el sistema se reduce a un problema algebraico lineal $[K]\{H\} = \{F\}$. Para un elemento $e$ genérico con permeabilidad constante $k$, su matriz de permeabilidad (o rigidez hidraúlica) local se calcula mediante la integral jacobiana:
    """)
    st.latex(r"[k^{(e)}] = \int_{-1}^{1} \int_{-1}^{1} [B]^T k [B] |J| \, d\xi \, d\eta")
    
    st.markdown("""
    Donde $[B]$ es la matriz de derivadas de las funciones de forma. El programa utiliza la librería `scikit-fem` para realizar la cuadratura de Gauss y ensamblar estas matrices locales en la matriz global rala $[K]$.
    
    ### 5. Ejemplo de Solución Nodal
    Las condiciones de frontera de Dirichlet (Carga constante) se imponen modificando la matriz ensamblada. Por ejemplo, si el Nodo 1 está aguas arriba ($h=50\text{ m}$) y el Nodo 2 está aguas abajo ($h=5\text{ m}$), las ecuaciones del sistema global toman la forma:
    """)
    st.latex(r"""
    \begin{bmatrix}
    1 & 0 & \cdots & 0 \\
    0 & 1 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    K_{i1} & K_{i2} & \cdots & K_{in}
    \end{bmatrix}
    \begin{Bmatrix}
    h_1 \\ h_2 \\ \vdots \\ h_i
    \end{Bmatrix}
    =
    \begin{Bmatrix}
    50 \\ 5 \\ \vdots \\ 0
    \end{Bmatrix}
    """)
    st.markdown("""
    Al resolver este sistema lineal usando `scipy.sparse.linalg.spsolve`, obtenemos el vector de cargas. Posteriormente, los gradientes en cada elemento se calculan directamente proyectando las derivadas de las funciones de forma: $i = -\nabla h$.
    """)
