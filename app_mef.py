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
    prof = st.slider("Prof. Muro (m)", 0.0, 20.0, 10.0, 1.0)
    pos_muro = st.slider("Posición Muro (m)", 0.0, 15.0, 5.0, 0.5)
    ancho_3d = st.slider("Ancho Z (Modelo 3D)", 10.0, 100.0, 30.0, 5.0)
    
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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Red de Flujo y Malla", "🧊 Modelo 3D Físico", "🛡️ Seguridad", "📄 Memoria de Cálculo Detallada"])
Lx, x_i, x_f, x_m, prof_muro = res['params']

with tab1:
    # 1. GENERACIÓN DE LA MÁSCARA BOOLEANA (Corrige el error visual)
    triang = mtri.Triangulation(res['mesh'].p[0], res['mesh'].p[1])
    x_tri = res['mesh'].p[0, triang.triangles].mean(axis=1)
    y_tri = res['mesh'].p[1, triang.triangles].mean(axis=1)
    
    mask_presa = (x_tri >= x_i) & (x_tri <= x_f) & (y_tri >= 25.0)
    mask_muro = (x_tri >= x_m - 0.25) & (x_tri <= x_m + 0.25) & (y_tri >= 25.0 - prof_muro)
    triang.set_mask(mask_presa | mask_muro) # ¡Esto prohíbe dibujar en el hueco!

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico A: Red de Flujo
    cf1 = ax1.tricontourf(triang, res['h'], levels=40, cmap='Blues', alpha=0.9)
    ax1.tricontour(triang, res['h'], levels=15, colors='black', linewidths=0.6)
    
    xi, yi = np.meshgrid(np.linspace(0, Lx, 200), np.linspace(0, 30, 100))
    interp_ix = mtri.LinearTriInterpolator(triang, res['ix'])
    interp_iy = mtri.LinearTriInterpolator(triang, res['iy'])
    ax1.streamplot(xi, yi, interp_ix(xi, yi), interp_iy(xi, yi), color='darkblue', linewidth=1.0, density=1.5)
    
    verts = res['mesh'].p[:, res['mesh'].t].T
    ax1.add_collection(PolyCollection(verts, edgecolors='black', facecolors='none', linewidths=0.2, alpha=0.4))
    
    ax1.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0: ax1.fill([x_m-0.2, x_m+0.2, x_m+0.2, x_m-0.2], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
    ax1.set_title('Malla Estructurada Cuadriculada con Equipotenciales y Líneas de Flujo')
    
    # Gráfico B: Mapa de Gradientes
    interp_imag = mtri.LinearTriInterpolator(triang, res['imag'])
    cf2 = ax2.contourf(xi, yi, interp_imag(xi, yi), levels=np.linspace(0, res['ic']*1.2, 50), cmap='turbo', extend='max')
    plt.colorbar(cf2, ax=ax2, label='Gradiente Hidráulico (i)')
    ax2.contour(xi, yi, interp_imag(xi, yi), levels=[res['ic']], colors='red', linewidths=2.5, linestyles='dashed')
    ax2.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0: ax2.fill([x_m-0.2, x_m+0.2, x_m+0.2, x_m-0.2], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
    ax2.set_title('Concentración Vectorial de Gradientes (Foco de Singularidad en la Punta)')

    for ax in [ax1, ax2]: ax.set_aspect('equal'); ax.set_xlim(0, Lx); ax.set_ylim(0, 30)
    st.pyplot(fig)

with tab2:
    st.subheader("Modelación CAD Física en 3D")
    fig3d = go.Figure()

    # Proyección de Cargas sobre el Bloque Físico 3D
    HI_3d = interp_h_3d = mtri.LinearTriInterpolator(triang, res['h'])(xi, yi)
    mask_3d = (xi >= x_i) & (xi <= x_f) & (yi >= 25.0) | ((xi >= x_m - 0.25) & (xi <= x_m + 0.25) & (yi >= 25.0 - prof_muro))
    HI_3d[mask_3d] = np.nan # Recortar geometría
    
    # Caras del Terreno
    fig3d.add_trace(go.Surface(x=xi, y=np.zeros_like(xi), z=yi, surfacecolor=HI_3d, colorscale='Blues', cmin=0, cmax=h1, name='Frente'))
    fig3d.add_trace(go.Surface(x=xi, y=np.full_like(xi, ancho_3d), z=yi, surfacecolor=HI_3d, colorscale='Blues', showscale=False, name='Fondo'))

    # Función creadora de bloques convexos 3D
    def crear_bloque(x0, x1, y0, y1, z0, z1, color, nombre):
        return go.Mesh3d(x=[x0, x0, x1, x1, x0, x0, x1, x1], y=[y0, y1, y1, y0, y0, y1, y1, y0], z=[z0, z0, z0, z0, z1, z1, z1, z1], alphahull=0, color=color, flatshading=True, name=nombre)

    fig3d.add_trace(crear_bloque(x_i, x_f, 0, ancho_3d, 25, 30, '#888888', 'Presa'))
    if prof_muro > 0: fig3d.add_trace(crear_bloque(x_m-0.2, x_m+0.2, 0, ancho_3d, 25-prof_muro, 25, '#222222', 'Tablestaca'))

    fig3d.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Ancho Z (m)', zaxis_title='Elevación Y (m)', aspectmode='data'), height=700, template="plotly_dark")
    st.plotly_chart(fig3d, use_container_width=True)

with tab3:
    st.header("Evaluación Estructural de Sifonamiento")
    if fs_val > 1.5: st.success(f"✅ ESTABLE (FS = {fs_val:.2f})")
    elif fs_val > 1.0: st.warning(f"⚠️ CRÍTICO (FS = {fs_val:.2f})")
    else: st.error(f"🚨 FALLA INMINENTE POR TUBIFICACIÓN (FS = {fs_val:.2f})")

with tab4:
    st.header("📄 Memoria de Cálculo y Marco Teórico MEF")
    
    st.markdown("""
    El método desarrollado resuelve el campo bidimensional de presiones intersticiales empleando el **Método de los Elementos Finitos (MEF)** bajo una formulación tensorial estricta, lo que garantiza resultados numéricos superiores a los esquemas clásicos de Diferencias Finitas.
    """)
    
    st.subheader("1. Ecuación Gobernante y Régimen de Flujo")
    st.markdown("La filtración de fluidos en un medio poroso isotrópico y homogéneo bajo un régimen laminar obedece a la integración de la Ley de Darcy junto con la ecuación de conservación de masa, dando como resultado la Ecuación Diferencial Parcial (EDP) de Laplace:")
    st.latex(r"\nabla \cdot (k \nabla h) = 0 \quad \Rightarrow \quad \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} = 0")
    
    st.subheader("2. Formulación Débil (Método de Galerkin)")
    st.markdown("Dado que la solución analítica exacta de la EDP es imposible para geometrías complejas (como la inclusión asimétrica de una tablestaca), se procede a la minimización del error residual. Multiplicando por una función de peso $v(x,y)$ e integrando por partes en el dominio $\Omega$ usando el Teorema de Green, obtenemos la forma variacional débil:")
    st.latex(r"\int_{\Omega} \left( \frac{\partial v}{\partial x} k \frac{\partial h}{\partial x} + \frac{\partial v}{\partial y} k \frac{\partial h}{\partial y} \right) d\Omega = \int_{\Gamma_q} v \bar{q} d\Gamma")
    st.markdown("En este proyecto, las fronteras laterales y de fondo se asumen impermeables, por lo que el término de flujo natural en la frontera Neumann $\int_{\Gamma_q}$ se vuelve idéntico a cero.")

    st.subheader("3. Discretización Espacial y Funciones de Forma")
    st.markdown("El medio se discretizó mediante una grilla tensorial estructurada conformada estrictamente por **elementos cuadriláteros bilineales de 4 nodos (Quad-4)**. La variable de estado $h$ dentro de cada elemento $e$ se interpola de sus valores nodales usando funciones de forma isoparamétricas $N_i(\xi, \eta)$:")
    st.latex(r"h^{(e)}(\xi, \eta) \approx \sum_{i=1}^{4} N_i(\xi, \eta) h_i \quad ; \quad N_i = \frac{1}{4}(1 \pm \xi)(1 \pm \eta)")

    st.subheader("4. Matriz Jacobiana e Integración Numérica")
    st.markdown("Para integrar sobre cuadriláteros deformados topológicamente, se mapea el espacio físico $(x,y)$ al espacio natural $(\xi, \eta)$ utilizando la Matriz Jacobiana $[J]$. La matriz de rigidez hidráulica del elemento $[K_e]$ se obtiene numéricamente (mediante Cuadratura de Gauss):")
    st.latex(r"[K^{(e)}] = \int_{-1}^{1} \int_{-1}^{1} [B]^T k [B] \cdot \det|J| \, d\xi \, d\eta")
    st.markdown("Donde $[B]$ es la matriz que contiene las derivadas espaciales de las funciones de forma $\\nabla N_i$.")

    st.subheader("5. Topología de Excavación Booleana y Ensamblaje")
    st.markdown("A diferencia de programas comerciales estándar, este algoritmo crea una red densa y posteriormente ejecuta una **operación booleana de substracción**, eliminando físicamente los elementos que interfieren con la cimentación de la presa y el grosor de la tablestaca. Tras la eliminación de los nodos huérfanos, las matrices locales se acoplan en la Matriz Global de Rigidez $[K]$ de tamaño $(n \times n)$ grados de libertad.")

    st.subheader("6. Solución del Sistema y Post-Procesamiento Vectorial")
    st.markdown("Imponiendo las condiciones de frontera de Dirichlet (Cargas $H_1$ y $H_2$ en la superficie), se resuelve el sistema de ecuaciones algebraicas:")
    st.latex(r"[K_{ff}] \{H_f\} = \{F\} - [K_{fc}] \{H_c\} \quad \Rightarrow \quad \{H_f\} = [K_{ff}]^{-1} (\{F\} - [K_{fc}] \{H_c\})")
    st.markdown("Finalmente, los vectores de gradiente local se recuperan proyectando la matriz de derivadas sobre el campo de presiones resuelto, y el caudal de infiltración exacto $Q$ se extrae sumando la matriz de fuerzas de reacción nodales en el lecho aguas arriba, garantizando la conservación perfecta de masa:")
    st.latex(r"\mathbf{i} = - \nabla h = - [B]\{H\} \quad ; \quad Q = \sum | \{R_{inlet}\} |")
