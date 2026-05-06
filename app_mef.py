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
    # EL FIX ABSOLUTO: Obligar a Matplotlib a respetar los cuadriláteros (SIN inventar triángulos en los huecos)
    t_quads = res['mesh'].t.T
    triangles = np.vstack([t_quads[:, [0, 1, 2]], t_quads[:, [0, 2, 3]]])
    triang = mtri.Triangulation(res['mesh'].p[0], res['mesh'].p[1], triangles)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico A: Red de Flujo
    cf1 = ax1.tricontourf(triang, res['h'], levels=40, cmap='Blues', alpha=0.9)
    ax1.tricontour(triang, res['h'], levels=15, colors='black', linewidths=0.6)
    
    xi, yi = np.meshgrid(np.linspace(0, Lx, 250), np.linspace(0, 30, 120))
    interp_ix = mtri.LinearTriInterpolator(triang, res['ix'])
    interp_iy = mtri.LinearTriInterpolator(triang, res['iy'])
    
    # Al no haber triángulos en la tablestaca, la interpolación arroja NaN automáticamente y las flechas la rodean
    ax1.streamplot(xi, yi, interp_ix(xi, yi), interp_iy(xi, yi), color='darkblue', linewidth=1.0, density=1.8)
    
    verts = res['mesh'].p[:, res['mesh'].t].T
    ax1.add_collection(PolyCollection(verts, edgecolors='black', facecolors='none', linewidths=0.2, alpha=0.4))
    
    ax1.fill([x_i, x_f, x_f, x_i], [25, 25, 30, 30], color='#444444', zorder=10)
    if prof_muro > 0: ax1.fill([x_m-grosor/2, x_m+grosor/2, x_m+grosor/2, x_m-grosor/2], [25-prof_muro, 25-prof_muro, 25, 25], color='#222222', zorder=10)
    ax1.set_title('Malla Estructurada Topológica: El agua rodea físicamente el obstáculo')
    
    # Gráfico B: Mapa de Gradientes Exacto
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

    # Función para inyectar cortes transversales del campo de presiones en 3D
    def agregar_corte_flujo(depth_y, show_colorbar):
        fig3d.add_trace(go.Mesh3d(
            x=res['mesh'].p[0], y=np.full_like(res['mesh'].p[0], depth_y), z=res['mesh'].p[1],
            i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
            intensity=res['h'], colorscale='Turbo', showscale=show_colorbar, name=f'Corte Flujo (Y={depth_y}m)'
        ))

    # Generamos 3 tajadas para visualizar el flujo dentro del volumen
    agregar_corte_flujo(0, True)
    agregar_corte_flujo(ancho_3d / 2, False)
    agregar_corte_flujo(ancho_3d, False)

    # Creador de bloques de concreto/acero 3D
    def crear_bloque(x0, x1, elev0, elev1, depth0, depth1, color, nombre):
        return go.Mesh3d(
            x=[x0, x0, x1, x1, x0, x0, x1, x1], y=[depth0, depth0, depth0, depth0, depth1, depth1, depth1, depth1], z=[elev0, elev1, elev1, elev0, elev0, elev1, elev1, elev0],
            alphahull=0, color=color, flatshading=True, name=nombre
        )

    # Estructuras Físicas
    fig3d.add_trace(crear_bloque(x_i, x_f, 25, 30, 0, ancho_3d, '#888888', 'Presa de Concreto'))
    if prof_muro > 0: fig3d.add_trace(crear_bloque(x_m-grosor/2, x_m+grosor/2, 25-prof_muro, 25, 0, ancho_3d, '#222222', 'Tablestaca de Acero'))

    fig3d.update_layout(
        scene=dict(
            xaxis_title='Longitud X (m)', yaxis_title='Profundidad / Ancho Z (m)', zaxis_title='Elevación (m)', 
            aspectmode='manual', aspectratio=dict(x=3, y=1.5, z=1)
        ), 
        height=750, template="plotly_dark", margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig3d, use_container_width=True)

with tab3:
    st.header("Evaluación Estructural de Sifonamiento")
    if fs_val > 1.5: st.success(f"✅ ESTABLE (FS = {fs_val:.2f})")
    elif fs_val > 1.0: st.warning(f"⚠️ CRÍTICO (FS = {fs_val:.2f})")
    else: st.error(f"🚨 FALLA INMINENTE POR TUBIFICACIÓN (FS = {fs_val:.2f})")

with tab4:
    st.header("📄 Memoria de Cálculo y Marco Teórico MEF")
    
    st.markdown(r"""
El método desarrollado resuelve el campo bidimensional de presiones intersticiales empleando el **Método de los Elementos Finitos (MEF)** bajo una formulación tensorial estricta, lo que garantiza resultados numéricos superiores a los esquemas clásicos de Diferencias Finitas.

### 1. Ecuación Gobernante y Régimen de Flujo
La filtración de fluidos en un medio poroso isotrópico y homogéneo bajo un régimen laminar obedece a la integración de la Ley de Darcy junto con la ecuación de conservación de masa, dando como resultado la Ecuación Diferencial Parcial (EDP) de Laplace:
$$ \nabla \cdot (k \nabla h) = 0 \quad \Rightarrow \quad \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} = 0 $$

### 2. Formulación Débil (Método de Galerkin)
Dado que la solución analítica exacta de la EDP es imposible para geometrías complejas (como la inclusión asimétrica de una tablestaca), se procede a la minimización del error residual. Multiplicando por una función de peso $v(x,y)$ e integrando por partes en el dominio $\Omega$ usando el Teorema de Green, obtenemos la forma variacional débil:
$$ \int_{\Omega} \left( \frac{\partial v}{\partial x} k \frac{\partial h}{\partial x} + \frac{\partial v}{\partial y} k \frac{\partial h}{\partial y} \right) d\Omega = 0 $$
En este proyecto, las fronteras laterales y de fondo se asumen impermeables, por lo que el término de flujo natural en la frontera Neumann se vuelve idéntico a cero.

### 3. Discretización Espacial y Topología Booleana
El medio se discretiza mediante una grilla tensorial estructurada conformada estrictamente por **elementos cuadriláteros bilineales de 4 nodos (Quad-4)**. 
A diferencia de programas comerciales estándar, este algoritmo genera la matriz y posteriormente ejecuta una **operación booleana de substracción**, eliminando físicamente los elementos que interfieren con la cimentación de la presa y el grosor de la tablestaca. La variable de estado $h$ se interpola usando funciones de forma isoparamétricas $N_i(\xi, \eta)$:
$$ h^{(e)}(\xi, \eta) \approx \sum_{i=1}^{4} N_i(\xi, \eta) h_i \quad ; \quad N_i = \frac{1}{4}(1 \pm \xi)(1 \pm \eta) $$

### 4. Solución del Sistema y Caudal Exacto
Las matrices locales se acoplan en la Matriz Global de Rigidez $[K]$. Imponiendo las cargas de frontera $H_1$ y $H_2$, se resuelve el sistema lineal:
$$ [K_{ff}] \{H_f\} = \{F\} - [K_{fc}] \{H_c\} $$
El caudal de infiltración exacto $Q$ se extrae sumando la matriz de fuerzas de reacción nodales en el lecho aguas arriba, garantizando la conservación perfecta de masa:
$$ Q = \sum | \{R_{inlet}\} | $$
    """)
