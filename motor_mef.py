import signal
# --- PARCHE GLOBAL PARA STREAMLIT (Anti-crash) ---
def parche_signal(*args, **kwargs): pass
signal.signal = parche_signal
# --------------------------------------------------

import gmsh
import numpy as np
from skfem import MeshQuad, Basis, ElementQuad1, asm, condense, solve
from skfem.models.poisson import laplace

def resolver_mef_presa(Lx, prof_muro, pos_muro, h1, h2, k_suelo, gs, e_vacios):
    
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("MEF_Estructurado")

    x_inicio = (Lx - 15.0) / 2.0
    x_fin = x_inicio + 15.0
    x_muro = x_inicio + pos_muro
    ic_critico = (gs - 1) / (1 + e_vacios)

    # --- GEOMETRÍA DEL DOMINIO ---
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, 30, 0)
    p4 = gmsh.model.geo.addPoint(0, 30, 0)
    
    l1 = gmsh.model.geo.addLine(p1, p2) # Fondo
    l2 = gmsh.model.geo.addLine(p2, p3) # Lateral Der
    l3 = gmsh.model.geo.addLine(p3, p4) # Superficie
    l4 = gmsh.model.geo.addLine(p4, p1) # Lateral Izq
    
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # --- TABLLESTACA EMBEBIDA ---
    if prof_muro > 0:
        p_sup = gmsh.model.geo.addPoint(x_muro, 30, 0)
        p_inf = gmsh.model.geo.addPoint(x_muro, 30 - prof_muro, 0)
        l_muro = gmsh.model.geo.addLine(p_sup, p_inf)
        gmsh.model.geo.synchronize()
        # Incrustamos la línea del muro en la superficie
        gmsh.model.mesh.embed(1, [l_muro], 2, surf)

    gmsh.model.geo.synchronize()
    
    # --- IMPLEMENTACIÓN DE MALLA CUADRICULADA ESTRUCTURADA (Transfinita) ---
    # Decidimos el número de divisiones por lado
    nodes_x = 70 # Divisiones en la base
    nodes_y = 30 # Divisiones en la profundidad

    # 1. Imponemos divisiones iguales en las líneas horizontales
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nodes_x)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nodes_x)
    
    # 2. Imponemos divisiones iguales en las líneas verticales
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, nodes_y)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, nodes_y)
    
    # 3. Obligamos a la superficie a seguir estas divisiones (Cuadrícula perfecta)
    gmsh.model.geo.mesh.setTransfiniteSurface(surf)
    
    # 4. Forzamos a que TODOS los elementos sean Cuadriláteros puros
    gmsh.model.geo.mesh.setRecombine(2, surf)
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # --- EXTRACCIÓN DE DATOS PARA SCIKIT-FEM ---
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    # Mapeo rápido de Tags de Gmsh a Índices de Python
    tag2idx = {tag: i for i, tag in enumerate(nodeTags)}
    nodes = coords.reshape((-1, 3))[:, :2].T # Solo X e Y
    
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
    
    # Buscamos los elementos Quad-4 (Type 3 en Gmsh)
    if 3 in elemTypes:
        idx_quad = np.where(elemTypes == 3)[0][0]
        # Convertimos los tags de nodos a índices usando el mapeo
        elements_tags = elemNodeTags[idx_quad].reshape((-1, 4))
        elements = np.array([[tag2idx[tag] for tag in elem] for elem in elements_tags]).T
        
        m = MeshQuad(nodes, elements)
        basis = Basis(m, ElementQuad1())
    else:
        raise ValueError("Error Crítico: Gmsh falló al generar una malla de cuadriláteros estructurada.")

    # --- SOLUCIÓN MATEMÁTICA MEF ---
    K = asm(laplace, basis) * k_suelo
    
    # Condiciones de Frontera: Aguas arriba (x < x_inicio) y aguas abajo (x > x_fin) en y=30
    dofs_h1 = basis.get_dofs(lambda x: (x[0] <= x_inicio + 0.1) & (x[1] >= 29.9)).all()
    dofs_h2 = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.1) & (x[1] >= 29.9)).all()
    
    h = np.zeros(basis.N)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    frontera = np.union1d(dofs_h1, dofs_h2)
    h_sol = solve(*condense(K, np.zeros(basis.N), h, D=frontera))

    # --- POST-PROCESO Y CAUDAL ---
    grad_eval = basis.interpolate(h_sol).grad
    ix, iy = -basis.project(grad_eval[0]), -basis.project(grad_eval[1])
    imag = np.sqrt(ix**2 + iy**2)

    # Caudal Exacto (Reacciones)
    flujo_nodal = K @ h_sol
    Q = np.sum(np.abs(flujo_nodal[dofs_h1]))

    # Gradiente Máximo en la zona de salida (Talón aguas abajo)
    i_exit_max = np.max(iy[dofs_h2]) if len(dofs_h2) > 0 else 0.01

    return {
        "mesh": m, "h": h_sol, "ix": ix, "iy": iy, "imag": imag,
        "Q": Q, "fs": ic_critico / i_exit_max, "ic": ic_critico, "i_exit": i_exit_max,
        "params": (Lx, x_inicio, x_fin, x_muro, prof_muro)
    }
