import signal

# --- PARCHE GLOBAL PARA STREAMLIT ---
def parche_signal(*args, **kwargs): pass
signal.signal = parche_signal
# ------------------------------------

import gmsh
import numpy as np
from skfem import *
from skfem.models.poisson import laplace

def resolver_mef_presa(Lx, prof_muro, pos_muro, h1, h2, k_suelo, d_global, d_critica):
    
    # 1. Inicialización segura
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear() 
    gmsh.model.add("MEF_Geotecnia")

    # 2. Coordenadas clave
    x_inicio = (Lx - 15.0) / 2.0
    x_fin = x_inicio + 15.0
    x_muro_absoluto = x_inicio + pos_muro

    # 3. Geometría Dinámica a prueba de fallos topológicos
    p1 = gmsh.model.geo.addPoint(0, 0, 0, d_global)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, d_global)

    # Encontrar todos los puntos únicos de la superficie (evita duplicados si pos_muro es 0 o 15)
    top_xs = sorted(list(set([0.0, x_inicio, x_muro_absoluto, x_fin, Lx])), reverse=True)
    top_pts = []
    
    # Crear puntos de la superficie
    for x_val in top_xs:
        d_local = d_critica if x_val in [x_inicio, x_fin, x_muro_absoluto] else d_global
        top_pts.append(gmsh.model.geo.addPoint(x_val, 30.0, 0, d_local))

    # Conectar el loop exterior
    lines = []
    lines.append(gmsh.model.geo.addLine(p1, p2)) # Fondo
    lines.append(gmsh.model.geo.addLine(p2, top_pts[0])) # Lateral Der
    for i in range(len(top_pts)-1):
        lines.append(gmsh.model.geo.addLine(top_pts[i], top_pts[i+1])) # Superficie
    lines.append(gmsh.model.geo.addLine(top_pts[-1], p1)) # Lateral Izq

    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # 4. Tablestaca embebida de forma segura
    if prof_muro > 0:
        p_muro_inf = gmsh.model.geo.addPoint(x_muro_absoluto, 30.0 - prof_muro, 0, d_critica)
        # Buscar el ID del punto superior que ya pertenece a la frontera
        idx_muro_sup = top_xs.index(x_muro_absoluto)
        p_muro_sup = top_pts[idx_muro_sup]
        
        l_muro = gmsh.model.geo.addLine(p_muro_sup, p_muro_inf)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [l_muro], 2, surf) # Ahora sí incrusta sin romper la malla

    gmsh.model.geo.synchronize()

    # 5. Zonas de Refinamiento (Atracción)
    gmsh.model.mesh.field.add("Distance", 1)
    attr_pts = [top_pts[top_xs.index(x_inicio)], top_pts[top_xs.index(x_fin)]]
    if prof_muro > 0:
        attr_pts.append(p_muro_inf)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", attr_pts)
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", d_critica)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", d_global)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 15.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # 6. Generar Malla de Cuadriláteros
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8) 
    gmsh.model.mesh.generate(2)

    # 7. Extracción blindada de nodos para Scikit-fem
    nodeTags, coord, _ = gmsh.model.mesh.getNodes()
    nodes = coord.reshape((-1, 3))[:, :2].T
    
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
    
    if len(elemTypes) == 0:
        raise ValueError("Error crítico: Gmsh no generó elementos 2D.")

    # Priorizar cuadriláteros, fallback a triángulos si es necesario
    idx_quads = np.where(elemTypes == 3)[0]
    if len(idx_quads) > 0:
        elements = elemNodeTags[idx_quads[0]].reshape((-1, 4)) - 1
        m = MeshQuad(nodes, elements.T)
        basis = Basis(m, ElementQuadP1())
    else:
        idx_tris = np.where(elemTypes == 2)[0]
        elements = elemNodeTags[idx_tris[0]].reshape((-1, 3)) - 1
        m = MeshTri(nodes, elements.T)
        basis = Basis(m, ElementTriP1())

    # 8. Ensamblaje y Física
    K = asm(laplace, basis) * k_suelo
    
    # Física Mejorada: Todo el lecho expuesto recibe carga de agua
    dofs_h1 = m.nodes_satisfying(lambda x: (x[0] <= x_inicio + 0.01) & (x[1] >= 29.9))
    dofs_h2 = m.nodes_satisfying(lambda x: (x[0] >= x_fin - 0.01) & (x[1] >= 29.9))
    
    h = np.zeros(m.nverts)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    frontera = np.union1d(dofs_h1, dofs_h2)
    h_sol = solve(*condense(K, np.zeros(m.nverts), h, D=frontera))

    return m, h_sol
