import signal

# --- PARCHE GLOBAL PARA STREAMLIT ---
def parche_signal(*args, **kwargs): pass
signal.signal = parche_signal
# ------------------------------------

import gmsh
import numpy as np
# Importaciones explícitas corregidas
from skfem import MeshQuad, MeshTri, Basis, ElementQuad1, ElementTriP1, asm, condense, solve
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

    # 3. Geometría Dinámica
    p1 = gmsh.model.geo.addPoint(0, 0, 0, d_global)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, d_global)

    top_xs = sorted(list(set([0.0, x_inicio, x_muro_absoluto, x_fin, Lx])), reverse=True)
    top_pts = []
    
    for x_val in top_xs:
        d_local = d_critica if x_val in [x_inicio, x_fin, x_muro_absoluto] else d_global
        top_pts.append(gmsh.model.geo.addPoint(x_val, 30.0, 0, d_local))

    lines = []
    lines.append(gmsh.model.geo.addLine(p1, p2)) 
    lines.append(gmsh.model.geo.addLine(p2, top_pts[0])) 
    for i in range(len(top_pts)-1):
        lines.append(gmsh.model.geo.addLine(top_pts[i], top_pts[i+1])) 
    lines.append(gmsh.model.geo.addLine(top_pts[-1], p1)) 

    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # 4. Tablestaca embebida
    if prof_muro > 0:
        p_muro_inf = gmsh.model.geo.addPoint(x_muro_absoluto, 30.0 - prof_muro, 0, d_critica)
        idx_muro_sup = top_xs.index(x_muro_absoluto)
        p_muro_sup = top_pts[idx_muro_sup]
        
        l_muro = gmsh.model.geo.addLine(p_muro_sup, p_muro_inf)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [l_muro], 2, surf) 

    gmsh.model.geo.synchronize()

    # 5. Zonas de Refinamiento
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
    
    # 6. Generar Malla
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8) 
    gmsh.model.mesh.generate(2)

    # 7. Extracción blindada
    nodeTags, coord, _ = gmsh.model.mesh.getNodes()
    nodes = coord.reshape((-1, 3))[:, :2].T
    
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
    
    idx_quads = np.where(elemTypes == 3)[0]
    if len(idx_quads) > 0:
        elements = elemNodeTags[idx_quads[0]].reshape((-1, 4)) - 1
        m = MeshQuad(nodes, elements.T)
        # --- AQUÍ ESTÁ EL FIX: ElementQuad1 en lugar de ElementQuadP1 ---
        basis = Basis(m, ElementQuad1()) 
    else:
        idx_tris = np.where(elemTypes == 2)[0]
        elements = elemNodeTags[idx_tris[0]].reshape((-1, 3)) - 1
        m = MeshTri(nodes, elements.T)
        basis = Basis(m, ElementTriP1())

    # 8. Ensamblaje y Física
    K = asm(laplace, basis) * k_suelo
    
    # Física Mejorada: Extraemos los Grados de Libertad (DOFs) correctos
    dofs_h1 = basis.get_dofs(lambda x: (x[0] <= x_inicio + 0.01) & (x[1] >= 29.9)).all()
    dofs_h2 = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.01) & (x[1] >= 29.9)).all()
    
    # Inicializamos vectores usando basis.N (Total de grados de libertad matemáticos)
    h = np.zeros(basis.N)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    frontera = np.union1d(dofs_h1, dofs_h2)
    h_sol = solve(*condense(K, np.zeros(basis.N), h, D=frontera))

    return m, h_sol
