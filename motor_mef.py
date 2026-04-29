import signal

# --- PARCHE GLOBAL PARA STREAMLIT ---
# Apagamos la función de interrupción del sistema ANTES de cargar gmsh
def parche_signal(*args, **kwargs):
    pass
signal.signal = parche_signal
# ------------------------------------

import gmsh
import numpy as np
from skfem import *
from skfem.models.poisson import laplace

def resolver_mef_presa(Lx, prof_muro, pos_muro, h1, h2, k_suelo, d_global, d_critica):
    
    # 1. Inicialización segura (evita colapsos al mover los sliders)
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear() 
    gmsh.model.add("MEF_Geotecnia")

    # 2. Geometría (Puntos del contorno exterior)
    p1 = gmsh.model.geo.addPoint(0, 0, 0, d_global)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, d_global)
    p3 = gmsh.model.geo.addPoint(Lx, 30, 0, d_global)
    p4 = gmsh.model.geo.addPoint(0, 30, 0, d_global)
    
    # 3. Líneas y Superficie del Suelo
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # 4. Geometría de la Tablestaca (Barrera física)
    x_inicio = (Lx - 15) / 2
    x_fin = x_inicio + 15
    y_punta = 30 - prof_muro
    x_muro_absoluto = x_inicio + pos_muro
    
    # Incrustar el muro en la malla para que el flujo lo rodee
    if prof_muro > 0:
        p_muro_sup = gmsh.model.geo.addPoint(x_muro_absoluto, 30, 0, d_critica)
        p_muro_inf = gmsh.model.geo.addPoint(x_muro_absoluto, y_punta, 0, d_critica)
        l_muro = gmsh.model.geo.addLine(p_muro_sup, p_muro_inf)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [l_muro], 2, surf) # Fuerza la malla a respetar el muro

    # 5. Puntos de Atracción (Alta densidad de elementos finitos)
    p_toe_in = gmsh.model.geo.addPoint(x_inicio, 30, 0, d_critica)
    p_toe_out = gmsh.model.geo.addPoint(x_fin, 30, 0, d_critica)
    p_punta_attr = gmsh.model.geo.addPoint(x_muro_absoluto, y_punta, 0, d_critica)

    gmsh.model.geo.synchronize()

    # 6. Configurar la "Gravedad" de los elementos (Refinamiento)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [p_toe_in, p_toe_out, p_punta_attr])
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", d_critica)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", d_global)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 15.0)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # 7. Generar Malla de Cuadriláteros
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8) 
    gmsh.model.mesh.generate(2)

    # 8. Extraer datos para la matemática (Scikit-fem)
    nodeTags, coord, _ = gmsh.model.mesh.getNodes()
    nodes = coord.reshape((-1, 3))[:, :2].T
    
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
    # Extraemos solo los cuadriláteros (código 3 en gmsh)
    idx_quads = np.where(elemTypes == 3)[0]
    if len(idx_quads) > 0:
        elements = elemNodeTags[idx_quads[0]].reshape((-1, 4)) - 1
        m = MeshQuad(nodes, elements.T)
    else:
        # Fallback a triángulos si falla la recombinación
        idx_tris = np.where(elemTypes == 2)[0]
        elements = elemNodeTags[idx_tris[0]].reshape((-1, 3)) - 1
        m = MeshTri(nodes, elements.T)

    # 9. Solución del Sistema de Ecuaciones
    if isinstance(m, MeshQuad):
        basis = Basis(m, ElementQuadP1())
    else:
        basis = Basis(m, ElementTriP1())
        
    K = asm(laplace, basis) * k_suelo
    
    # Condiciones Frontera
    dofs_h1 = m.nodes_satisfying(lambda x: x[0] < 0.1)
    dofs_h2 = m.nodes_satisfying(lambda x: x[0] > Lx - 0.1)
    
    h = np.zeros(m.nverts)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    frontera = np.union1d(dofs_h1, dofs_h2)
    h_sol = solve(*condense(K, np.zeros(m.nverts), h, D=frontera))

    return m, h_sol
