import gmsh
import numpy as np
from skfem import *
from skfem.models.poisson import laplace
from scipy.sparse.linalg import spsolve
import signal

def resolver_mef_presa(Lx, prof_muro, pos_muro, h1, h2, k_suelo, d_global, d_critica):
    
    # --- FIX DEFINITIVO PARA STREAMLIT (Monkey Patch) ---
    original_signal = signal.signal
    signal.signal = lambda *args, **kwargs: None
    try:
        gmsh.initialize()
    finally:
        signal.signal = original_signal
    # ----------------------------------------------------
    gmsh.model.add("MEF_Geotecnia")
    # ...

    # Geometría del Dominio (30m de profundidad)
    p1 = gmsh.model.geo.addPoint(0, 0, 0, d_global)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, d_global)
    p3 = gmsh.model.geo.addPoint(Lx, 30, 0, d_global)
    p4 = gmsh.model.geo.addPoint(0, 30, 0, d_global)
    
    # Líneas contorno
    l1 = gmsh.model.geo.addLine(p1, p2) # Fondo (Neumann natural)
    l2 = gmsh.model.geo.addLine(p2, p3) # Salida Derecha
    l3 = gmsh.model.geo.addLine(p3, p4) # Superficie
    l4 = gmsh.model.geo.addLine(p4, p1) # Entrada Izquierda

    # Definición de la Presa y Tablestaca (No-flujo)
    x_inicio = (Lx - 15) / 2
    x_fin = x_inicio + 15
    # Puntos de la estructura para refinamiento
    p_toe_in = gmsh.model.geo.addPoint(x_inicio, 30, 0, d_critica)
    p_toe_out = gmsh.model.geo.addPoint(x_fin, 30, 0, d_critica)
    p_punta = gmsh.model.geo.addPoint(x_inicio + pos_muro, 25 - prof_muro, 0, d_critica)

    # REFINAMIENTO: Campos de distancia (Attractors)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [p_toe_in, p_toe_out, p_punta])
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", d_critica)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", d_global)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 10)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.geo.synchronize()
    
    # Generar Malla de Cuadriláteros (Quad-4)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8) 
    gmsh.model.mesh.generate(2)

    # Extraer Malla para Scikit-fem
    nodes = gmsh.model.mesh.getNodes()[1].reshape((-1, 3))[:, :2].T
    elements = gmsh.model.mesh.getElements(2)[2][0].reshape((-1, 4)) - 1
    
    m = MeshQuad(nodes, elements.T)
    gmsh.finalize()

    # Ensamblaje MEF
    basis = Basis(m, ElementQuadP1())
    K = asm(laplace, basis) * k_suelo
    
    # Condiciones de Frontera (Dirichlet)
    # h = h1 a la izquierda, h = h2 a la derecha
    # Simplificación para el ejemplo: nodos en x=0 y x=Lx
    dofs_h1 = m.nodes_satisfying(lambda x: x[0] < 0.1)
    dofs_h2 = m.nodes_satisfying(lambda x: x[0] > Lx - 0.1)
    
    h = np.zeros(m.nverts)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    free_dofs = basis.complement_dofs(np.union1d(dofs_h1, dofs_h2))
    h = solve(*condense(K, np.zeros(m.nverts), h, D=np.union1d(dofs_h1, dofs_h2)))

    return m, h
