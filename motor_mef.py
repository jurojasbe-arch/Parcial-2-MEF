import signal
def parche_signal(*args, **kwargs): pass
signal.signal = parche_signal

import gmsh
import numpy as np
from skfem import MeshQuad, MeshTri, Basis, ElementQuad1, ElementTriP1, asm, condense, solve
from skfem.models.poisson import laplace

def resolver_mef_presa(Lx, prof_muro, pos_muro, h1, h2, k_suelo, d_global, d_critica, gs, e_vacios):
    if not gmsh.isInitialized(): gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("MEF_OCC")

    x_inicio = (Lx - 15.0) / 2.0
    x_fin = x_inicio + 15.0
    x_muro = x_inicio + pos_muro
    ic_critico = (gs - 1) / (1 + e_vacios)

    # --- GEOMETRÍA DE SÓLIDOS (OpenCASCADE) ---
    # 1. Creamos el macizo de suelo completo (hasta y=30)
    gmsh.model.occ.addRectangle(0, 0, 0, Lx, 30, tag=1)
    
    # 2. Creamos el bloque de la presa (Empotrada 5m: de y=25 a y=30)
    gmsh.model.occ.addRectangle(x_inicio, 25, 0, 15, 5, tag=2)
    
    tags_a_restar = [(2, 2)]
    
    # 3. Creamos la tablestaca con GROSOR REAL (0.2m)
    if prof_muro > 0:
        gmsh.model.occ.addRectangle(x_muro - 0.1, 25 - prof_muro, 0, 0.2, prof_muro, tag=3)
        tags_a_restar.append((2, 3))

    # 4. OPERACIÓN BOOLEANA: Restamos la presa y tablestaca del suelo
    # Esto crea un hueco físico infranqueable para el agua
    gmsh.model.occ.cut([(2, 1)], tags_a_restar, removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    # --- REFINAMIENTO DE MALLA ---
    puntos = gmsh.model.getEntities(0)
    pts_criticos = []
    for dim, tag in puntos:
        x, y, z = gmsh.model.getValue(dim, tag, [])
        # Refinar en el talón de salida y en la punta del muro
        if (abs(x - x_fin) < 0.1 and abs(y - 25) < 0.1) or \
           (prof_muro > 0 and abs(y - (25 - prof_muro)) < 0.1 and abs(x - x_muro) < 0.5):
            pts_criticos.append(tag)

    if pts_criticos:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", pts_criticos)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", d_critica)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", d_global)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 10.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(2)

    # --- EXTRACCIÓN ROBUSTA DE NODOS ---
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    tag2idx = {tag: i for i, tag in enumerate(nodeTags)}
    nodes = coords.reshape((-1, 3))[:, :2].T
    
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
    
    if 3 in elemTypes: # Cuadriláteros
        idx = np.where(elemTypes == 3)[0][0]
        elements = np.array([tag2idx[t] for t in elemNodeTags[idx]]).reshape((-1, 4)).T
        m = MeshQuad(nodes, elements)
        basis = Basis(m, ElementQuad1())
    else: # Triángulos (Fallback)
        idx = np.where(elemTypes == 2)[0][0]
        elements = np.array([tag2idx[t] for t in elemNodeTags[idx]]).reshape((-1, 3)).T
        m = MeshTri(nodes, elements)
        basis = Basis(m, ElementTriP1())

    # --- FÍSICA Y SOLUCIÓN ---
    K = asm(laplace, basis) * k_suelo
    
    # El agua entra y sale por la superficie (y=30)
    dofs_h1 = basis.get_dofs(lambda x: (x[0] <= x_inicio + 0.01) & (x[1] >= 29.99)).all()
    dofs_h2 = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.01) & (x[1] >= 29.99)).all()
    
    h = np.zeros(basis.N)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    frontera = np.union1d(dofs_h1, dofs_h2)
    h_sol = solve(*condense(K, np.zeros(basis.N), h, D=frontera))

    # --- POST-PROCESO ---
    grad_eval = basis.interpolate(h_sol).grad
    ix, iy = -basis.project(grad_eval[0]), -basis.project(grad_eval[1])
    imag = np.sqrt(ix**2 + iy**2)

    # CAUDAL EXACTO (Por fuerzas de reacción de la matriz)
    flujo_nodal = K @ h_sol
    Q = np.sum(np.abs(flujo_nodal[dofs_h1]))

    # Gradiente de salida máximo (Revisando los primeros 5 metros aguas abajo)
    dofs_salida = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.1) & (x[0] <= x_fin + 5.0) & (x[1] >= 29.99)).all()
    i_exit_max = np.max(iy[dofs_salida]) if len(dofs_salida) > 0 else 0.01
    if i_exit_max <= 0: i_exit_max = 0.001

    return {
        "mesh": m, "h": h_sol, "ix": ix, "iy": iy, "imag": imag,
        "Q": Q, "fs": ic_critico / i_exit_max, "ic": ic_critico, "i_exit": i_exit_max,
        "params": (Lx, x_inicio, x_fin, x_muro, prof_muro)
    }
