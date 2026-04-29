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
    gmsh.model.add("MEF_Geotecnia")

    x_inicio, x_fin = (Lx - 15.0) / 2.0, (Lx + 15.0) / 2.0
    x_muro_absoluto = x_inicio + pos_muro
    ic_critico = (gs - 1) / (1 + e_vacios)

    # --- Geometría Física con Empotramiento (5m) ---
    pts_coords = [
        (0, 0, d_global), (Lx, 0, d_global), (Lx, 30, d_global),
        (x_fin, 30, d_critica), (x_fin, 25, d_critica), # Escalón derecho
        (x_muro_absoluto, 25, d_critica),               # Base del muro
        (x_inicio, 25, d_critica), (x_inicio, 30, d_critica), # Escalón izquierdo
        (0, 30, d_global)
    ]
    
    # Limpieza de duplicados si el muro está en un borde
    clean_coords = [pts_coords[0]]
    for c in pts_coords[1:]:
        if c[:2] != clean_coords[-1][:2]: clean_coords.append(c)

    top_pts = [gmsh.model.geo.addPoint(x, y, 0, d) for x, y, d in clean_coords]
    lines = [gmsh.model.geo.addLine(top_pts[i], top_pts[(i+1)%len(top_pts)]) for i in range(len(top_pts))]
    surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])

    # --- Tablestaca embebida ---
    p_inf = None
    if prof_muro > 0:
        p_inf = gmsh.model.geo.addPoint(x_muro_absoluto, 25.0 - prof_muro, 0, d_critica)
        idx_muro_sup = next(i for i, c in enumerate(clean_coords) if c[0] == x_muro_absoluto and c[1] == 25.0)
        l_muro = gmsh.model.geo.addLine(top_pts[idx_muro_sup], p_inf)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [l_muro], 2, surf)

    gmsh.model.geo.synchronize()
    
    # --- Refinamiento ---
    gmsh.model.mesh.field.add("Distance", 1)
    puntos_atraccion = [top_pts[next(i for i, c in enumerate(clean_coords) if c[:2] == (x_inicio, 25))],
                        top_pts[next(i for i, c in enumerate(clean_coords) if c[:2] == (x_fin, 25))]]
    if p_inf: puntos_atraccion.append(p_inf)
    
    gmsh.model.mesh.field.setNumbers(1, "PointsList", puntos_atraccion)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", d_critica); gmsh.model.mesh.field.setNumber(2, "SizeMax", d_global)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0); gmsh.model.mesh.field.setNumber(2, "DistMax", 15.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    gmsh.option.setNumber("Mesh.RecombineAll", 1); gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(2)

    # --- Exportar a Scikit-fem ---
    nodes = gmsh.model.mesh.getNodes()[1].reshape((-1, 3))[:, :2].T
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
    
    if 3 in elemTypes:
        m = MeshQuad(nodes, elemNodeTags[np.where(elemTypes==3)[0][0]].reshape((-1, 4)).T - 1)
        basis = Basis(m, ElementQuad1())
    else:
        m = MeshTri(nodes, elemNodeTags[np.where(elemTypes==2)[0][0]].reshape((-1, 3)).T - 1)
        basis = Basis(m, ElementTriP1())

    # --- Solución Matemática ---
    K = asm(laplace, basis) * k_suelo
    dofs_h1 = basis.get_dofs(lambda x: (x[0] <= x_inicio + 0.1) & (x[1] >= 29.9)).all()
    dofs_h2 = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.1) & (x[1] >= 29.9)).all()
    h = np.zeros(basis.N); h[dofs_h1], h[dofs_h2] = h1, h2
    h_sol = solve(*condense(K, np.zeros(basis.N), h, D=np.union1d(dofs_h1, dofs_h2)))

    # --- Vectores y Post-proceso ---
    grad_eval = basis.interpolate(h_sol).grad
    ix, iy = -basis.project(grad_eval[0]), -basis.project(grad_eval[1])
    imag = np.sqrt(ix**2 + iy**2)
    Q = np.sum(k_suelo * ix[dofs_h1]) * (30.0 / len(dofs_h1))
    i_exit_max = np.max(iy[dofs_h2]) if len(dofs_h2) > 0 else 0.01

    return {
        "mesh": m, "h": h_sol, "ix": ix, "iy": iy, "imag": imag,
        "Q": Q, "fs": ic_critico / i_exit_max, "ic": ic_critico, "i_exit": i_exit_max,
        "params": (Lx, x_inicio, x_fin, x_muro_absoluto, prof_muro)
    }
