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

    # --- Geometría ---
    p1 = gmsh.model.geo.addPoint(0, 0, 0, d_global)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, d_global)
    top_xs = sorted(list(set([0.0, x_inicio, x_muro_absoluto, x_fin, Lx])), reverse=True)
    top_pts = [gmsh.model.geo.addPoint(x, 30.0, 0, d_critica if x in [x_inicio, x_fin, x_muro_absoluto] else d_global) for x in top_xs]
    
    lines = [gmsh.model.geo.addLine(p1, p2), gmsh.model.geo.addLine(p2, top_pts[0])]
    for i in range(len(top_pts)-1): lines.append(gmsh.model.geo.addLine(top_pts[i], top_pts[i+1]))
    lines.append(gmsh.model.geo.addLine(top_pts[-1], p1))
    
    surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])

    if prof_muro > 0:
        p_inf = gmsh.model.geo.addPoint(x_muro_absoluto, 30.0 - prof_muro, 0, d_critica)
        l_muro = gmsh.model.geo.addLine(top_pts[top_xs.index(x_muro_absoluto)], p_inf)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [l_muro], 2, surf)

    gmsh.model.geo.synchronize()
    
    # Refinamiento
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [top_pts[top_xs.index(x_inicio)], top_pts[top_xs.index(x_fin)], p_inf if prof_muro > 0 else top_pts[0]])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", d_critica); gmsh.model.mesh.field.setNumber(2, "SizeMax", d_global)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0); gmsh.model.mesh.field.setNumber(2, "DistMax", 15.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    gmsh.option.setNumber("Mesh.RecombineAll", 1); gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(2)

    # Exportar a Scikit-fem
    nodes = gmsh.model.mesh.getNodes()[1].reshape((-1, 3))[:, :2].T
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
    
    if 3 in elemTypes: # Quads
        m = MeshQuad(nodes, elemNodeTags[np.where(elemTypes==3)[0][0]].reshape((-1, 4)).T - 1)
        basis = Basis(m, ElementQuad1())
    else: # Tris
        m = MeshTri(nodes, elemNodeTags[np.where(elemTypes==2)[0][0]].reshape((-1, 3)).T - 1)
        basis = Basis(m, ElementTriP1())

    # Solución
    K = asm(laplace, basis) * k_suelo
    dofs_h1 = basis.get_dofs(lambda x: (x[0] <= x_inicio + 0.1) & (x[1] >= 29.9)).all()
    dofs_h2 = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.1) & (x[1] >= 29.9)).all()
    h = np.zeros(basis.N); h[dofs_h1], h[dofs_h2] = h1, h2
    h_sol = solve(*condense(K, np.zeros(basis.N), h, D=np.union1d(dofs_h1, dofs_h2)))

    # Post-proceso: Gradientes (Calculados por elemento y promediados a nodos)
    # i = -grad(h)
    grad_h = basis.project(basis.interpolate(h_sol).grad)
    ix, iy = -grad_h[0], -grad_h[1]
    imag = np.sqrt(ix**2 + iy**2)

    # Caudal Q (Integral de flujo en la entrada x=0)
    Q = np.sum(k_suelo * ix[dofs_h1]) * (30.0 / len(dofs_h1)) # Aproximación por área tributaria

    # FS Sifonamiento
    i_exit_max = np.max(iy[dofs_h2]) if len(dofs_h2) > 0 else 0.01
    fs = ic_critico / i_exit_max

    return {
        "mesh": m, "h": h_sol, "ix": ix, "iy": iy, "imag": imag,
        "Q": Q, "fs": fs, "ic": ic_critico, "i_exit": i_exit_max,
        "params": (Lx, x_inicio, x_fin, x_muro_absoluto, prof_muro)
    }
