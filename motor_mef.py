import numpy as np
from skfem import MeshQuad, Basis, ElementQuad1, asm, condense, solve
from skfem.models.poisson import laplace

def resolver_mef_presa(Lx, prof_muro, pos_muro, h1, h2, k_suelo, gs, e_vacios):
    
    x_inicio = (Lx - 15.0) / 2.0
    x_fin = x_inicio + 15.0
    x_muro = x_inicio + pos_muro
    ic_critico = (gs - 1) / (1 + e_vacios)

    # 1. GENERACIÓN DE GRILLA MATEMÁTICA PURA (100% Cuadriláteros)
    nx, ny = 160, 60  # Alta resolución de la cuadrícula
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, 30, ny)
    m = MeshQuad.init_tensor(x_coords, y_coords)

    # 2. EXCAVACIÓN BOOLEANA (Recorte de la estructura)
    # Calculamos el centroide de cada elemento cuadriculado
    cx = m.p[0, m.t].mean(axis=0)
    cy = m.p[1, m.t].mean(axis=0)
    
    # Identificamos los elementos que caen dentro del empotramiento de la presa (y=25 a y=30)
    excavacion_presa = (cx >= x_inicio) & (cx <= x_fin) & (cy >= 25.0)
    
    # Identificamos los elementos de la tablestaca (grosor de 0.4m para exactitud nodal)
    excavacion_muro = (cx >= x_muro - 0.2) & (cx <= x_muro + 0.2) & (cy >= 25.0 - prof_muro)
    
    # Conservamos únicamente los elementos que son "Suelo"
    elementos_suelo = ~(excavacion_presa | excavacion_muro)
    m = MeshQuad(m.p, m.t[:, elementos_suelo])
    m = m.remove_orphans()  # Limpia los nodos que quedaron flotando
    
    basis = Basis(m, ElementQuad1())

    # 3. ENSAMBLAJE Y SOLUCIÓN
    K = asm(laplace, basis) * k_suelo
    
    # Nodos de carga aguas arriba (izquierda) y aguas abajo (derecha) en la superficie
    dofs_h1 = basis.get_dofs(lambda x: (x[0] <= x_inicio + 0.1) & (x[1] >= 29.9)).all()
    dofs_h2 = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.1) & (x[1] >= 29.9)).all()
    
    h = np.zeros(basis.N)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    frontera = np.union1d(dofs_h1, dofs_h2)
    h_sol = solve(*condense(K, np.zeros(basis.N), h, D=frontera))

    # 4. POST-PROCESO VECTORIAL
    grad_eval = basis.interpolate(h_sol).grad
    ix = -basis.project(grad_eval[0])
    iy = -basis.project(grad_eval[1])
    imag = np.sqrt(ix**2 + iy**2)

    # Caudal Exacto (Por matriz de fuerzas)
    flujo_nodal = K @ h_sol
    Q = np.sum(np.abs(flujo_nodal[dofs_h1]))

    # Gradiente de Salida (Monitoreo en el talón de descarga)
    dofs_salida = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.1) & (x[0] <= x_fin + 5.0) & (x[1] >= 24.9)).all()
    i_exit_max = np.max(iy[dofs_salida]) if len(dofs_salida) > 0 else 0.01

    return {
        "mesh": m, "h": h_sol, "ix": ix, "iy": iy, "imag": imag,
        "Q": Q, "fs": ic_critico / i_exit_max, "ic": ic_critico, "i_exit": i_exit_max,
        "params": (Lx, x_inicio, x_fin, x_muro, prof_muro)
    }
