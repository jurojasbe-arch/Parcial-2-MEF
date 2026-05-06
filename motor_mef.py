import numpy as np
from skfem import MeshQuad, Basis, ElementQuad1, asm, condense, solve
from skfem.models.poisson import laplace

def resolver_mef_presa(Lx, prof_muro, pos_muro, h1, h2, k_suelo, gs, e_vacios):
    
    x_inicio = (Lx - 15.0) / 2.0
    x_fin = x_inicio + 15.0
    x_muro = x_inicio + pos_muro
    grosor_muro = 0.4 # Grosor físico real
    ic_critico = (gs - 1) / (1 + e_vacios)

    # 1. GENERACIÓN DE GRILLA MATEMÁTICA ESTRICTA
    # Forzamos los nodos para que coincidan milimétricamente con la estructura
    x_m_izq = x_muro - grosor_muro/2
    x_m_der = x_muro + grosor_muro/2
    
    # Ensamblamos las coordenadas X asegurando las posiciones críticas
    xs = [0.0, x_inicio, x_m_izq, x_m_der, x_fin, Lx]
    x_coords = []
    for i in range(len(xs)-1):
        if xs[i] < xs[i+1]:
            num_pts = int(max(10, 50 * abs(xs[i+1]-xs[i])/Lx))
            if i in [1, 2, 3]: num_pts = max(num_pts, 15) # Mayor densidad bajo la presa
            x_coords.extend(np.linspace(xs[i], xs[i+1], num_pts)[:-1])
    x_coords.append(Lx)
    x_coords = np.unique(np.sort(x_coords))

    # Ensamblamos las coordenadas Y asegurando la profundidad de la punta
    y_punta = 25.0 - prof_muro
    ys = [0.0, y_punta, 25.0, 30.0]
    y_coords = []
    for i in range(len(ys)-1):
        if ys[i] < ys[i+1]:
            num_pts = int(max(10, 40 * abs(ys[i+1]-ys[i])/30.0))
            y_coords.extend(np.linspace(ys[i], ys[i+1], num_pts)[:-1])
    y_coords.append(30.0)
    y_coords = np.unique(np.sort(y_coords))

    m = MeshQuad.init_tensor(x_coords, y_coords)

    # 2. EXCAVACIÓN BOOLEANA (Ahora sí es infalible)
    cx = m.p[0, m.t].mean(axis=0)
    cy = m.p[1, m.t].mean(axis=0)
    
    excavacion_presa = (cx >= x_inicio) & (cx <= x_fin) & (cy >= 25.0)
    excavacion_muro = (cx >= x_m_izq) & (cx <= x_m_der) & (cy >= y_punta)
    
    elementos_suelo = ~(excavacion_presa | excavacion_muro)
    t_suelo = m.t[:, elementos_suelo]
    
    # Limpieza manual de nodos huérfanos
    nodos_activos = np.unique(t_suelo)
    mapa_nodos = np.zeros(m.p.shape[1], dtype=np.int64) - 1
    mapa_nodos[nodos_activos] = np.arange(len(nodos_activos))
    
    p_nuevo = m.p[:, nodos_activos]
    t_nuevo = mapa_nodos[t_suelo]
    
    m = MeshQuad(p_nuevo, t_nuevo)
    basis = Basis(m, ElementQuad1())

    # 3. ENSAMBLAJE Y SOLUCIÓN
    K = asm(laplace, basis) * k_suelo
    
    dofs_h1 = basis.get_dofs(lambda x: (x[0] <= x_inicio + 0.01) & (x[1] >= 29.99)).all()
    dofs_h2 = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.01) & (x[1] >= 29.99)).all()
    
    h = np.zeros(basis.N)
    h[dofs_h1] = h1
    h[dofs_h2] = h2
    
    frontera = np.union1d(dofs_h1, dofs_h2)
    h_sol = solve(*condense(K, np.zeros(basis.N), h, D=frontera))

    # 4. POST-PROCESO VECTORIAL EXACTO
    grad_eval = basis.interpolate(h_sol).grad
    ix = -basis.project(grad_eval[0])
    iy = -basis.project(grad_eval[1])
    imag = np.sqrt(ix**2 + iy**2)

    # Caudal
    flujo_nodal = K @ h_sol
    Q = np.sum(np.abs(flujo_nodal[dofs_h1]))

    # Gradiente de Salida Real
    dofs_salida = basis.get_dofs(lambda x: (x[0] >= x_fin - 0.01) & (x[0] <= x_fin + 5.0) & (x[1] >= 24.99)).all()
    i_exit_max = np.max(iy[dofs_salida]) if len(dofs_salida) > 0 else 0.01

    return {
        "mesh": m, "h": h_sol, "ix": ix, "iy": iy, "imag": imag,
        "Q": Q, "fs": ic_critico / i_exit_max, "ic": ic_critico, "i_exit": i_exit_max,
        "params": (Lx, x_inicio, x_fin, x_muro, prof_muro, grosor_muro)
    }
