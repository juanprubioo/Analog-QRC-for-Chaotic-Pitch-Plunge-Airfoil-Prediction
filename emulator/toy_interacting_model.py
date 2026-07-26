"""toy_interacting_model.py — Apéndice: dinámica unitaria exacta de la cadena de 8 átomos.

H = sum_i [ (Omega/2) sigma_x^i - Delta_i n_i ] + sum_{i<j} (C6 / r_ij^6) n_i n_j
Delta_i = ENC/2 - ENC * x_i  (encoding local del pipeline; ENC = 9 -> Delta in [-4.5, +4.5])
psi(0) = |00...0>, quench de Omega constante (como build_local_task, sin rampas).

Para N muestras x ~ U[0,1]^8 se computa <n_i(t)> exacto (diagonalización, dim 256) y:
  corr_x(t)  = media_i |corr(<n_i(t)>, x_i)|        -> canal de SIGNO
  corr_ax(t) = media_i |corr(<n_i(t)>, |x_i-0.5|)|  -> canal de PARIDAD
con V = C6/a^6 (a = 10 um) y con V = 0 (control no interactuante).

Predicciones a contrastar con lo medido en el emulador (66 ventanas reales, 500 shots):
  - V=5.42: minimo de corr_x cerca de t ~ 0.70 us (nodo medido).
  - V=0:    corr_x ~ 0 en todo t (degeneracion de paridad) y corr_ax alto,
            con nodo de corr_ax cerca de 0.91 us (modelo de 1 atomo).
"""
import numpy as np

OMEGA = 6.283; ENC = 9.0
C6 = 5.42e6           # rad/us * um^6
A_UM = 10.0           # espaciado
NATOMS = 8
NSAMPLES = 200
TS = np.arange(0.05, 3.001, 0.025)
RNG = np.random.default_rng(7)

def build_static_ops(n):
    dim = 2**n
    # n_i diagonales
    n_ops = np.zeros((n, dim))
    for i in range(n):
        for b in range(dim):
            if (b >> i) & 1: n_ops[i, b] = 1.0
    # sum sigma_x (matriz dispersa como pares de indices)
    rows, cols = [], []
    for b in range(dim):
        for i in range(n):
            rows.append(b); cols.append(b ^ (1 << i))
    return n_ops, (np.array(rows), np.array(cols))

def hamiltonian(x, V, n_ops, sx_idx):
    n, dim = n_ops.shape
    H = np.zeros((dim, dim))
    H[sx_idx] += OMEGA / 2.0
    Delta = ENC/2.0 - ENC * x
    diag = -(Delta[:, None] * n_ops).sum(0)
    if V > 0:
        for i in range(n):
            for j in range(i+1, n):
                diag += (V / abs(i-j)**6) * n_ops[i] * n_ops[j]
    H[np.arange(dim), np.arange(dim)] += diag
    return H

def run(V):
    n_ops, sx_idx = build_static_ops(NATOMS)
    dim = 2**NATOMS
    X = RNG.uniform(0, 1, (NSAMPLES, NATOMS))
    N_t = np.empty((NSAMPLES, len(TS), NATOMS))
    psi0 = np.zeros(dim); psi0[0] = 1.0
    for k in range(NSAMPLES):
        H = hamiltonian(X[k], V, n_ops, sx_idx)
        w, U = np.linalg.eigh(H)
        c0 = U.T @ psi0
        for it, t in enumerate(TS):
            psi = U @ (np.exp(-1j * w * t) * c0)
            p = np.abs(psi)**2
            N_t[k, it] = n_ops @ p
    corr_x = np.empty(len(TS)); corr_ax = np.empty(len(TS))
    for it in range(len(TS)):
        cx, cax = [], []
        for i in range(NATOMS):
            cx.append(abs(np.corrcoef(N_t[:, it, i], X[:, i])[0, 1]))
            cax.append(abs(np.corrcoef(N_t[:, it, i], np.abs(X[:, i]-0.5))[0, 1]))
        corr_x[it] = np.mean(cx); corr_ax[it] = np.mean(cax)
    return corr_x, corr_ax

if __name__ == "__main__":
    V = C6 / A_UM**6
    print(f"V (a={A_UM} um) = {V:.3f} rad/us | {NSAMPLES} muestras | dim = {2**NATOMS}")
    cx_int, cax_int = run(V)
    cx_0, cax_0 = run(0.0)

    def local_minima(y, thresh):
        return [i for i in range(2, len(TS)-2)
                if y[i] < y[i-1] and y[i] < y[i+1] and y[i] < thresh]

    print("\nV = %.2f (interactuante):" % V)
    for i in local_minima(cx_int, 0.5)[:6]:
        print(f"  minimo de corr_x en t = {TS[i]:.3f} us (corr = {cx_int[i]:.3f})")
    print("V = 0 (control):")
    print(f"  corr_x media sobre t: {cx_0.mean():.3f} (esperado ~0: degeneracion de paridad)")
    for i in local_minima(cax_0, 0.35)[:6]:
        print(f"  minimo de corr_ax en t = {TS[i]:.3f} us (corr = {cax_0[i]:.3f})")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(TS, cx_int, color="#2a78d6", lw=1.8, label=f"V={V:.1f}: |corr(⟨n⟩, x)| (signo)")
    ax.plot(TS, cx_0, color="#2a78d6", lw=1.2, ls=":", label="V=0: |corr(⟨n⟩, x)| — colapsa (paridad)")
    ax.plot(TS, cax_0, color="#d64a2a", lw=1.4, ls="--", label="V=0: |corr(⟨n⟩, |x−0.5|)| (paridad)")
    ax.axvspan(0.65, 0.80, color="#555", alpha=0.12, label="nodo medido en emulador (~0.70 µs)")
    ax.set_xlabel("t [µs]"); ax.set_ylabel("correlación media por átomo")
    ax.set_title("Modelo exacto de la cadena (quench, dim 256):\ncanales de signo y paridad vs tiempo")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("toy_interacting_model.png", dpi=160)
    print("\nfigura: toy_interacting_model.png")
