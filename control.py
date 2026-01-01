import json
import jax
import jax.numpy as jnp
import equinox as eqx
import scipy

def build_rfftn_modes_single(Nt, Nx, *, n, m, A=1.0, phi_t=0.0, phi_x=0.0, dtype=jnp.complex64):
    """
    Returns modes of shape (Nt//2+1, Nx//2+1) compatible with u = irfftn_time_space(modes, Nt, Nx).
    Here rfftn means: FFT along time axis (full, complex), rFFT along space axis (one-sided).
    """
    assert 0 <= n < Nt
    assert 0 <= m <= Nx // 2

    modes = jnp.zeros((Nt//2+1, Nx // 2 + 1), dtype=dtype)

    # Put phase into one complex number; time and space phases just add for a separable cosine product
    phi = phi_t + phi_x

    # For a real cosine, the spectrum has conjugate symmetry in the *time* axis:
    # modes[Nt-n, m] = conj(modes[n, m])
    c = (A / 2.0) * jnp.exp(1j * phi)

    modes = modes.at[n, m].add(c)

    return modes

def enforce_time_hermitian(w, Nt):
    """
    w: (Nt//2+1, Nx_rfft) complex -- independent time frequencies.
    returns modes: (Nt, Nx_rfft) complex satisfying modes[Nt-n] = conj(modes[n])
    """
    Nt_pos = Nt//2 + 1
    assert w.shape[0] == Nt_pos

    # Start with zeros
    modes = jnp.zeros((Nt, w.shape[1]), dtype=w.dtype)

    # DC must be real
    modes = modes.at[0].set(jnp.real(w[0]))

    # Positive frequencies
    modes = modes.at[1:Nt_pos].set(w[1:Nt_pos])

    # Negative frequencies via conjugate symmetry
    # These correspond to n = Nt_pos .. Nt-1  <-> 1..Nt_pos-2
    if Nt_pos > 2:
        modes = modes.at[Nt_pos:Nt].set(jnp.conj(w[1:Nt_pos-1][::-1]))

    # Nyquist must be real when Nt even
    if Nt % 2 == 0:
        modes = modes.at[Nt//2].set(jnp.real(w[Nt//2]))

    return modes

def irfftn_time_space(modes, Nt, Nx):
    # Invert space (irfft) then invert time (ifft)
    modes = enforce_time_hermitian(modes, Nt) 
    u_tfreq_x = jnp.fft.irfft(modes, n=Nx, axis=1)      # (Nt, Nx) still in time-frequency domain
    u_tx = jnp.fft.ifft(u_tfreq_x, axis=0).real         # (Nt, Nx) real-valued
    return u_tx

class FourierActuator(eqx.Module):
    modes: jax.Array
    K0: jax.Array = eqx.field(static=True)
    u_max: jax.Array = eqx.field(static=True)
    zero: bool = eqx.field(static=True)
    Nt: int = eqx.field(static=True)
    N_mesh: int = eqx.field(static=True)
    closed_loop: bool = eqx.field(static=True)

    def __init__(self, Nt, N_mesh, modes=None, u_max=None, K0=None, zero=False, closed_loop=False):
        self.zero = zero
        self.modes = modes # (Nt//2+1, N_mesh//2+1) complex
        self.Nt = Nt
        self.N_mesh = N_mesh

        # For closed loop
        self.closed_loop = closed_loop
        self.K0 = K0
        self.u_max = u_max

    def __call__(self, n, x=None):
        if not self.zero:
            if self.closed_loop: # x must not be None
                # Implement using ROM!
                x_rom = x # Replace with function mapping to state controller was trained on
                u = -(self.K0 @ x_rom)
                u = self.u_max * jnp.tanh(u / self.u_max)
            else: # No signal in, just time
                u_all = irfftn_time_space(self.modes, self.Nt, self.N_mesh)
                u = u_all[n.astype(int)]
            return u
        else:
            return jnp.zeros(self.N_mesh)
    
    def save_model(self, filename):
        with open(filename, "wb") as f:
            hyperparams = {
                "zero": self.zero,
                "Nt": self.Nt,
                "N_mesh": self.N_mesh,
                "closed_loop": self.closed_loop,
            }
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
    
    @classmethod
    def load_model(cls, filename):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            zero = hyperparams["zero"]
            modes = hyperparams["modes"]
            Nt = hyperparams["Nt"]
            N_mesh = hyperparams["N_mesh"]
            closed_loop = hyperparams["closed_loop"]
            K0 = hyperparams["K0"]
            u_max = hyperparams["u_max"]

            # build skeleton with identical hyperparams
            model = cls(
                zero=zero,
                modes=modes,
                Nt=Nt,
                N_mesh=N_mesh,
                closed_loop=closed_loop,
                K0=K0,
                u_max=u_max
            )

            # load parameters into that structure
            return eqx.tree_deserialise_leaves(f, model)

def ctrb(A, B):
    n = A.shape[0]
    blocks = []
    AB = B
    for _ in range(n):
        blocks.append(AB)
        AB = A @ AB
    return jnp.concatenate(blocks, axis=1)

def continuous_lqr(A, B, Q=None, R=None):
    """
    Continuous-time LQR for xdot = A x + B u.
    Returns K, P, eigvals(A-BK)
    """
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)
    n = A_np.shape[0]
    m = B_np.shape[1]

    if Q is None:
        Q_np = jnp.eye(n)
    else:
        Q_np = jnp.asarray(Q)

    if R is None:
        R_np = jnp.eye(m)
    else:
        R_np = jnp.asarray(R)

    # Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
    P = scipy.linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)

    # K = R^{-1} B^T P
    K = jnp.linalg.solve(R_np, B_np.T @ P)

    eig_cl = jnp.linalg.eigvals(A_np - B_np @ K)
    return jnp.asarray(K), jnp.asarray(P), jnp.asarray(eig_cl)

def discrete_lqr(A, B, Q=None, R=None):
    """
    Discrete-time LQR for x_{k+1} = A x_k + B u_k.
    Minimizes sum_{k=0}^\infty (x_k^T Q x_k + u_k^T R u_k).
    Returns K, P, eigvals(A - B K)
    """
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)
    n = A_np.shape[0]
    m = B_np.shape[1]

    if Q is None:
        Q_np = jnp.eye(n)
    else:
        Q_np = jnp.asarray(Q)

    if R is None:
        R_np = jnp.eye(m)
    else:
        R_np = jnp.asarray(R)

    # Solve DARE: P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
    # scipy returns a NumPy array; we'll wrap back to jnp
    P = scipy.linalg.solve_discrete_are(
        jnp.asarray(A_np), jnp.asarray(B_np), jnp.asarray(Q_np), jnp.asarray(R_np)
    )

    P = jnp.asarray(P)

    # K = (R + B^T P B)^{-1} (B^T P A)
    S = R_np + B_np.T @ P @ B_np
    K = jnp.linalg.solve(S, B_np.T @ P @ A_np)

    eig_cl = jnp.linalg.eigvals(A_np - B_np @ K)
    return jnp.asarray(K), jnp.asarray(P), jnp.asarray(eig_cl)