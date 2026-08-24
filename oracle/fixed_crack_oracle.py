"""A forward model built to disagree with the identification, on purpose.

WHY this file exists. Every accuracy number in this study is obtained by
identifying fields that the study's own reference solver produced, with the
same constitutive map, the same element and the same equilibrium path used
on the other side of the identification. That is an inverse crime, and
perturbing a material constant inside the shared solver does not undo it:
the element, the cracking formulation and the path are still common to
generator and identifier, so a material perturbation prices only the part
of the error that the constitutive constants carry.

This module is the other half of the test. It is a smeared FIXED-crack
concrete model on four-node quadrilaterals, and it disagrees with the
Compatible Stress Field Method in the places that matter most:

  crack orientation   the crack normal is frozen at the principal direction
                      holding when the Rankine criterion is first met, and
                      never rotates again. The CSFM rotates its principal
                      frame with the current strain at every point and every
                      load level, so past first cracking the two models put
                      their stress on different axes.
  concrete tension    a real tensile strength f_ct with exponential
                      softening, regularised on the fracture energy over a
                      crack band, against the CSFM's outright neglect of
                      concrete tension.
  compression         Hognestad parabola with a genuine descending branch
                      and a residual plateau, against the CSFM's
                      parabola-rectangle that never unloads.
  compression softening
                      Vecchio and Collins 1993 Model B, a function of the
                      RATIO of the two principal strains, against the CSFM's
                      k_c2 = 1 / (0.8 + 140 eps_1), a function of the
                      tensile strain alone.
  shear on the crack  an explicit retention factor multiplying the elastic
                      shear modulus, a quantity the rotating formulation
                      does not possess because its principal frame carries
                      no shear by construction.
  element             four-node quadrilateral at 2 by 2 Gauss on an 80 by 40
                      grid, against constant-strain triangles on 40 by 20.

What is deliberately NOT changed: the material constants. f_c, E_c,
eps_c2, f_y, f_t, E_s and eps_u are the reference values, and the steel is
the same bilinear bare-bar law, because the cost of moving those constants
is already priced separately and mixing the two would make neither
readable. What this model prices is the form of the model, not its
numbers.

The tangent is formed by finite differences on the three-component
constitutive response, so no analytic linearisation is shared with the
reference either.

Run:  python fixed_crack_oracle.py     (verification, then one curve)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

# --------------------------------------------------------------------------- #
# Material constants
# --------------------------------------------------------------------------- #
# Shared verbatim with the reference solver, so that the comparison isolates
# the form of the model rather than the values in it.
FC = 30.0                      # cylinder strength (MPa)
EC = 4700.0 * FC ** 0.5        # 4700 lam sqrt(fc), as arclength_oracle.Material
EPS_C0 = 0.0020                # strain at peak compression
FY, FTS, ES, EPS_US = 500.0, 550.0, 200_000.0, 0.05      # steel, bilinear

# Required by the alternative form and having no counterpart in the CSFM.
FCT = 0.33 * FC ** 0.5         # tensile strength (MPa), ~1.81 at fc = 30
GF = 0.13                      # fracture energy (N/mm), fib MC2010 order
GC = 250.0 * GF                # compressive fracture energy (N/mm). The value
                               # is the Gc = 250 Gf of the Dutch RTD 1016
                               # guidelines for nonlinear FE analysis of
                               # concrete structures (Hendriks, de Boer,
                               # Belletti), i.e. 32.5 N/mm at Gf = 0.13; the
                               # alternative estimate of Nakamura and Higai
                               # (2001), Gc = 8.8 sqrt(fc) = 48 N/mm at
                               # fc = 30, is the same order. Not tuned.
RESID_C = 0.20                 # residual compressive plateau, fraction of fcp
NU = 0.2                       # Poisson ratio, used only for the shear modulus
G0 = EC / (2.0 * (1.0 + NU))
EPS_BETA = 4.0e-4              # crack strain at which shear retention halves
BETA_MIN = 0.01                # floor on the retention factor
EPS_CR = FCT / EC              # cracking strain
EPS_RAMP = 5.0 * EPS_CR        # crack strain at which compression softening is full
BETA_C_MIN = 0.15              # floor on the compression-softening factor

_TINY = 1e-12


# --------------------------------------------------------------------------- #
# Uniaxial laws
# --------------------------------------------------------------------------- #
def steel_stress(eps: np.ndarray) -> np.ndarray:
    """Bilinear bare-bar steel, identical to the reference law.

    Kept identical on purpose: the parameter being identified is read
    through this law, so changing it would price steel-model error rather
    than concrete-model error.
    """
    s = np.abs(eps)
    ey = FY / ES
    esh = (FTS - FY) / (EPS_US - ey)
    harden = np.minimum(FY + esh * (s - ey), FTS)
    return np.sign(eps) * np.where(s <= ey, ES * s, harden)


def comp_hognestad(e_mag: np.ndarray, fcp: np.ndarray,
                   gc_h: float) -> np.ndarray:
    """Compressive magnitude (MPa) on the Hognestad parabola plus descent.

    The descending branch is the point of the choice: the CSFM's
    parabola-rectangle holds its peak for ever, so the two laws separate
    exactly where a deep-beam strut lives, above eps_c2.

    The descent is crack-band regularised, exactly as the tensile branch
    already is. The first version was not: it ran from the peak at eps_c0
    to a fixed ultimate strain of 0.0035, so the energy dissipated by a
    softening band scaled with the element size and the model's capacity
    moved 25 per cent between a 40x20 and an 80x40 grid. Here the linear
    descent from fcp to the residual plateau spans a strain

        de = 2 Gc / (h fcp (1 - RESID_C)),

    which makes the triangle between the descending line and the residual,
    0.5 (1 - RESID_C) fcp de, dissipate Gc / h per unit volume, i.e. Gc per
    unit area of a band one element wide, independent of h. Gc is the
    compressive fracture energy declared with the material constants; the
    residual plateau itself is not regularisable and is left alone. fcp is
    the current softened peak, so the band's energy follows the softened
    strength, which is the standard crack-band treatment.
    """
    e = np.maximum(e_mag, 0.0)
    r = e / EPS_C0
    asc = fcp * (2.0 * r - r * r)
    de = 2.0 * gc_h / (np.maximum(fcp, _TINY) * (1.0 - RESID_C))
    desc = fcp * (1.0 - (1.0 - RESID_C) * (e - EPS_C0) / de)
    return np.where(e <= EPS_C0, asc, np.maximum(desc, RESID_C * fcp))


def beta_vc93b(e_ten: np.ndarray, e_comp: np.ndarray) -> np.ndarray:
    """Compression softening, Vecchio and Collins 1993 Model B.

    beta = 1 / (1 + K_c) with K_c = 0.35 (-eps_1/eps_2 - 0.28)^0.8. The
    argument is the RATIO of the principal strains, where the CSFM's k_c2
    depends on the tensile strain alone, so the two disagree in what drives
    the strut down as well as by how much.

    Two guards, both discovered by measuring rather than by anticipating.
    The published form is stated for CRACKED concrete, and a function of a
    ratio does not know how large the strains are: applied literally it cuts
    the strength by a fifth at a transverse strain of 1e-6, and the
    displacement-controlled Newton then failed to converge on a problem in
    which not one point had cracked. The factor is therefore ramped in over
    the crack strain, which is what the published derivation intends. The
    floor keeps the strut from vanishing where a heavily cracked point is
    only lightly compressed, a state in which the ratio runs away and the
    stress and the stiffness both go to zero.
    """
    ratio = np.maximum(e_ten, 0.0) / np.maximum(-e_comp, _TINY)
    kc = 0.35 * np.maximum(ratio - 0.28, 0.0) ** 0.8
    beta = np.maximum(1.0 / (1.0 + kc), BETA_C_MIN)
    ramp = np.clip((e_ten - EPS_CR) / (EPS_RAMP - EPS_CR), 0.0, 1.0)
    return 1.0 + (beta - 1.0) * ramp


def tension_soft(e: np.ndarray, eps_soft: float) -> np.ndarray:
    """Exponential tension softening, crack-band regularised.

    Linear to f_ct at the cracking strain, then f_ct exp(-(e - e_cr)/eps_soft)
    with eps_soft = G_f / (h f_ct), so the dissipated energy per unit crack
    area is h-independent. The CSFM carries no concrete tension at all.
    """
    # the exponent is clamped because both branches are evaluated before the
    # selection, and a compressive strain would otherwise overflow it
    return np.where(e <= EPS_CR, EC * e,
                    FCT * np.exp(-np.maximum(e - EPS_CR, 0.0) / eps_soft))


def uniaxial(e: np.ndarray, e_lat: np.ndarray, eps_soft: float,
             gc_h: float) -> np.ndarray:
    """Concrete stress along one axis of the crack frame."""
    fcp = FC * beta_vc93b(e_lat, e)
    return np.where(e < 0.0, -comp_hognestad(-e, fcp, gc_h),
                    tension_soft(e, eps_soft))


# --------------------------------------------------------------------------- #
# Two-dimensional constitutive with a frozen crack frame
# --------------------------------------------------------------------------- #
def crack_frame(ex, ey, gxy, cracked, th_cr):
    """The axes the stress is evaluated on, and which points have cracked.

    A point that has already cracked keeps the angle stored for it. A point
    that has not uses the current principal direction, and is flagged as
    newly cracked when the major principal strain passes the cracking
    strain, which for this tension law is the Rankine criterion written in
    strain. Freezing happens when the caller commits a converged step, so
    the direction a new crack takes is the one holding at equilibrium.
    """
    eav = 0.5 * (ex + ey)
    rad = np.hypot(0.5 * (ex - ey), 0.5 * gxy)
    e1 = eav + rad
    th_p = 0.5 * np.arctan2(gxy, ex - ey)
    th = np.where(cracked, th_cr, th_p)
    return th, (cracked | (e1 >= EPS_CR))


def concrete_stress(ex, ey, gxy, th, is_cracked, eps_soft, gc_h):
    """Plane-stress concrete response on the given (fixed) axes."""
    c, s = np.cos(th), np.sin(th)
    c2, s2, cs = c * c, s * s, c * s
    e_n = ex * c2 + ey * s2 + gxy * cs
    e_t = ex * s2 + ey * c2 - gxy * cs
    g_nt = 2.0 * (ey - ex) * cs + gxy * (c2 - s2)

    s_n = uniaxial(e_n, e_t, eps_soft, gc_h)
    s_t = uniaxial(e_t, e_n, eps_soft, gc_h)
    beta = np.maximum(BETA_MIN,
                      1.0 / (1.0 + np.maximum(e_n, 0.0) / EPS_BETA))
    t_nt = np.where(is_cracked, beta * G0 * g_nt, 0.0)

    sx = s_n * c2 + s_t * s2 - 2.0 * t_nt * cs
    sy = s_n * s2 + s_t * c2 + 2.0 * t_nt * cs
    txy = (s_n - s_t) * cs + t_nt * (c2 - s2)
    return sx, sy, txy


def total_stress(eps, th, is_cracked, rho_x, rho_y, eps_soft, gc_h):
    """Concrete plus smeared reinforcement, as an (n, 3) stress array."""
    ex, ey, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    sx, sy, txy = concrete_stress(ex, ey, gxy, th, is_cracked, eps_soft, gc_h)
    return np.stack([sx + rho_x * steel_stress(ex),
                     sy + rho_y * steel_stress(ey), txy], axis=1)


# --------------------------------------------------------------------------- #
# Q4 mesh on a uniform rectangle
# --------------------------------------------------------------------------- #
GP = 1.0 / np.sqrt(3.0)
GAUSS = [(-GP, -GP), (GP, -GP), (GP, GP), (-GP, GP)]


@dataclass
class Q4Grid:
    nx: int
    ny: int
    dx: float
    dy: float
    xy: np.ndarray
    dofs: np.ndarray            # (ne, 8)
    Be: np.ndarray              # (4, 3, 8), one per Gauss point, all elements
    wgt: float
    rho_x: np.ndarray           # (ne,)
    rho_y: np.ndarray           # (ne,)
    fixed: np.ndarray
    load_nodes: np.ndarray
    supp_left: np.ndarray
    supp_right: np.ndarray
    ndof: int
    h_band: float


def _b_matrices(dx: float, dy: float) -> tuple[np.ndarray, float]:
    xe = np.array([[0.0, 0.0], [dx, 0.0], [dx, dy], [0.0, dy]])
    out = []
    detJ = None
    for xi, eta in GAUSS:
        dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                              [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
        J = dN @ xe
        detJ = float(np.linalg.det(J))
        dNxy = np.linalg.solve(J, dN)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNxy[0]
        B[1, 1::2] = dNxy[1]
        B[2, 0::2] = dNxy[1]
        B[2, 1::2] = dNxy[0]
        out.append(B)
    return np.array(out), detJ


def build_grid(rho_tie: float, L=2000.0, H=1000.0, nx=80, ny=40,
               a=250.0, bearing=200.0, band=150.0,
               rho_min=0.0010, rho_stirrup=0.0015) -> Q4Grid:
    """Uniform Q4 grid with the reference geometry and reinforcement layout."""
    dx, dy = L / nx, H / ny
    nnx = nx + 1
    xs = np.linspace(0.0, L, nnx)
    ys = np.linspace(0.0, H, ny + 1)
    xy = np.array([[x, y] for y in ys for x in xs])
    nid = lambda i, j: j * nnx + i                                   # noqa: E731

    quads, rx, ry = [], [], []
    for j in range(ny):
        for i in range(nx):
            quads.append([nid(i, j), nid(i + 1, j),
                          nid(i + 1, j + 1), nid(i, j + 1)])
            yc = (j + 0.5) * dy
            rx.append(rho_tie if yc < band else rho_min)
            ry.append(rho_min + rho_stirrup)
    quads = np.array(quads)
    dofs = np.empty((len(quads), 8), dtype=int)
    dofs[:, 0::2] = 2 * quads
    dofs[:, 1::2] = 2 * quads + 1

    Be, detJ = _b_matrices(dx, dy)
    half = bearing / 2.0
    ndof = 2 * len(xy)
    fixed = np.zeros(ndof, bool)
    left, right = [], []
    for n in range(len(xy)):
        x, y = xy[n]
        if y < 1e-9 and abs(x - a) <= half + 1e-9:
            fixed[2 * n] = fixed[2 * n + 1] = True
            left.append(n)
        if y < 1e-9 and abs(x - (L - a)) <= half + 1e-9:
            fixed[2 * n + 1] = True
            right.append(n)
    load = np.array([n for n in range(len(xy))
                     if abs(xy[n, 1] - H) < 1e-9
                     and abs(xy[n, 0] - L / 2.0) <= half + 1e-9])
    return Q4Grid(nx=nx, ny=ny, dx=dx, dy=dy, xy=xy, dofs=dofs, Be=Be,
                  wgt=detJ, rho_x=np.array(rx), rho_y=np.array(ry),
                  fixed=fixed, load_nodes=load,
                  supp_left=np.array(left), supp_right=np.array(right),
                  ndof=ndof, h_band=float(np.sqrt(dx * dy)))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _response(eps, cracked, th_cr, rho_x, rho_y, eps_soft, gc_h):
    """Stress at a strain, with the crack frame resolved from the committed
    state. Points that have already cracked keep their frozen normal; points
    that have not are evaluated on the principal axes of the strain handed
    in."""
    th, is_cr = crack_frame(eps[:, 0], eps[:, 1], eps[:, 2], cracked, th_cr)
    return (total_stress(eps, th, is_cr, rho_x, rho_y, eps_soft, gc_h),
            th, is_cr)


def assemble(u, g: Q4Grid, state, thickness=300.0, tangent=True, fd=1e-8):
    """Internal force, and optionally a finite-difference consistent tangent.

    The tangent is differenced rather than derived so that no analytic
    linearisation is inherited from the reference implementation.

    The perturbation resolves the crack frame afresh instead of holding the
    base-point frame fixed. Holding it fixed was the first version and it is
    wrong: for a point that has not yet cracked the frame IS the principal
    direction of the strain, so it turns when the strain is perturbed, and a
    frozen-frame difference misses that rotation entirely. Measured against
    a difference of the assembled residual the frozen version was 38 per
    cent out, which was enough that the damped Newton rejected almost every
    step and stalled on a problem in which nothing had cracked.
    """
    ne = g.dofs.shape[0]
    U = u[g.dofs]
    w = g.wgt * thickness
    F = np.zeros(g.ndof)
    K_vals = np.zeros((ne, 8, 8)) if tangent else None
    new_cr = np.zeros((ne, 4), bool)
    new_th = np.zeros((ne, 4))
    eps_soft = GF / (g.h_band * FCT)
    gc_h = GC / g.h_band

    for k in range(4):
        B = g.Be[k]
        eps = U @ B.T
        ck, tk = state["cracked"][:, k], state["theta"][:, k]
        sig, th, is_cr = _response(eps, ck, tk, g.rho_x, g.rho_y,
                                   eps_soft, gc_h)
        new_cr[:, k], new_th[:, k] = is_cr, th
        F += np.bincount(g.dofs.ravel(),
                         weights=(np.einsum('ki,ek->ei', B, sig) * w).ravel(),
                         minlength=g.ndof)
        if tangent:
            D = np.empty((ne, 3, 3))
            for j in range(3):
                e2 = eps.copy()
                e2[:, j] += fd
                s2, _, _ = _response(e2, ck, tk, g.rho_x, g.rho_y,
                                     eps_soft, gc_h)
                D[:, :, j] = (s2 - sig) / fd
            K_vals += np.einsum('ki,ekl,lj->eij', B, D, B) * w

    if not tangent:
        return F, None, (new_cr, new_th)
    rows = np.repeat(g.dofs[:, :, None], 8, axis=2).ravel()
    cols = np.repeat(g.dofs[:, None, :], 8, axis=1).ravel()
    K = coo_matrix((K_vals.ravel(), (rows, cols)),
                   shape=(g.ndof, g.ndof)).tocsr()
    return F, K, (new_cr, new_th)


# --------------------------------------------------------------------------- #
# Displacement-controlled Newton
# --------------------------------------------------------------------------- #
STEP = 0.00625                 # deflection increment (mm), set by the study below


def schedule(delta: float, step: float = STEP) -> np.ndarray:
    """Deflection stations. Uniform, and much finer than looks necessary.

    This model commits history: a crack direction freezes at the state
    holding when the point cracks, so the answer depends on the path until
    the increment is small enough that the path is resolved. The size was
    fixed by measurement rather than by taste. At 0.1 mm the load factor
    jumped between branches and the curve was an artefact; at 0.0125 mm the
    curve looked smooth but the tie force at the identification cut was
    still 33 per cent away from its converged value, which is a reminder
    that a smooth curve is not evidence of a converged field. At 0.00625,
    0.003125 and 0.0015625 mm the load factor agrees to 1 per cent, the tie
    force at the cut to a few per cent, and the crack pattern is the same,
    so 0.00625 mm is used.
    """
    return np.arange(step, delta + 1e-9, step)


def solve(rho_tie: float, delta: float, nx=80, ny=40,
          thickness=300.0, P_ref=800.0e3, tol_rel=5e-4, max_it=60,
          lm_tries=20, verbose=False, steps=None):
    """Trace to a prescribed load-patch deflection and return the state there.

    Displacement control and Levenberg-Marquardt damping are the same
    strategy the reference driver uses. That is a shared solution algorithm,
    not a shared model: it converges each solver onto the equilibrium of its
    own constitutive.
    """
    g = build_grid(rho_tie, nx=nx, ny=ny)
    patch = 2 * g.load_nodes + 1
    prescribed = g.fixed.copy()
    prescribed[patch] = True
    free = ~prescribed
    tol = tol_rel * P_ref

    u = np.zeros(g.ndof)
    state = {"cracked": np.zeros((g.dofs.shape[0], 4), bool),
             "theta": np.zeros((g.dofs.shape[0], 4))}
    hist = []
    stations = schedule(delta) if steps is None else np.asarray(steps)
    n_steps = len(stations)
    for k, d in enumerate(stations):
        u[g.fixed] = 0.0
        u[patch] = -float(d)
        mu = 1e-3
        F, K, st = assemble(u, g, state, thickness)
        rn = float(np.linalg.norm(F[free]))
        converged = rn < tol
        nit = 0
        while not converged and nit < max_it:
            nit += 1
            Kff = K[free][:, free]
            scale = float(np.abs(Kff.diagonal()).mean()) or 1.0
            ok = False
            for _ in range(lm_tries):
                A = Kff + diags(mu * scale * np.ones(Kff.shape[0]))
                du = spsolve(A.tocsc(), -F[free])
                ut = u.copy()
                ut[free] += du
                F2, _, _ = assemble(ut, g, state, thickness, tangent=False)
                r2 = float(np.linalg.norm(F2[free]))
                if np.isfinite(r2) and r2 < rn:
                    ok = True
                    mu = max(mu * 0.5, 1e-9)
                    break
                mu *= 3.0
            if not ok:
                break
            u = ut
            F, K, st = assemble(u, g, state, thickness)
            rn = float(np.linalg.norm(F[free]))
            converged = rn < tol
        state["cracked"], state["theta"] = st
        lam = -float(F[patch].sum()) / P_ref
        hist.append((float(d), lam, rn, nit, converged,
                     float(state["cracked"].mean())))
        if verbose:
            print(f"  step {k + 1:>2}/{n_steps}  delta={d:5.3f}  "
                  f"lam={lam:+.4f}  it={nit:>2}  r={rn:.2e}  "
                  f"cracked={state['cracked'].mean():.2%}"
                  f"{'' if converged else '  [no conv]'}", flush=True)
    return u, g, state, hist


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def patch_test(nx=6, ny=4) -> float:
    """Uniform strain must produce zero residual at every interior node.

    The check exercises the element, the assembly and the constitutive
    together: any error in the B matrices or in the Gauss weights breaks it,
    and it is independent of anything the reference solver does.
    """
    g = build_grid(0.012, nx=nx, ny=ny)
    a, b, c = 3.0e-4, -1.0e-4, 2.0e-4
    x, y = g.xy[:, 0], g.xy[:, 1]
    u = np.empty(g.ndof)
    u[0::2] = a * x + 0.5 * c * y
    u[1::2] = b * y + 0.5 * c * x
    state = {"cracked": np.zeros((g.dofs.shape[0], 4), bool),
             "theta": np.zeros((g.dofs.shape[0], 4))}
    F, _, _ = assemble(u, g, state, tangent=False)
    interior = np.ones(g.ndof, bool)
    edge = ((x < 1e-9) | (x > g.xy[:, 0].max() - 1e-9)
            | (y < 1e-9) | (y > g.xy[:, 1].max() - 1e-9))
    interior[2 * np.where(edge)[0]] = False
    interior[2 * np.where(edge)[0] + 1] = False
    return float(np.abs(F[interior]).max())


def gauss_state(u, g: Q4Grid, state, thickness=300.0):
    """Gauss-point coordinates and stresses of the converged field.

    Written from this model's own constitutive, not the CSFM's, so the
    section check below is a statement about this solver rather than a
    comparison of two constitutives.
    """
    U = u[g.dofs]
    eps_soft = GF / (g.h_band * FCT)
    gc_h = GC / g.h_band
    xs, ys, sx, sy, txy = [], [], [], [], []
    ge = g.xy[g.dofs[:, 0::2] // 2]
    for k, (xi, eta) in enumerate(GAUSS):
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        xs.append(ge[:, :, 0] @ N)
        ys.append(ge[:, :, 1] @ N)
        eps = U @ g.Be[k].T
        sig, _, _ = _response(eps, state["cracked"][:, k],
                              state["theta"][:, k], g.rho_x, g.rho_y,
                              eps_soft, gc_h)
        sx.append(sig[:, 0])
        sy.append(sig[:, 1])
        txy.append(sig[:, 2])
    w = g.wgt * thickness
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(sx),
            np.concatenate(sy), np.concatenate(txy), w)


def section_check(u, g: Q4Grid, state, x_cut=700.0, H=1000.0,
                  thickness=300.0) -> dict:
    """Do the recovered stresses balance statics across a vertical cut?

    The identification is a moment reconciliation on a free body, so the
    only property of a generated field it actually needs is that the field
    equilibrates. This checks exactly that, and it checks it on the
    alternative model's own stresses, independently of anything the
    reference solver or the identification computes. The reaction arm is
    taken from where the bearing reaction really acts rather than from the
    nominal support centre, because on this problem the two differ by more
    than a hundred millimetres.
    """
    xg, yg, sx, sy, txy, w = gauss_state(u, g, state, thickness)
    sel = np.abs(xg - x_cut) < g.dx / 2.0
    dA = w / g.dx
    n_cut = float((sx[sel] * dA).sum())
    v_cut = float((txy[sel] * dA).sum())
    # Moment equilibrium of the free body to the left of the cut about the
    # mid-height of the cut gives M_int = -integral of sigma_x (y - y0) dA:
    # the first moment of the normal stress is the negative of the moment the
    # cut transmits, because the traction acts on a face whose outward normal
    # is +x. Reporting the raw first moment instead was the first version and
    # it made the closure come out at -115 per cent, which reads as a broken
    # field and is only a convention.
    m_cut = -float((sx[sel] * (yg[sel] - H / 2.0) * dA).sum())

    F, _, _ = assemble(u, g, state, thickness, tangent=False)
    ry = F[2 * g.supp_left + 1]
    reaction = float(ry.sum())
    x_r = float((g.xy[g.supp_left, 0] * ry).sum() / ry.sum())
    r_all = float(F[2 * np.concatenate([g.supp_left, g.supp_right]) + 1].sum())
    applied = -float(F[2 * g.load_nodes + 1].sum())
    return {"N_cut_kN": n_cut / 1e3, "V_cut_kN": v_cut / 1e3,
            "M_cut_kNm": m_cut / 1e6, "R_kN": reaction / 1e3,
            "x_reaction_mm": x_r,
            "M_statics_kNm": reaction * (x_cut - x_r) / 1e6,
            "V_statics_kN": -reaction / 1e3,
            "R_total_kN": r_all / 1e3, "P_applied_kN": applied / 1e3,
            "global_closure": r_all / applied}


def main() -> None:
    g0 = build_grid(0.012)
    print(f"crack band h = {g0.h_band:.1f} mm, f_ct = {FCT:.3f} MPa, "
          f"eps_soft = {GF / (g0.h_band * FCT):.2e}, Gc = {GC:.1f} N/mm, "
          f"compressive descent span at fcp = fc: "
          f"{2.0 * GC / g0.h_band / (FC * (1.0 - RESID_C)):.2e}")
    print(f"patch test, max interior residual = {patch_test():.3e} N")
    u, g, st, hist = solve(0.012, 3.5, verbose=True)
    d, lam, rn, nit, cv, fr = hist[-1]
    print(f"\nfinal: delta={d:.2f} mm  lam={lam:.4f}  resid={rn:.2e} N  "
          f"({rn / (lam * 800.0e3):.2%} of the applied load)  "
          f"cracked={fr:.1%}  converged={cv}")
    ck = section_check(u, g, st)
    print("\nsection check at x = 700 mm, from this model's own stresses")
    for k, v in ck.items():
        print(f"  {k:>16} {v:10.2f}")
    print(f"  moment closure {100 * ck['M_cut_kNm'] / ck['M_statics_kNm']:.1f}"
          f" %   shear closure "
          f"{100 * ck['V_cut_kN'] / ck['V_statics_kN']:.1f} %")


if __name__ == "__main__":
    main()
