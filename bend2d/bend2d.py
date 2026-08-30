from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from ufl import max_value, min_value, exp, ln
import numpy as np
import os
import contextlib
from meshgenbend import create_geometry_bend
import csv
from collections.abc import Iterable
from firedrake import DistributedMeshOverlapType


# --------------------------------------------------------------------
# Problem parameters
# --------------------------------------------------------------------

Re            = Constant(5000)  # Reynolds number
gbar          = 1.0           # max inlet/outlet velocity
dens          = Constant(1.0)  # density
l              = 1/5
mu            = dens * gbar / Re
nu            = 1.0 / Re       # nondimensional viscosity used in the forward problem
alphaunderbar = 2.5 * mu / (1 / 5**2)   # alpha_min in the original dimensional scaling
alphabar      = 1e4 * alphaunderbar     # alpha_max in the original dimensional scaling
q             = Constant(0.01)          # continuing parameter for SIMP interpolation
maxh          = 0.004*4                  # maximum mesh size
volfrac       = 1/4            # fluid volume fraction
target_volume = volfrac
alpha_init    = 2.5 * mu / (0.1**2)
r_min         = 0.04                  #filter radius
gamma         = Constant(1e4)        #aumented lagrangian penalty-coefficient

def alpha_perm(rho):
    """Inverse permeability as a function of rho."""
    return alphaunderbar + (alphabar - alphaunderbar) * (1 - rho) / (1 + q * rho)


# --------------------------------------------------------------------
# Illinois root finder
# --------------------------------------------------------------------

def illinois(f, a, b, tol=1e-8, maxiter=100):
    fa = f(a)
    fb = f(b)

    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    if fa * fb > 0:
        raise ValueError("Opposite sign needed for a and b.")

    for _ in range(maxiter):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if abs(fc) < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
            fa *= 0.5
        else:
            a = c
            fa = fc
            fb *= 0.5

    return c


# --------------------------------------------------------------------
# Mesh generation
# --------------------------------------------------------------------
ngmesh, markers = create_geometry_bend(maxh)
base = Mesh(
    ngmesh,
    distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)
    },
)
mh   = MeshHierarchy(base, 2)
mesh = mh[-1]
h = CellDiameter(mesh)
n = FacetNormal(mesh)

# ----------------------------------------------------------------
# Communicator / rank helpers (used throughout for parallel-safe
# I/O and printing -- every MPI rank executes this script, so any
# plain `print()` or `open(...)` call would otherwise run on every
# rank simultaneously).
# ----------------------------------------------------------------
comm  = mesh.comm
rank0 = (comm.rank == 0)

BOTTOM_WALL_LEFT  = markers["bottom_wall_left"]
BOTTOM_OUTLET     = markers["bottom_outlet"]
BOTTOM_WALL_RIGHT = markers["bottom_wall_right"]
RIGHT_WALL        = markers["right_wall"]
TOP_WALL          = markers["top_wall"]
LEFT_WALL_TOP     = markers["left_wall_top"]
LEFT_INLET        = markers["left_inlet"]
LEFT_WALL_BOTTOM  = markers["left_wall_bottom"]

(x, y) = SpatialCoordinate(mesh)

# --------------------------------------------------------------------
# Function spaces and test/trial functions
# --------------------------------------------------------------------

A    = FunctionSpace(mesh, "DG", 0)  # function space for rho
F    = FunctionSpace(mesh, "CG", 1)  # function space for filtered rho
VIZ  = FunctionSpace(mesh, "DG", 0)  # function space to visualize rho

U_h = FunctionSpace(mesh, "MTW", 1)          # function space velocity
P_h = FunctionSpace(mesh, "DG", 0, variant = "integral")  # function space pressure
W = MixedFunctionSpace([U_h, P_h])

# --------------------------------------------------------------------
# Pre-assemble Riesz solver (mass matrix factorised once)
# --------------------------------------------------------------------

a_test = TestFunction(A)
b_trial = TrialFunction(A)

sp_riesz = {
    "ksp_type": "cg",
    #"ksp_monitor": None,
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-12,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "jacobi",
}

Riesz = LinearSolver(
    assemble(inner(a_test, b_trial) * dx),
    solver_parameters = sp_riesz,

)


# --------------------------------------------------------------------
# Forward solve
# --------------------------------------------------------------------

val1 = gbar * (1 - (2 * (y - 0.8) / l) ** 2)  # inlet parabolic profile
val2 = -gbar * (1 - (2 * (x - 0.8) / l) ** 2) # outlet parabolic profile

# Boundary conditions for the forward problem
bcs = [
    # u = 0 on all walls
    DirichletBC(W.sub(0), Constant((0.0,0.0)), [
        BOTTOM_WALL_LEFT,
        BOTTOM_WALL_RIGHT,
        RIGHT_WALL,
        TOP_WALL,
        LEFT_WALL_TOP,
        LEFT_WALL_BOTTOM,
    ]),

    # ux parabolic on inlet
    DirichletBC(W.sub(0), as_vector([val1,0.0]), LEFT_INLET),

    #uy parabolic on outlet, comment for Neumann
    #DirichletBC(W.sub(0), as_vector([0.0,val2]) , BOTTOM_OUTLET),  

]

bcs_hom = homogenize(bcs)

# Solver parameters for the forward problem

sp_adj = {
    'mat_type': 'matfree',
    #'snes_monitor': None,
    #'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    #'ksp_converged_reason': None,
    #'ksp_monitor_true_residual': None,
    'ksp_max_it': 300,
    'ksp_atol': 1e-08,
    'ksp_rtol': 1e-10,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'pc_fieldsplit_0_fields': 1,
    'pc_fieldsplit_1_fields': 0,
    'fieldsplit_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'jacobi',
    'fieldsplit_1': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled': {
           'pc_use_amat': False,
           'pc_type': 'mg',
           'pc_mg_type': 'full',
           'mg_coarse_mat_type': 'aij',
           'mg_coarse_pc_type': 'lu',
           'mg_coarse_pc_factor_mat_solver_type': 'mumps',
           'mg_coarse_mat_mumps_icntl_14': 1000,
           'mg_levels': {
               'ksp_convergence_test': 'skip',
               'ksp_max_it': 5,
               'ksp_type': 'fgmres',
               'pc_type': 'python',
               'pc_python_type': 'firedrake.ASMStarPC',
           },
        },
    },
}

sp = {
    'mat_type': 'matfree',
    #'snes_monitor': None,
    #'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    #'ksp_converged_reason': None,
    #'ksp_monitor_true_residual': None,
    'ksp_max_it': 300,
    'ksp_atol': 1e-08,
    'ksp_rtol': 1e-10,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'pc_fieldsplit_0_fields': 1,
    'pc_fieldsplit_1_fields': 0,
    'fieldsplit_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'jacobi',
    'fieldsplit_1': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled': {
           'pc_use_amat': False,
           'pc_type': 'mg',
           'pc_mg_type': 'full',
           'mg_coarse_mat_type': 'aij',
           'mg_coarse_pc_type': 'lu',
           'mg_coarse_pc_factor_mat_solver_type': 'mumps',
           'mg_coarse_mat_mumps_icntl_14': 1000,
           'mg_levels': {
               'ksp_convergence_test': 'skip',
               'ksp_max_it': 5,
               'ksp_type': 'fgmres',
               'pc_type': 'python',
               'pc_python_type': 'firedrake.ASMStarPC',
           },
        },
    },
}

rho_k = Function(A)  #density function
rho_k_filtered = Function(F)  #filtered density

w = Function(W)  # create function to hold the solution (u,p)
(u, p) = split(w)

y_test = TestFunction(W)
(v, q_test) = split(y_test)

uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u   #flux of u across internal facets to stabilise the advection term

Res = (
      2/Re                 * inner(sym(grad(u)), sym(grad(v)))*dx#(degree=8)
    -                   inner(u ,div(outer(v,u)))*dx(degree=10)
    +                        inner(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS(degree=8)
    -                        inner(p, div(v))*dx#(degree=8)
    -                        inner(q_test, div(u))*dx#(degree=8)
    + alpha_perm(rho_k_filtered) * inner(u,v) * dx(degree=8)
    )

def c_bc(u, v, bid, g):
    if g is None:
        uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
    else:
        uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g
    return dot(v, uflux_ext)*ds(bid,degree=10)

exterior_markers = set(mesh.exterior_facets.unique_markers)

for bc in bcs:
    if "DG" in str(bc._function_space):
        continue
    g = bc.function_arg
    bid = bc.sub_domain
    if isinstance(bid, Iterable):
        [exterior_markers.remove(_) for _ in bid]
    else:
        exterior_markers.remove(bid)
    Res += c_bc(u, v, bid, g) 

for bid in exterior_markers:
    Res += c_bc(u, v, bid, None)

Fp = Res - inner(p/gamma, q_test)*dx  + inner(div(u)*gamma, div(v))*dx  #preconditioned residual
Jp = derivative(Fp, w)  #preconditioned jacobian
J = derivative(Res,w)  #jacobian

problem = NonlinearVariationalProblem(Res, w, J=J , Jp=Jp, bcs=bcs)
NS_solver = NonlinearVariationalSolver(problem, solver_parameters = sp)


def forward():
    """Solve the forward problem for a given fluid distribution rho(x) and save the solution in w."""
    NS_solver.solve()
    return

# --------------------------------------------------------------------
# Filtering operation
# --------------------------------------------------------------------

v_test = TestFunction(F)
rho_trial = TrialFunction(F)

sp_filter = {
                "ksp_type": "cg", # use conjugate gradients
                #"ksp_monitor": None, # print info about iteration
                "ksp_rtol": 1.0e-10, # residual relative tolerance
                "pc_type": "mg", # use geometric multigrid
                #"ksp_view": None,
            }

a = r_min**2 * inner(grad(rho_trial), grad(v_test)) * dx + rho_trial * v_test * dx
L = rho_k * v_test * dx#(degree=1)
problem = LinearVariationalProblem(a, L, rho_k_filtered)
filter_solver = LinearVariationalSolver(problem, solver_parameters = sp_filter)


# --------------------------------------------------------------------
# Backward solve
# --------------------------------------------------------------------

w_adj = Function(W)
lam,pi = split(w_adj)
lam2 = Function(F)  #filter adjoint
w_trial = TrialFunction(W)
v_trial, p_trial = split(w_trial)
adj_ns = adjoint(derivative(Res, w))
JpT = adjoint(Jp)

rhs = (2/Re * inner(sym(grad(u)), sym(grad(v))) + alpha_perm(rho_k_filtered) * inner(u, v)) * dx

_adj_prob = LinearVariationalProblem(adj_ns, rhs, w_adj, bcs=bcs_hom, aP=JpT)
_adj_solver = LinearVariationalSolver(_adj_prob, solver_parameters=sp_adj)

alpha_prime = -(alphabar - alphaunderbar) * (1 + q) / (1 + q * rho_k_filtered)**2
rhs_adj_filt = alpha_prime * v_test *(0.5*inner(u,u) - inner(u,lam)) *dx
_filter_adj_prob = LinearVariationalProblem(a, rhs_adj_filt, lam2)
_filter_adj_solver = LinearVariationalSolver(_filter_adj_prob, solver_parameters=sp_filter)

Jobj = 0.5 * (
    2.0/Re * inner(sym(grad(u)), sym(grad(u)))
    + alpha_perm(rho_k_filtered) * inner(u, u)
) * dx


def build_functional():
    filter_solver.solve()
    forward()
    return assemble(Jobj)


def compute_derivative():
    _adj_solver.solve()
    _filter_adj_solver.solve()
    return lam2

# --------------------------------------------------------------------
# SiMPL optimizer
# --------------------------------------------------------------------

def simpl(
    tol,
    rho0,
    target_volume,
    q_values=(0.01, 0.1),
    iters_per_q=(18, 30),
    c1=1e-3,
    simpl_type="A",
    max_backtrack=50,
    descent_tol=None,
):
    if len(q_values) != len(iters_per_q):
        raise ValueError("q_values and iters_per_q must have the same length.")
    if simpl_type not in ("A", "B"):
        raise ValueError("simpl_type must be 'A' or 'B'.")

    if descent_tol is None:
        descent_tol = tol

    def sigma(psi):
        return 1.0 / (1.0 + exp(-psi))

    def sigma_inv(rho):
        return ln(rho / (1.0 - rho))

    # ------------------------------------------------------------------
    # Preallocate all working Functions BEFORE the closures that use them
    # ------------------------------------------------------------------
    rho_viz = Function(VIZ)
    rho_filtered_prev = Function(F, name="rho_filtered_prev")
    psi_k = Function(A, name="psi_k")
    psi_prev = Function(A, name="psi_prev")
    g_k = Function(A, name="g_k")
    g_prev = Function(A, name="g_prev")
    psi_half = Function(A, name="psi_half")
    psi_new = Function(A, name="psi_new")
    rho_old = Function(A, name="rho_old")
    lam_kkt = Function(A, name="lam_kkt")
    eta = Function(A, name="eta")
    difference = Function(A, name="Difference")
    mu_c = Constant(0.0)
    u_curr, p_curr = w.subfunctions  # current velocity and pressure

    alpha_c = Constant(1.0)
    mu_val_c = Constant(0.0)

    bb_num_f = Function(A, name="bb_num_scratch")
    bb_den_f = Function(A, name="bb_den_scratch")

    div_scratch = Function(A, name="kl_div_scratch")

    vol_check_f = Function(A, name="vol_check_scratch")

    def divergence(rho, q_ref):
        """KL-divergence-like term for type-B line search."""
        div_scratch.interpolate(rho * ln(rho / q_ref) + (1 - rho) * ln((1 - rho) / (1 - q_ref)))
        return assemble(div_scratch * dx)

    # ------------------------------------------------------------------
    # KKT estimator
    # ------------------------------------------------------------------
    def kkt_error(psi_new, psi_old, alpha_step, rho_old):
        eps = 1e-8
        alpha_c.assign(alpha_step)
        lam_kkt.interpolate((psi_new - psi_old) / alpha_c)  # approximate Lagrange multiplier

        eta.interpolate(max_value(-rho_old * lam_kkt, (1 - rho_old) * lam_kkt))

        return assemble(eta * dx)

    # ------------------------------------------------------------------
    # Generalised Barzilai-Borwein step size
    # ------------------------------------------------------------------
    def alpha_gbb(alpha_prev):
        bb_num_f.interpolate((psi_k - psi_prev) * (rho_k - rho_old))
        bb_den_f.interpolate((g_k - g_prev) * (rho_k - rho_old))
        num = assemble(bb_num_f * dx)
        den = abs(assemble(bb_den_f * dx))

        if num <= 0.0:
            raise ValueError("Negative numerator in GBB step size calculation.")

        return np.sqrt((num / den) * alpha_prev)

    # ------------------------------------------------------------------
    # Volume projection
    # ------------------------------------------------------------------
    def find_mu(alpha_step):
        alpha_c.assign(alpha_step)
        psi_half.interpolate(psi_k - alpha_c * g_k)

        vol_check_f.interpolate(sigma(psi_half))
        if assemble(vol_check_f * dx) <= target_volume:  # inactive constraint
            return 0.0

        vol = sigma(psi_half - alpha_c * mu_c) * dx
        def residual(mu_val):
            mu_c.assign(mu_val)
            return assemble(vol) - target_volume

        with g_k.dat.vec_ro as gvec:
            _, min_val = gvec.min()
        b = float(-min_val)

        if b <= 0.0:
            if rank0:
                PETSc.Sys.Print("Warning: non-positive b in find_mu. Returning 0.0.")
            return 0.0

        return illinois(residual, 0.0, b)

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    rho_k.assign(rho0)
    psi_k.interpolate(sigma_inv(rho_k))
    filter_solver.solve()
    rho_filtered_prev.assign(rho_k_filtered)

    if rank0:
        os.makedirs("output-rol-firedrake2", exist_ok=True)
    comm.barrier()

    controls = VTKFile("output-rol-firedrake2/control_iterations.pvd")
    rhofilts = VTKFile("output-rol-firedrake2/rho_filtered_iterations.pvd")
    velocities = VTKFile("output-rol-firedrake2/velocity_iterations.pvd")

    rho_viz.interpolate(rho0)
    controls.write(rho_viz)

    VTKFile("output-rol-firedrake2/rho_latest.pvd").write(rho_viz)
    log_path = "output-rol-firedrake2/iterazioni.csv"
    stage_summary_path = "output-rol-firedrake2/stage_summary.csv"

    riesz_its_history = []           # CG its, mass-matrix inversion (Riesz representative)
    filter_fwd_its_history = []      # CG its, forward filter solve
    filter_adj_its_history = []      # CG its, adjoint filter solve
    adj_ns_its_history = []          # FGMRES its, adjoint Navier-Stokes solve
    newton_its_history = []          # Newton its, forward Navier-Stokes solve
    krylov_per_newton_history = []   # average FGMRES its per Newton step, forward NS solve

    log_ctx = open(log_path, "w", newline="") if rank0 else contextlib.nullcontext()
    stage_ctx = open(stage_summary_path, "w", newline="") if rank0 else contextlib.nullcontext()
    with log_ctx as log_file, stage_ctx as stage_file:
        writer = csv.writer(log_file) if rank0 else None
        stage_writer = csv.writer(stage_file) if rank0 else None
        if rank0:
            writer.writerow([
                "stage", "iter", "kkt", "descent", "J", "alpha", "volume", "backtracking",
                "newton_its", "krylov_its", "krylov_per_newton",
                "riesz_its", "filter_fwd_its", "filter_adj_its", "adj_ns_its",
            ])
            stage_writer.writerow([
                "stage", "q_value", "n_iterations_requested", "n_iterations_run", "converged",
                "final_kkt", "final_descent",
            ])

        # ------------------------------------------------------------------
        # Outer loop over q-continuation stages
        # ------------------------------------------------------------------
        for stage, (q_value, niter) in enumerate(zip(q_values, iters_per_q), start=1):
            PETSc.Sys.Print(f"\n{'=' * 60}")
            PETSc.Sys.Print(f"Stage {stage}: q = {float(q_value)}, max_iter = {niter}")
            PETSc.Sys.Print(f"{'=' * 60}")

            q.assign(float(q_value))

            # initialise variables for the inner loop
            alpha_prev = None
            converged = False
            g_initialised = False
            n_bt = 0
            k = -1  # in case niter == 0, so n_iterations_run below is well defined
            kkt = float("nan")
            descent_val = float("nan")

            for k in range(niter):
                if k == 0:
                    J_current = float(build_functional())

                # ---- Gradient ------------------------------------------
                gk_dual = compute_derivative()

                adj_ns_its = _adj_solver.snes.getLinearSolveIterations()      # FGMRES its, adjoint NS
                filter_adj_its = _filter_adj_solver.snes.getLinearSolveIterations()  # CG its, adjoint filter

                Riesz.solve(g_k, assemble(inner(gk_dual, TestFunction(A)) * dx))

                riesz_its = Riesz.ksp.getIterationNumber()

                with g_k.dat.vec_ro as gvec:
                    gnorm = gvec.norm(PETSc.NormType.NORM_INFINITY)

                if gnorm < 1e-14:
                    PETSc.Sys.Print(f"  k={k}: zero gradient — stopping.")
                    converged = True
                    break

                # ---- Step size -----------------------------------------
                if not g_initialised:
                    alpha_step = 1.0 / gnorm
                else:
                    alpha_step = alpha_gbb(alpha_prev)

                rho_old.assign(rho_k)

                # ---- Armijo backtracking --------------------------------
                if k > 0 and n_bt == 0:
                    alpha_step *= 1.5
                n_bt = 0
                while True:
                    mu_val = find_mu(alpha_step)
                    mu_val_c.assign(mu_val)
                    alpha_c.assign(alpha_step)

                    psi_new.interpolate(psi_half - alpha_c * mu_val_c)
                    rho_k.interpolate(sigma(psi_new))

                    J_new = float(build_functional())
                    difference.interpolate(rho_k - rho_old)
                    descent = assemble(g_k * difference * dx)

                    if descent > 0:
                        #PETSc.Sys.Print(f"  k={k}: non-descent direction (descent={descent:.3e})")
                        break

                    if simpl_type == "A":
                        armijo_rhs = J_current + c1 * descent
                    else:
                        armijo_rhs = J_current + descent + (1.0 / alpha_step) * divergence(rho_k, rho_old)

                    if J_new <= armijo_rhs:
                        break

                    alpha_step *= 0.5
                    n_bt += 1
                    if n_bt > max_backtrack:
                        break

                if n_bt > max_backtrack:
                    PETSc.Sys.Print(f"  k={k}: maximum backtracking iterations reached.")
                    break
                if alpha_step < 1e-8:
                    PETSc.Sys.Print(f"  k={k}: step size too small during backtracking.")
                    break

                nonlinear_its = NS_solver.snes.getIterationNumber()
                linear_its = NS_solver.snes.getLinearSolveIterations()
                krylov_per_newton = (linear_its / nonlinear_its) if nonlinear_its > 0 else 0.0

                filter_fwd_its = filter_solver.snes.getLinearSolveIterations()

                # ---- KKT residual --------------------------------------
                kkt = kkt_error(psi_new, psi_k, alpha_step, rho_k)
                if k == 0 and stage == 1:
                    kkt0 = max(abs(kkt), 1e-16)   # evita divisione per zero
                    kkt_rel = 1.0
                else:
                    kkt_rel = abs(kkt) / kkt0
                descent_val = float(descent)
                vol = assemble(rho_k * dx)
                PETSc.Sys.Print(
                    f"  k={k:3d}  J={J_new:.6e}  "
                    f"KKT={kkt:.3e}  descent={descent_val:.3e}  vol={vol:.4f}  α={alpha_step:.3e}  bt={n_bt}  "
                    f"Newton={nonlinear_its}  Krylov={linear_its} ({krylov_per_newton:.1f}/Newton)  "
                    f"Riesz={riesz_its}  FiltFwd={filter_fwd_its}  FiltAdj={filter_adj_its}  AdjNS={adj_ns_its}"
                )
                if rank0:
                    writer.writerow([
                        stage,
                        k,
                        float(kkt),
                        float(descent_val),
                        float(J_new),
                        float(alpha_step),
                        float(vol),
                        n_bt,
                        int(nonlinear_its),
                        int(linear_its),
                        float(krylov_per_newton),
                        int(riesz_its),
                        int(filter_fwd_its),
                        int(filter_adj_its),
                        int(adj_ns_its),
                    ])
                    log_file.flush()

                    # NEW: accumulate for the end-of-run averages.
                    riesz_its_history.append(riesz_its)
                    filter_fwd_its_history.append(filter_fwd_its)
                    filter_adj_its_history.append(filter_adj_its)
                    adj_ns_its_history.append(adj_ns_its)
                    newton_its_history.append(nonlinear_its)
                    krylov_per_newton_history.append(krylov_per_newton)

                # ---- Update memory -------------------------------------
                psi_prev.assign(psi_k)
                g_prev.assign(g_k)
                rho_filtered_prev.assign(rho_k_filtered)
                rho_viz.interpolate(sigma(psi_new))
                alpha_prev = alpha_step
                g_initialised = True

                # ---- Accept step ---------------------------------------
                J_current = J_new
                psi_k.assign(psi_new)

                # write out the current state of the optimization
                controls.write(rho_viz)
                rhofilts.write(rho_k_filtered)
                velocities.write(u_curr)

                VTKFile("output-rol-firedrake2/rho_latest.pvd").write(rho_viz)
                VTKFile("output-rol-firedrake2/velocity_latest.pvd").write(u_curr)

                # ---- Convergence check ---------------------------------
                if kkt_rel <= tol or descent_val >= -descent_tol:
                    PETSc.Sys.Print(
                        f"  Stage {stage}: converged (KKT or descent) in {k + 1} iterations."
                    )
                    converged = True
                    break

            n_iterations_run = k + 1
            if rank0:
                stage_writer.writerow([
                    stage,
                    float(q_value),
                    niter,
                    n_iterations_run,
                    bool(converged),
                    float(kkt),
                    float(descent_val),
                ])
                stage_file.flush()

            if not converged:
                PETSc.Sys.Print(f"  Stage {stage}: maximum iterations reached without convergence.")

        VTKFile("output-rol-firedrake2/rho_final.pvd").write(rho_viz)
        VTKFile("output-rol-firedrake2/velocity_final.pvd").write(u_curr)

    def _safe_mean(history):
        return float(np.mean(history)) if len(history) > 0 else float("nan")

    avg_riesz = _safe_mean(riesz_its_history)
    avg_filter_fwd = _safe_mean(filter_fwd_its_history)
    avg_filter_adj = _safe_mean(filter_adj_its_history)
    avg_adj_ns = _safe_mean(adj_ns_its_history)
    avg_newton_its = _safe_mean(newton_its_history)
    avg_krylov_per_newton = _safe_mean(krylov_per_newton_history)

    if rank0:
        PETSc.Sys.Print(f"\n{'=' * 60}")
        PETSc.Sys.Print("Average solver iterations over the whole optimization run:")
        PETSc.Sys.Print(f"  Riesz / mass-matrix inversion (CG) ......... {avg_riesz:.2f}")
        PETSc.Sys.Print(f"  Forward filter solve (CG) ................. {avg_filter_fwd:.2f}")
        PETSc.Sys.Print(f"  Adjoint filter solve (CG) .................. {avg_filter_adj:.2f}")
        PETSc.Sys.Print(f"  Adjoint Navier-Stokes solve (FGMRES) ....... {avg_adj_ns:.2f}")
        PETSc.Sys.Print(f"  Forward Navier-Stokes Newton iterations .... {avg_newton_its:.2f}")
        PETSc.Sys.Print(f"  Forward Navier-Stokes FGMRES per Newton .... {avg_krylov_per_newton:.2f}")
        PETSc.Sys.Print(f"{'=' * 60}")

        summary_path = "output-rol-firedrake2/solver_iterations_summary.csv"
        with open(summary_path, "w", newline="") as f:
            summary_writer = csv.writer(f)
            summary_writer.writerow(["quantity", "average_iterations", "n_samples"])
            summary_writer.writerow(["riesz_mass_matrix_cg", avg_riesz, len(riesz_its_history)])
            summary_writer.writerow(["filter_forward_cg", avg_filter_fwd, len(filter_fwd_its_history)])
            summary_writer.writerow(["filter_adjoint_cg", avg_filter_adj, len(filter_adj_its_history)])
            summary_writer.writerow(["adjoint_ns_fgmres", avg_adj_ns, len(adj_ns_its_history)])
            summary_writer.writerow(["forward_ns_newton_its", avg_newton_its, len(newton_its_history)])
            summary_writer.writerow(["forward_ns_fgmres_per_newton", avg_krylov_per_newton, len(krylov_per_newton_history)])

    return rho_k, J_current


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == "__main__":
    if rank0:
        os.makedirs("output-rol-firedrake", exist_ok=True)
    comm.barrier()

    rho_k.interpolate(Constant(float(target_volume)))
    filter_solver.solve()
    q_0 = ((alphabar - alpha_init) - target_volume * (alphabar - alphaunderbar)) / (target_volume * (alpha_init - alphaunderbar))
    q_vec = q_0 * np.array([1, 0.5,0.1,0.05])
    c1 = 1e-3
    iters_per_q = (5, 5, 5, 100)
    toll = 1e-5

    if float(Re) > 5:
        PETSc.Sys.Print(f"Using continuing strategy to reach Re={float(Re)}.")
        q.assign(float(q_0))
        Re_final = int(float(Re))
        Re_v = [1,10, 100] + list(range(200, Re_final + 1, 300))
        if Re_v[-1] != Re_final:
            Re_v.append(Re_final)

        u_curr, p_curr = w.subfunctions
        for Re_ in Re_v:
            PETSc.Sys.Print(f"Current Re={Re_}")
            Re.assign(Re_)
            forward()

    rho_opt, J_filtered = simpl(
        tol=toll,
        rho0=rho_k,
        target_volume=target_volume,
        q_values=q_vec,
        iters_per_q=iters_per_q,
        c1=c1,
        simpl_type="A",
        max_backtrack=10,
        descent_tol=1e-9
    )

    PETSc.Sys.Print(f"\nFinal volume: {float(assemble(rho_opt * dx)):.6f}  (target {target_volume:.6f})")
    PETSc.Sys.Print(f"Final objective filtered: {J_filtered:.6e}")