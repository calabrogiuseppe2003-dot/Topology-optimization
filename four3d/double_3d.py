from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from ufl import max_value, min_value, exp, ln
import numpy as np
import os
import contextlib
from boxdouble import create_geometry_3d
import csv
from collections.abc import Iterable
from firedrake import DistributedMeshOverlapType
from netgen.occ import *


# --------------------------------------------------------------------
# Problem parameters
# --------------------------------------------------------------------

Re_value      = 2000
Re            = Constant(Re_value)  # Reynolds number
gbar          = 1.0           # max inlet/outlet velocity
dens          = Constant(1.0)  # density
l              = 1/6
mu            = dens * gbar / Re
nu            = 1.0 / Re       # nondimensional viscosity used in the forward problem
alphaunderbar = 0   # alpha_min in the original dimensional scaling
alphabar      = 2.5 * mu / (0.01**2)    # alpha_max in the original dimensional scaling
q             = Constant(0.01)          # continuing parameter for SIMP interpolation
maxh          = 0.1                   # maximum mesh size
volfrac       = 0.0873            # fluid volume fraction
delta = 1.5
target_volume = volfrac * delta
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
ngmesh, markers = create_geometry_3d(delta,maxh)
base = Mesh(
    ngmesh,
    distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)
    },
)
mh   = MeshHierarchy(base, refinement_levels=2, refinements_per_level=1)
mesh = mh[-1]

comm  = mesh.comm
rank0 = (comm.rank == 0)

h = CellDiameter(mesh)
n = FacetNormal(mesh)
(x,y,z) = SpatialCoordinate(mesh)

WALLS = [
    markers["wall_xmin"],
    markers["wall_xmax"],
    markers["wall_ymin"],
    markers["wall_ymax"],
    markers["wall_zmin"],
    markers["wall_zmax"],
]

INLET1 = markers["inlet1"]
INLET2 = markers["inlet2"]
INLET3 = markers["inlet3"]
INLET4 = markers["inlet4"]
OUTLET1 = markers["outlet1"]
OUTLET2 = markers["outlet2"]
OUTLET3 = markers["outlet3"]
OUTLET4 = markers["outlet4"]


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

r_pipe = l / 2  # raggio dei fori (diametro l/5)

# ----------------------------
# Inlet 1
# ----------------------------
rho1_sq = (y-0.3)**2 + (z-0.3)**2
val1 = gbar*(1 - rho1_sq/r_pipe**2)

# ----------------------------
# Inlet 2
# ----------------------------
rho2_sq = (y-0.3)**2 + (z-0.7)**2
val2 = gbar*(1 - rho2_sq/r_pipe**2)

# ----------------------------
# Inlet 3
# ----------------------------
rho3_sq = (y-0.7)**2 + (z-0.3)**2
val3 = gbar*(1 - rho3_sq/r_pipe**2)

# ----------------------------
# Inlet 4
# ----------------------------
rho4_sq = (y-0.7)**2 + (z-0.7)**2
val4 = gbar*(1 - rho4_sq/r_pipe**2)

# Boundary conditions for the forward problem
bcs = [
    # # u inlet and outlet
    DirichletBC(W.sub(0), as_vector([val1,0,0]), [INLET1,OUTLET1]),
    DirichletBC(W.sub(0), as_vector([val2,0,0]), [INLET2,OUTLET2]),
    DirichletBC(W.sub(0), as_vector([val3,0,0]), [INLET3,OUTLET3]),
    DirichletBC(W.sub(0), as_vector([val4,0,0]), [INLET4,OUTLET4]),
    # u inlet (use this instead of the previous to apply Neumann conditions)
    # DirichletBC(W.sub(0), as_vector([val1,0,0]), [INLET1]),
    # DirichletBC(W.sub(0), as_vector([val2,0,0]), [INLET2]),
    # DirichletBC(W.sub(0), as_vector([val3,0,0]), [INLET3]),
    # DirichletBC(W.sub(0), as_vector([val4,0,0]), [INLET4]),

    #u = 0 on walls
    DirichletBC(W.sub(0), Constant((0,0,0)) , WALLS ),
]

bcs_hom = homogenize(bcs)

# Solver parameters for the forward problem

sp_adj = {
    'mat_type': 'matfree',
    'snes_monitor': None,
    #'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 5e-6,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    #'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 300,
    'ksp_atol': 1e-07,
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
    'snes_monitor': None,
    #'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 5e-6,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    #'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 300,
    'ksp_atol': 1e-07,
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
      2*mu                 * inner(sym(grad(u)), sym(grad(v)))*dx(degree=10)
    -dens*                   inner(u ,div(outer(v,u)))*dx(degree=10)
    +                        inner(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS(degree=10)
    -                        inner(p, div(v))*dx(degree=10)
    -                        inner(q_test, div(u))*dx(degree=10)
    + alpha_perm(rho_k_filtered) * inner(u,v) * dx(degree=10)
    )

sig = Constant(1e4)

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

Fp = Res - inner(p/gamma, q_test)*dx(degree=10)  + inner(div(u)*gamma, div(v))*dx(degree=10)  #preconditioned residual
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
            }

a = r_min**2 * inner(grad(rho_trial), grad(v_test)) * dx + rho_trial * v_test * dx
L = rho_k * v_test * dx#(degree=10)
problem = LinearVariationalProblem(a, L, rho_k_filtered)
filter_solver = LinearVariationalSolver(problem, solver_parameters = sp_filter)


# --------------------------------------------------------------------
# Backward solve
# --------------------------------------------------------------------

lam = Function(W)  #NS adjoint
lam2 = Function(F)  #filter adjoint

(u, p) = split(w)
Jobj = 0.5 * (
    2.0*mu * inner(sym(grad(u)), sym(grad(u)))
    + alpha_perm(rho_k_filtered) * inner(u, u)
) * dx(degree=10)

# Navier-Stokes adjoint problem (built once)
adjA = adjoint(derivative(Res, w))
rhs = -derivative(Jobj, w)
JpT = adjoint(Jp)
_adj_prob = LinearVariationalProblem(adjA, rhs, lam, aP=JpT, bcs=bcs_hom)
_adj_solver = LinearVariationalSolver(_adj_prob, solver_parameters=sp_adj)

# dJ/d(rho_filtered) (built once, symbolic)
dFdrhof = derivative(Res, rho_k_filtered)
dJdrhof = derivative(Jobj, rho_k_filtered) + action(adjoint(dFdrhof), lam)

# Filter adjoint problem (built once)
rhs2 = -dJdrhof
_filter_adj_prob = LinearVariationalProblem(a, rhs2, lam2)
_filter_adj_solver = LinearVariationalSolver(_filter_adj_prob, solver_parameters=sp_filter)

# dJ/d(rho) (built once, symbolic)
dFdrho = -derivative(L, rho_k)
dJdrho_form = action(adjoint(dFdrho), lam2)


def build_functional():
    filter_solver.solve()
    forward()
    return assemble(Jobj)


def compute_derivative():
    """Solve the two adjoint systems (already built/factorised once
    above) and return the reduced gradient as an assembled cofunction."""
    _adj_solver.solve()
    _filter_adj_solver.solve()
    g_dual = assemble(dJdrho_form)
    return g_dual

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
    descent_tol=1e-8,
):
    if len(q_values) != len(iters_per_q):
        raise ValueError("q_values and iters_per_q must have the same length.")
    if simpl_type not in ("A", "B"):
        raise ValueError("simpl_type must be 'A' or 'B'.")

    def sigma(psi):
        return 1.0 / (1.0 + exp(-psi))

    def sigma_inv(rho):
        return ln(rho / (1.0 - rho))

    # ------------------------------------------------------------------
    # Preallocate all working Functions BEFORE the closures that use them
    # ------------------------------------------------------------------
    rho_viz = Function(VIZ)
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

        vol = sigma(psi_half - alpha_c * mu_c) * dx(degree=10)
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

    if rank0:
        os.makedirs("output-rol-firedrake2", exist_ok=True)
    comm.barrier()

    controls = VTKFile("output-rol-firedrake2/control_iterations.pvd")
    rhofilts = VTKFile("output-rol-firedrake2/rho_filtered_iterations.pvd")
    velocities = VTKFile("output-rol-firedrake2/velocity_iterations.pvd")
    rho_final = VTKFile("output-rol-firedrake2/rho_final.pvd")
    velocities_final = VTKFile("output-rol-firedrake2/velocity_final.pvd")

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
                "stage", "iter", "kkt", "J", "alpha", "volume", "backtracking",
                "newton_its", "krylov_its", "krylov_per_newton",
                "riesz_its", "filter_fwd_its", "filter_adj_its", "adj_ns_its",
            ])
            stage_writer.writerow([
                "stage", "q_value", "n_iterations_requested", "n_iterations_run", "converged",
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

            for k in range(niter):
                if k == 0:
                    J_current = float(build_functional())

                # ---- Gradient ------------------------------------------
                gk_dual = compute_derivative()

                adj_ns_its = _adj_solver.snes.getLinearSolveIterations()
                filter_adj_its = _filter_adj_solver.snes.getLinearSolveIterations()

                Riesz.solve(g_k, gk_dual)

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

                    rho_viz.interpolate(sigma(psi_new))

                    J_new = float(build_functional())

                    filter_fwd_its = filter_solver.snes.getLinearSolveIterations()
                    difference.interpolate(rho_k - rho_old)
                    descent = assemble(action(gk_dual, difference))

                    if descent > 0:
                        PETSc.Sys.Print(f"  k={k}: non-descent direction (descent={descent:.3e})")
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

                if descent > 0:
                    PETSc.Sys.Print(f"  k={k}: non-descent direction (descent={descent:.3e})")
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

                # ---- KKT residual --------------------------------------
                kkt = kkt_error(psi_new, psi_k, alpha_step, rho_k)
                if k == 0 and stage == 1:
                    kkt0 = max(abs(kkt), 1e-16)
                    kkt_rel = 1.0
                else:
                    kkt_rel = abs(kkt) / kkt0
                descent_val = float(descent)
                vol = assemble(rho_k * dx)
                PETSc.Sys.Print(
                    f"  k={k:3d}  J={J_new:.6e}  "
                    f"KKT={kkt:.3e}  vol={vol:.4f}  α={alpha_step:.3e}  bt={n_bt}  "
                    f"Newton={nonlinear_its}  Krylov={linear_its} ({krylov_per_newton:.1f}/Newton)  "
                    f"Riesz={riesz_its}  FiltFwd={filter_fwd_its}  FiltAdj={filter_adj_its}  AdjNS={adj_ns_its}"
                )
                if rank0:
                    writer.writerow([
                        stage,
                        k,
                        float(kkt),
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

                    # keep a history for end-of-run averages
                    riesz_its_history.append(riesz_its)
                    filter_fwd_its_history.append(filter_fwd_its)
                    filter_adj_its_history.append(filter_adj_its)
                    adj_ns_its_history.append(adj_ns_its)
                    newton_its_history.append(nonlinear_its)
                    krylov_per_newton_history.append(krylov_per_newton)

                # ---- Update memory -------------------------------------
                psi_prev.assign(psi_k)
                g_prev.assign(g_k)
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
                        f"  Stage {stage}: converged (KKT + descent) in {k + 1} iterations."
                    )
                    converged = True
                    break

            # NEW: log how many iterations this stage actually used,
            # against how many were requested, and whether it converged.
            n_iterations_run = k + 1
            if rank0:
                stage_writer.writerow([
                    stage,
                    float(q_value),
                    niter,
                    n_iterations_run,
                    bool(converged),
                ])
                stage_file.flush()

            if not converged:
                PETSc.Sys.Print(f"  Stage {stage}: maximum iterations reached without convergence.")

        rho_final.write(rho_viz)
        velocities_final.write(u_curr)
        rho_k_filtered.interpolate(rho_k)

    # ------------------------------------------------------------------
    # Average solver iterations over the whole optimization run.
    # ------------------------------------------------------------------
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
        PETSc.Sys.Print(f"  Forward filter solve (CG) ................... {avg_filter_fwd:.2f}")
        PETSc.Sys.Print(f"  Adjoint filter solve (CG) ................... {avg_filter_adj:.2f}")
        PETSc.Sys.Print(f"  Adjoint Navier-Stokes solve (FGMRES) ........ {avg_adj_ns:.2f}")
        PETSc.Sys.Print(f"  Forward Navier-Stokes Newton iterations ..... {avg_newton_its:.2f}")
        PETSc.Sys.Print(f"  Forward Navier-Stokes FGMRES per Newton ..... {avg_krylov_per_newton:.2f}")
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
        os.makedirs("output-rol-firedrake2", exist_ok=True)
    comm.barrier()

    rho_k.interpolate(Constant(float(volfrac)))
    filter_solver.solve()
    q_0 = 0.01*((alphabar - alpha_init) - volfrac * (alphabar - alphaunderbar)) / (volfrac * (alpha_init - alphaunderbar))
    q_vec =q_0 * np.array([1,0.5,0.1])
    c1 = 1e-3
    iters_per_q = (5,5,50)

    if float(Re) > 5:
        PETSc.Sys.Print(f"Using continuing strategy to reach Re={float(Re)}, with q={float(q_0)}.")
        Re_final = int(float(Re))
        Re_v = [1,10, 100] + list(range(200, Re_final + 1, 400))
        if Re_v[-1] != Re_final:
            Re_v.append(Re_final)

        u_curr, p_curr = w.subfunctions
        q.assign(float(q_0))

        for Re_ in Re_v:
            PETSc.Sys.Print(f"Current Re={Re_}")
            Re.assign(Re_)
            forward()
            VTKFile("output-rol-firedrake2/velocity_latest.pvd").write(u_curr)

    rho_opt, J_filtered = simpl(
        tol=3e-5,
        rho0=rho_k,
        target_volume=target_volume,
        q_values=q_vec,
        iters_per_q=iters_per_q,
        c1=c1,
        simpl_type="A",
        max_backtrack=10,
    )

    PETSc.Sys.Print(f"\nFinal volume: {float(assemble(rho_opt * dx)):.6f}  (target {target_volume:.6f})")
    PETSc.Sys.Print(f"Final objective filtered: {J_filtered:.6e}")