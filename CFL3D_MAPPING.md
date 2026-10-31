# CFL3D mapping (M0)

This note records the source path actually inspected in the local reference
`CFL3D-master` tree (the task table's `dist/mgblk.F` plus routines under
`source/cfl3d/libs/`).  The reference is CFL3D's single structured-block
inviscid path with `ivisc(1:3)=0`, `i2d=1`, and no multiblock/overset coupling.
The original source is left untouched.

## Data and index mapping

`resid.F` declares `q(jdim,kdim,idim,5)` and all boundary arrays use the same
three index directions.  `q` is primitive-like CFL3D storage, not the new
solver's conservative state: `q(1)=rho`, `q(2)=u`, `q(3)=v`, `q(4)=w`, and
`q(5)=p`.  The conversion is explicit in `wrest.F`/`rrest.F`: conservative
momentum is `rho*u,v,w`, and total-energy density is
`p/(gamma-1)+0.5*rho*(u^2+v^2+w^2)`.  The new two-dimensional code stores
`(rho,rho*u,rho*v,rho*E)` and sets `w=0`; this is a deliberate representation
difference.

For a two-dimensional block, CFL3D's physical plane is the J--K plane while I
is the third (extruded) direction.  `gfluxr.F` computes J-direction fluxes and
`hfluxr.F` computes K-direction fluxes; `ffluxr.F` is the I-direction routine.
The new code names its physical coordinates `xi=x` (axial) and `eta=y`
(transverse), so `xi` is mapped to the CFL3D K direction and `eta` to CFL3D J.
This mapping is recorded explicitly because I/J/K are storage directions, not
axis names.

There is also a physical-axis naming distinction: `update.F:74–75,123`
sets `q(...,3)` (the official physical **v** velocity) to zero when `i2d=1`.
For that standard x–z plane, the project's physical `(x,y)` and `(u,v)`
correspond to official `(x,z)` and `(u,w)` after renaming the transverse
coordinate. The project's omitted spanwise `w=0` is its own coordinate
convention; it is not a claim that official CFL3D zeroes `q(...,4)`.

## Residual, time step, and update

`dist/mgblk.F` drives the residual/update loop.  `resid.F` calls the three
direction inviscid routines, stores signed face contributions in `res`, and
computes the local step in `dtj`; the inviscid-only reduction used here is the
same face-sum convention.  `update.F` calls `af3f` for an implicit update and
then applies positivity safeguards before writing the new state.  The CPU
baseline uses a conservative finite-volume residual with one old-state pass
and forward Euler; this is the documented simplification of the full CFL3D
multigrid/time-stepping driver.

## Flux choice and switches

`gfluxr.F` and `hfluxr.F` select the inviscid scheme from `ifds(2)` and
`ifds(3)`:

* `0`: flux-vector splitting (`fluxp`/`fluxm`),
* `1`: flux-difference splitting (`fhat`),
* other values: MAPS+ splitting (`fmaps`).

The local `fhat.F` has now been inspected directly (2026-10-26). Its
purpose explicitly says "Compute Roe's generalized flux". The `cprec=0`
branch forms square-root-density Roe averages. At lines 203–214, the energy
dissipation contains the contact coefficient multiplied by
`H - c^2/(gamma-1) = (u^2+v^2+w^2)/2`, and the shear energy is the dot
product of the averaged velocity with the tangential momentum increment.
This exposed two errors in the project: contact energy `H-K` instead of `K`,
and shear energy `-ut` instead of `ut`. Both are corrected in CPU Roe and
in the duplicated device formula. Equal-state flux checks cannot detect
these errors; moving contact, shear, and supersonic upwind tests now do.

The project's entropy fix still uses a fixed width `0.1*c` on all four
characteristic speeds. Official `fhat.F` uses `epsa_r*(c+abs(u)+abs(v)+abs(w))`
and its own threshold convention. Thus the Roe eigenstructure is aligned,
but these entropy-fix parameters and the optional preconditioning are not
claimed to match a complete CFL3D configuration.

## Boundaries

`bc1005.F` constructs a wall-face state by removing the contravariant
normal velocity (lines 84–93), sets the boundary flag `bcj=1` (line 161),
and supplies boundary gradient data (lines 166–179). With this flag,
`gfluxr.F` uses the boundary state on both sides of the wall interface.
The former mapping's description of this as a simple reflected ghost cell
was imprecise. The project uses a reflected ghost **of the reconstructed
interior face state** in its Riemann flux; this is a simplified slip-wall
closure, not a line-by-line transplant of `bc1005.F`.

`bc2009.F` retains the total-inflow role and `bc2002.F` the specified-pressure
outflow role. The project's characteristic-compatible inlet and outlet use
the reconstructed state at the actual boundary face; the normal velocity
is reflected only at walls. Supersonic outlet states are extrapolated.
No artificial zero-slope band is imposed at the first/last row or column.
One-sided primitive gradients supply the missing boundary stencil.

## Explicit MUSCL repair (2026-10-26)

`gfluxr.F:101–160` and `hfluxr.F` difference each of the five primitive-like
`q` components, then call `xlim.F`. `xlim.F:113–145` includes an unlimited
kappa reconstruction and a minmod option, and `xlim.F:169–187` calls optional
positivity safeguards. The original half-factors act on **cell-to-cell
increments**. They do not justify an extra factor of one-half when the
project already multiplies a physical gradient by the **centre-to-face
distance**.

The project now reconstructs `(rho,u,v,p)` using physical-coordinate minmod
gradients. The mesh has constant-x columns: compute `dQ/dy` first, then
subtract `dQ/dy * delta_y` from axial neighbour differences to obtain
`dQ/dx`. It reconstructs at the actual face midpoint with the full
`gradient dot displacement`. This reproduces an affine field on the curved
mesh, including boundary cells. It is a mesh-specific two-dimensional
linear MUSCL implementation, not the exact CFL3D kappa family.

A cell-local factor theta scales all primitive gradients only if a face
would violate positive density/pressure. Every trigger is counted and
localized in `reconstruction.csv`. Forward-Euler substages are checked before
SSPRK3 convex combinations; a failed stage retries the entire step from the
saved old state with a halved step. `update.F:106–126` also distinguishes
density and pressure protection, but its implicit increment damping is not
copied into this explicit implementation.

SSPRK3 is an explicit addition in this project, not a claim about CFL3D's
`mgblk -> resid -> update -> af3f` implicit driver. The face-summed local
pseudo-time step uses the maximum wave speed of both reconstructed face
states. Local steps are held fixed throughout one RK cycle. See
[the repair report](docs/MUSCL_REPAIR_REPORT.md) for evidence, the unchanged
convergence criteria, and independent spatial-accuracy tests.

## Implicit path

`af3f.F` performs a three-factor directional approximate factorization.  It
calls `amafj.F` and `amafk.F` to assemble direction matrices, then either
`vlutr.F` + `bsub.F` for coupled 5x5 block tridiagonal lines or `diagj.F` and
`diagk.F` for scalar/diagonal approximations.  The block routines explicitly
document 5x5 blocks and non-pivoted LU.  `amafj/amafk` contain the state and
metric-dependent coefficients and add the local time term; the diagonal path
is a different approximation, not equivalent to the block path.

M3 uses the five-variable block structure as the CFL3D reference (the fifth
component is the spanwise momentum, held at zero in the 2-D physical model).
The M2 solver remains a four-variable Euler solver.  The M3 implementation is
an approximate CPU block-Thomas analogue with frozen numerical-flux Jacobians;
it does not claim to reproduce all CFL3D metric, preconditioning, viscous, or
boundary linearization terms.

## Nondimensionalization and scope

CFL3D uses reference values from `/ivals/` (`p0`, `rho0`, `c0`, etc.) and may
rescale momentum/energy in `rrest.F`/`wrest.F`; the exact values depend on the
input deck.  The new cases use one explicit internal convention,
`R=1, T0=1, p0=1, gamma=1.4`, and compare dimensional numerical values only
after documenting that convention.  No implicit equality between CFL3D and
case units is assumed.

| Item | This project | Status relative to CFL3D |
|---|---|---|
| Mesh/residual | 2-D cell-centred face sum | retained concept, single block |
| State | 4 conservative variables | simplified from 5 primitive-like `q` storage |
| Inviscid flux | Roe-type FDS + entropy fix | selected `ifds=1` analogue; exact `fhat` path not asserted |
| Reconstruction | primitive, geometry-aware MUSCL + minmod | explicit implementation; not identical to CFL3D kappa |
| Time update | Euler for first order; explicit SSPRK3 for MUSCL | simplified from CFL3D driver |
| Walls/inlet/outlet | slip / total inlet / pressure or extrapolated outlet | role retained, bookkeeping simplified |
| AF | CPU 5x5 block line solve with frozen Jacobians | analogue of `af3f`/`amaf*`/`vlutr`/`bsub` |
| Viscosity/turbulence/MG/MPI | omitted | out of scope |
