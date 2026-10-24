# Replica machinery for BootLoops — design

Status: draft for approval. Nothing has been built.
Marks: **[RV]** = decided by Renato; **[call]** = my call, open to override.

## Intent

BootLoops lets an LLM do quantitative science whose answers can be
checked. Statistical mechanics of disordered systems is a large body of
published calculations (spin glasses, neural-network capacity, learning
curves, inference) that all run on the same replica method, and today an
agent has no instrument for it.

**[RV]** Build general machinery, not one model. It must know the replica
calculations in the literature *in detail*, be able to **repeat their
steps**, and **reuse similar steps in new calculations**.

Success: given a model that is not in the catalog but is built from familiar
parts (a known Hamiltonian family with a changed coupling law, constraint,
or prior), the package assembles the calculation from catalogued steps,
solves it, and the answer survives an independent check that did not feed it.

## Shape of the thing

A new package under `tools/`, working name **`replicant`** **[call]**, laid
out like its neighbours: `GUIDE.md`, `README.md`, code, a selftest battery
(row in `tools/README.md`, line in `tools/BATTERIES.json`), default battery
about 20 s. Python with sympy and mpmath, certified numerics through
`baller` where it applies **[call]**; package docs in English to match the
repository.

**[RV]** Steps exist at two levels, and both are in v1:

- **(a) Symbolic derivation.** Steps act on sympy expressions and derive
  the equations from the model: the Gaussian disorder average, the
  introduction of q_ab and conjugates, the saddle-point equations, the
  n→0 rules for sums over replica indices.
- **(b) Family pipelines.** For each catalogued family (SK, perceptron,
  Hopfield, ...), a pre-derived pipeline of named stages with hand-written,
  tested implementations of the heavier parts (ansatz, solver, stability).

**[call]** The two levels check each other: wherever (a) can derive a
catalogued family's equations, they must agree with (b)'s pre-derived ones,
and the battery tests that. (b) is the dependable base; (a) extends reach
to models no family covers, as far as its steps allow, and the composer
says plainly where it stops.

Three layers:

1. **Step library.** Each step is a typed transformation with stated
   preconditions and a stated output, so steps compose and a failed
   precondition is a named refusal, not a silent wrong answer. Steps:
   replicate (Zⁿ); average over disorder; introduce order parameters q_ab
   (and conjugates); factorize into single-site or single-pattern problems;
   write the saddle-point equations; impose an ansatz (RS, 1RSB, full RSB);
   continue n→0; solve (fixed point, population dynamics, Parisi PDE);
   stability (replicon / Almeida–Thouless); extract observables (free
   energy, overlap, capacity, generalization error).
2. **Catalog.** Each published calculation is a recorded sequence of those
   steps over a model specification, with its citation, its steps, and its
   banked reference values.
3. **Composer.** Builds a new calculation by matching a model specification
   against catalogued steps, and says which steps it reused, which it
   adapted, and which it could not supply.

## Cavity method in v1 **[RV: include cavity]**

The cavity method is a second route to the same quantities, built from the
same kind of typed steps: remove a site, write the cavity fields' joint
law, add the site back, impose self-consistency. Scope **[call]**:

- RS cavity on fully connected models (SK, Hopfield, perceptron), giving
  the same equations as the replica RS saddle point.
- Cavity on sparse random graphs: belief-propagation fixed points and
  population dynamics (Mézard–Parisi), with 1RSB cavity (survey-style
  population of populations) for the sparse catalog entries that need it.
- Cavity steps share the catalog format: each entry may carry a replica
  route, a cavity route, or both, with its citation.

TAP equations stay out of v1; cavity is used here for the cavity fields
and self-consistency, not for TAP free energies.

**Cross-check added:** for every catalog entry that has both routes, the
replica result and the cavity result must agree to the stated tolerance.
This is a third independent check beside the (a)/(b) agreement, and it
also gives the held-out models a second independent route where one
exists. Population-dynamics numbers are statistical and are labelled as
such, not certified.

Effect on size: this enlarges v1 noticeably. If it proves too large at
plan time, the plan may stage cavity as the second milestone, but it stays
in v1 scope.

## Initial catalog (v1) **[call]**

Random energy model; sparse-graph entries with a cavity route (e.g.
Viana–Bray / Ising spin glass on a random regular or Erdős–Rényi graph);
Sherrington–Kirkpatrick (RS, Almeida–Thouless line,
1RSB, Parisi full RSB); spherical and Ising perceptron (Gardner capacity);
Hopfield (Amit–Gutfreund–Sompolinsky); teacher–student perceptron learning
curves; spherical p-spin. Reference values are taken from the papers with
citation during the build, never from memory, and each entry is also pinned
by a route that does not use the paper's number (closed form where one
exists, exact enumeration at small N, or flagged-statistical simulation).

## Verification

- Every step has a unit test against an independently derived expression.
- Every catalog entry reproduces its published numbers to the stated
  digits.
- **Held-out test of reuse** **[RV: follow my suggestion]**: two
  calculations the composer assembles for models not in the catalog, each
  checked against an independent route (enumeration at small N, or a limit
  with a known answer): (1) SK with a changed coupling law, e.g. ±J instead
  of Gaussian; (2) the perceptron with a changed prior or constraint, e.g.
  Ising instead of spherical. Each must use a model for which a valid
  ansatz exists. A positive control shows the check can fail (a
  deliberately wrong ansatz must be rejected).
- **Level agreement:** (a)'s derived equations equal (b)'s pre-derived ones
  for every family where both exist.
- Provenance: a number that fed a fit never certifies it (house rule).
- Refusal over invention: if a precondition fails or the composer lacks a
  step, it stops and names what is missing. No silent fallback to RS when
  the stability test says RS is unstable.
- **Honest labelling.** The n→0 continuation and RSB ansätze are
  physics-level, not proven. Outputs say so; the numerics inside them carry
  their own certified or statistical error bars, stated as such.

## Goals

- Steps usable on their own, not only inside catalog entries.
- A reader can follow any catalog entry step by step to the published
  derivation.
- Catalog grows by adding entries, without touching the core.
- **[RV, 2026-10-23]** The system writes LaTeX that details each calculation: every step with its equations, the citations, the banked-versus-computed check table, and the physics-level and statistical labels. Displayed equations that the code can evaluate are tested against the implementation, so the document cannot drift from the code. A PDF is compiled when a TeX engine (`tectonic` or `pdflatex`) is on the path.

## Non-goals (v1) **[call]**

TAP equations and dynamical (DMFT) methods (cavity is in v1, see below); quantum models; automatic
discovery of the right RSB ansatz for an arbitrary model (the ansatz is
chosen by the catalog or the user); rigorous proofs (Guerra, Talagrand);
non-Gaussian disorder beyond what the catalog entries need.

## Anti-goals

- Reference numbers typed in from the paper and compared to themselves.
- Steps that only work inside the one entry they came from, so that reuse
  is nominal.
- A confident answer for a model where a precondition failed.

## Left to the builder

How far level (a) reaches beyond the catalogued families, and the order in
which its steps are built; internal representation of model specifications and steps; how the composer
matches and adapts steps; numerics for the full-RSB Parisi PDE;
implementation order within the v1 catalog; names of everything below the
package name.

## Calls settled after the builder check **[call]**

- **Interface.** Python library plus CLI, like the other `tools/` packages,
  driven by an LLM agent. Every run writes a machine-readable result and a
  rendered step trace; the trace is a deliverable, because the steps are
  recorded anyway and it is how a reader follows an entry to the paper.
- **Reference digits.** "Stated digits" means the digits the source states.
  Where sources disagree (e.g. SK ground-state energy across papers), both
  are recorded with citations and the tolerance covers the spread. Papers
  are fetched from the web during the build and cited; they are not stored
  in the repo.
- **Full RSB (Parisi PDE).** Stretch, not a gate for v1. It sits behind a
  flag with its own slower battery; the default battery (about 20 s) covers
  RS and 1RSB entries.
- **Refusal as an end state.** If stability fails and the catalog has no
  higher ansatz for that model, refusing is correct output. The held-out
  success case must be a model for which a valid ansatz exists.
- **`baller`.** Used for the banked reference values where roots and
  integrals can be enclosed; optional elsewhere. `replicant` stays a
  working name until approval.

## Not committed

This document is not committed; no product code, scaffolding, or
dependencies exist yet.
