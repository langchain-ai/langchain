# Request coalescing implementation ledger

- [x] Preflight: reviewed the task contract, pinned base, and planning workflow.
- [x] System map review: `Runnable` owns the public execution APIs; wrappers delegate
  schemas and graphs; exports are lazy in `runnables/__init__.py`.
- [x] Feature plan review: add a backend abstraction and `Runnable.with_coalesce`,
  covering invoke, streaming, batch, callbacks, stats, cancellation, and exports.
- [x] Implementation: added the coalescing backend/wrapper and focused tests.
- [x] Review: fixed sync/async shared-flight behavior and batch delegation.
- [x] Validation: focused runnable and import tests pass.
- [ ] Changelog and plan-diff validation (not available for this standalone pinned
  source checkout; no PR number exists).

## Reviewed implementation plan

1. Define a thread-safe backend with one shared flight table, sync and async wait
   paths, error/result propagation, statistics, and clear cancellation.
2. Add a serializable-compatible `RunnableCoalesce` wrapper and expose it through
   `Runnable.with_coalesce`; preserve schemas, graph delegation, transform behavior,
   and callback semantics.
3. Coalesce invoke/stream/batch APIs using an input-only canonical key, replay stream
   chunks, preserve batch ordering, and keep duplicate batch-as-completed results
   adjacent.
4. Add public exports and focused behavioral tests.

## Notes

The repository-wide system-map generator could not read the pinned checkout because
it attempts UTF-8 decoding on tracked binary fixtures. The generated process
instructions were reviewed manually and the affected Runnable/runnables test files
were inspected directly.
