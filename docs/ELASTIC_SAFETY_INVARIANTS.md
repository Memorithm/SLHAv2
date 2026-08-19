# Elastic Safety Invariants

Hard invariants of the Elastic runtime. These are enforced by tests in
`elastic-core`, `elastic-runtime` and `elastic-testkit`; new Elastic code
must keep them.

## ElasticContext invariants

1. **PINNED content is not evicted by normal pressure.** The tier machine
   has no `Pinned => Evicted` edge; `Pinned => !Evicted` is a
   compile-time-validated negative edge.
2. **Committed/resident memory never exceeds a hard budget after a
   successful adaptation cycle** (except documented unavoidable
   atomic-transition peak memory).
3. **A failed transition cannot lose the only valid representation.**
   `run_transaction` never destroys the old state before commit; prepare and
   validate failures leave the old state intact.
4. **Counters match actual residency.** Reported bytes correspond to
   physically allocated/resident bytes (never logical accounting presented
   as physical).
5. **No tile exists simultaneously in mutually exclusive ownership states**
   unless it is an explicit transition replica.
6. **State restoration validates representation before promotion.**
7. **The model positional limit is never silently exceeded.** True model
   limits are `ModelLimit` hard constraints, not `ElasticTarget`s.
8. **Different sequences cannot read each other's KV** (isolation is a
   backend contract).
9. **Concurrent transitions cannot expose partial state** (transactional
   commit is the only publication point).
10. **Deterministic mode is reproducible:** same observations/budgets/
    history/policy/config ⇒ same decisions; the core never reads wall-clock
    time.

## ECA invariants

- **Hard constraints have priority over utility.** Critical pressure
  (`used > capacity`) overrides every soft objective.
- **No oscillation:** HIGH/LOW watermarks (0.85/0.70 default), cooldown and
  minimum-interval guards prevent compress/decompress thrash around a noisy
  threshold.
- **Forecast is bounded:** the trend derivative is clamped, so a single
  spike cannot produce an absurd forecast or a bogus exhaustion prediction.
- **Rollback failures are hard errors** (`TransitionOutcome::RollbackFailed`),
  never silently swallowed.

## Budget invariants

- A child commit fails closed (`BudgetError`) when the parent's hard limit
  would be exceeded (`ParentExhausted`).
- A commit never partially applies.
- Reservations protect bytes from borrowing; priority guards prevent a
  lower-priority borrower from displacing a higher-priority tenant.
- `release` is saturating; counters can never go negative.

## Transactional peak memory

A transition HOT(128 B) → WARM(96 B) requires up to `128 + 96 + workspace`
bytes until commit. The planner must account for transition peak memory,
especially under OOM pressure; if there is no room for a safe transition,
the controller returns a controlled no-feasible-action rather than a
corrupting partial transition.

## Security / governance

- An Elastic controller is a **resource authority**. Untrusted model text
  must never set memory quotas, pinned status, tenant limits or hard safety
  thresholds.
- Policy mutation is governed (Enterprise RBAC); observation and
  recommendation are separate from authorized mutation.
- No cross-tenant telemetry leak; journals record resource ids and reason
  codes, never tenant payloads.
