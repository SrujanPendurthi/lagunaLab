# Design note: GPU-side expert routing (removing the per-layer sync)

Status: **stage 1 measured, fails the quality gate; no-go at 64 slots** (see
"Stage 1 result" at the end). Written 2026-10-15.
Measured numbers come from `results/stream_profile.json` and are named
inline. Estimates are labelled as estimates.

## The problem

Every streamed MoE layer, at every decode step, does this
(`laguna_mlx/streaming.py`, `StreamingSwitchGLU.__call__`):

1. `np.array(indices)`: block until the GPU has run everything so far (this
   layer's attention and router), then copy the 8 chosen expert ids to the host.
2. `ExpertStore.load`: map experts to slots, `pread` the misses from SSD, and
   write them into slot tensors.
3. Run the quantized `SwitchGLU` on slot indices, then build the next layer's
   graph, up to its sync.

That's 39 host round-trips per token. While the host works, the GPU is idle.

What it costs, measured (`sweep_capped`, `async_ab`, 32k context, 1 GB pool cap):

| | Step ms |
|---|---|
| Resident pruned model (`resident`) | 46.8 |
| Streaming, 128 slots (every expert present, zero loads) | 68.9 |
| Streaming, 64 slots (`async_ab`, async off) | 68.7 |
| Streaming, 64 slots (`sweep_capped`) | 72.2 |

The 22 ms gap at 128 slots is the round-trips alone: host bookkeeping is
0.38 ms per token (measured, CPU only), and the rest is GPU idle time and Metal
commit latency. That's about 30% of the 64-slot step. Already ruled out:
`mx.async_eval` (1.5× slower), batched slot writes (slower), `mx.compile` (the
KV cache offsets retrace every token), and prefetch (break-even in moeStream).

## The hard constraint: an exact miss needs the host

The SSD read must start on the host, and the host can't know what to read
until the router has run. So **there is no exact, sync-free way to handle a
miss.** Every exact variant checked brings the sync back, or does worse:

- **Sync only when a miss happens.** Learning that a miss happened is itself
  the sync. It costs the same as today, because the cost is the stall, not the
  8 integers.
- **Run optimistically, check once per token, redo on a miss.** The KV caches
  could be rolled back with the trimmable cache from speculative decoding. But
  at 64 slots a token has ~21.6 misses across 312 expert picks (`sweep_capped`,
  32k). The chance a whole token has no miss is about 0.93^312, which is
  effectively zero, so almost every token would run twice.
- **Let the GPU read expert pages from an mmap of the pack.** That would need
  MLX to wrap file-backed memory as a Metal buffer, plus GPU page faulting on
  non-resident pages. Neither is available in pinned MLX 0.32.0 from Python.
  Out of scope: no MLX upgrades, no C++.

## The proposal: drop on miss (lossy, opt-in)

Sync once per token instead of once per layer, by accepting an approximation
on misses:

1. **Device-side table.** Each layer keeps `slot_of: int32[E]` on the GPU:
   the slot of each expert, or −1 if it isn't resident. In the graph, slot
   indices = `slot_of[indices]` and present = `slot_of[indices] >= 0`.
   Missing picks point at slot 0 and are masked out. Shapes are fixed:
   [1,1,8] for decode at any miss count.
2. **A missed expert contributes zero.** The remaining gate weights are
   renormalized over the experts that are present. This needs `SparseMoeBlock`
   to take the mask: an additive change to `laguna_mlx.laguna`, default off,
   and the stable interface is unchanged.
3. **The miss list reaches the host once per token.** Decode already syncs
   once per token for the sampled id. Read the miss mask in the same eval, load
   those experts on the 8-thread pool, and write them into slots before the
   next token's graph is built. Writes queue between tokens and never race
   kernels in flight. The LFU bookkeeping is the same.
4. **Prefill keeps today's exact sync path.** A 2,048-token chunk touches most
   experts, so dropping there would be large and it isn't the bottleneck.
   Drop-on-miss is decode only.

Interface: `load_streaming(pack, slots, cache_limit_gb=None, miss_policy="load")`.
`"load"` is today's exact behaviour and stays the default; `"drop"` is the new
mode. moeStream's imports don't change.

### Memory cost

The table is 128 experts × 4 bytes × 39 layers ≈ 20 KB, and the miss mask is
about the same. Nothing else is new: slots, KV and the pool are as today. At
64 slots, 82k peak should stay about 10.16 GB (measured for the current path,
`kv.json` → `streaming-64slots-cap1.0gb`).

### Expected gain (estimate)

- **Upper bound:** the step approaches the resident model with the same KV,
  minus nothing. That is ~47 ms at 32k against 68.7–72.2 today, about 1.5×
  (13.9 to ~21 tok/s at 32k). It also removes the remaining load time from
  the critical path (4.7 ms/step at 64 slots, `async_ab`), since reads happen
  between tokens.
- **Realistic: about 52–58 ms/step at 32k, 1.2–1.3×.** Between tokens the host
  still does the reads and slot writes, and the step grows by that wall time
  unless it overlaps the next token's early layers. The gather on the device
  table and the mask also add a little GPU work. This hasn't been measured; the
  first A/B decides.

### Quality cost (the real risk)

At 64 slots and 32k, about 7% of expert picks would be dropped: 21.6 misses
out of 312 picks (`sweep_capped`). The model is already pruned to 50% of its
experts (KL 0.081 against the full model, `prune_point.json`). Whether another
7% of dropped picks on top of that stays within the Part 3 gate (top-1 ≥ 0.98,
KL ≤ 0.02 against the pruned model) is **unknown**, and it is the go/no-go
question.

Factors:

- **It helps** that misses skew toward rarely used experts, since LFU keeps the
  hot ones.
- **It hurts** that a missed expert can be the top-1 pick.
- **More slots reduce the drop rate** (16→64 slots cut misses 165→21.6 at 32k)
  at a cost in memory.

## Risks

- **Exactness is lost.** Today streaming is exact (top-1 1.0, KL 0), which is
  a selling point. Drop mode must stay opt-in and be reported with its own
  replay quality numbers.
- **Behaviour depends on the cache.** Output depends on which experts happen to
  be resident, so it depends on history and the slot count, not only on the
  input. That makes results harder to reproduce and debug. The replay metric
  still measures it fine.
- **Complexity.** It adds a table kept in step with the host's slot map, writes
  between tokens, a mask through `SparseMoeBlock`, and two code paths (prefill
  exact, decode drop). That's roughly 150 lines plus tests, in the module most
  coupled to moeStream.
- **Graph shapes are not a risk.** Decode shapes are fixed: the table lookup is
  a gather on [1,1,8], and the mask is elementwise.
- **Interaction with moeStream:**
  - It imports `load_streaming`. The new keyword is additive, and the default
    keeps today's exact behaviour.
  - moeStream's own engine (prefetch, tiers) has its own store; it would need
    its own port of the idea.
  - Prefetch becomes complementary: a correct prefetch guess now turns a
    dropped pick into a real one, a quality win, not just a latency win.

## Would moeStream benefit?

Probably, and possibly more. Its engine has the same per-layer host sync,
reported at 53 ms/token. That's moeStream's own number, not measured here.
The same drop-on-miss design removes it, with the same quality question. Its
router-lookahead prefetch (~0.70 accuracy, by its report) would cut the drop
rate, which is the main risk here. So the design fits moeStream's engine at
least as well as this loader. Both projects should share the quality
measurement before either builds the full thing.

## What to test first

### Off-GPU, tiny models (`tests/tiny.py`)

1. With `miss_policy="drop"` and `slots == num_experts` (nothing can miss),
   decode output is bit-exact against `"load"`.
2. With forced misses, output equals a hand-built reference: missing experts
   zeroed and gates renormalized over the present ones.
3. Count `np.array` calls during decode: exactly one per token, not 39.
4. A missed expert becomes resident before the next token, and slot writes
   never land mid-token.

### The cheap first real-weight step: measure quality before building anything

Simulate drop semantics inside today's sync path. At the sync, the host
already knows the misses, so it zeros them, renormalizes, and loads them for
later tokens. That's ~20 lines, with no sync removal and no async machinery.
Then run `replay` on the longest django session at 64 slots against `refs/kv`:
KL and top-1 under drop. If it fails the gate, stop; nothing else gets built.

## GPU time

| Step | Run | Time (estimate) |
|---|---|---|
| 1 | Quality of simulated drop, 64 slots, longest session (82,671 tokens), replay | ~40 min (the 64-slot `bench_kv` run took 35 min) |
| 1b | Same on all 5 django sessions, if 1 passes | ~1.5–2 h |
| 2 | Speed A/B, real sync-free path against today, 64 slots, 256 and 32k (`profile_stream`-style, same process, fresh models) | ~15 min |
| 3 | 82k memory check of the final configuration (`bench_kv --target streaming`) | ~40 min |

## Recommendation

**Conditional go, staged:**

1. **Go now: the cheap quality simulation** (about 1 h of work plus ~40 min
   GPU). It answers the only question that can kill the idea.
2. **Build the sync-free path only if** simulated drop at 64 slots passes the
   Part 3 gate (top-1 ≥ 0.98, KL ≤ 0.02 against pruned) on all 5 sessions.
   Effort: about half a day of focused work (~150 lines plus tests), then
   ~1 h of GPU.
3. **Keep `miss_policy="load"`, the exact mode, as the default** either way.
   The current 16 GB answer (64 slots, cap, exact, 10.16 GB, 6.87 tok/s at 82k)
   stands. Drop mode would be a faster, approximate option on top of it.

**No-go if exact output is a requirement** for the 16 GB target. Without it,
the sync floor stays. The remaining exact lever is more slots, which costs
memory: 128 slots is 68.9 ms against 72.2 at 64.

## Stage 1 result (2026-10-15): fails the gate

Source: `results/drop_sim.json` (`scripts/sim_drop.py`). Setup:

- pruned-50 pack, 64 slots, 1 GB pool cap;
- the 5 django sessions, each processed exactly up to 32k tokens;
- 8 decode windows of 128 tokens per session, teacher-forced token by token,
  each snapped forward to an assistant span;
- every window run with drop on, then trimmed off the cache and run exactly.

KL is full-vocab KL(exact || drop) on the assistant target tokens.

| Session | Assistant tokens | KL mean | KL p99 | Top-1 | Drop rate |
|---|---|---|---|---|---|
| 10914 | 473 | 0.182 | 4.16 | 0.930 | 0.145 |
| 10924 | 862 | 0.177 | 2.59 | 0.910 | 0.149 |
| 11001 | 561 | 0.119 | 2.72 | 0.936 | 0.128 |
| 11019 | 680 | 0.168 | 2.65 | 0.943 | 0.160 |
| 11039 | 793 | 0.092 | 1.25 | 0.966 | 0.119 |
| **All (token-weighted)** | 3,369 | **0.146** | 4.16 (max) | **0.937** | 0.140 (mean) |

The gate was KL ≤ 0.02 and top-1 ≥ 0.98. KL is 7× over it, and top-1 misses
by 4 points.

- **The drop rate is twice the estimate:** 14%, not 7%. The 7% came from the
  steady-state decode misses in `sweep_capped`, measured after 32 decode steps.
  These windows start right after a prefill chunk, when the slot cache holds
  what the prompt used, not what generation uses. A real agent alternates
  between the two every turn, so this is the realistic rate.
- **Per layer,** the drop rate is 11–25%. It's highest in the first two MoE
  layers (25%, 22%).
- **Verdict: no-go for drop-on-miss at 64 slots.** Building the sync-free path
  (stage 2) isn't justified. More slots lower the drop rate, but 128 slots
  holds every expert, where the sync floor also comes with 12.2 GB. The current
  exact answer (64 slots, cap, 10.16 GB, 6.87 tok/s at 82k) stands.
- **Still open, the user's call:**
  - a drop variant that keeps the top-1 pick exact: sync only when the top-1
    expert misses. That needs its sync per layer, so it saves nothing unless
    the check moves on device. Not pursued;
  - **removing the simulation code** (`StreamingSwitchGLU.drop_misses`, off,
    and `scripts/sim_drop.py`). It's kept for now so this result can be
    reproduced.
