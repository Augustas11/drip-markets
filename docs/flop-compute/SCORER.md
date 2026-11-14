# FLOP mark-rule scorer

Measures the OpenRouter mark rule in [SPEC.md](SPEC.md). It changes nothing
live. Code: `flop-index/` (the rule itself, shared with the pusher, and the
`flop-score` binary).

## One command, one number

    flop-index/pins/fetch.sh 2026-11-11 <dir>        # once: download the pin
    cargo run --release --manifest-path flop-index/Cargo.toml --bin flop-score -- \
      --data <dir> --rule <name> --split tune|heldout [--seed N]

The first line is the number: `gain <g> se <s> n <days>`. Gain is the
baseline's loss minus the candidate's on the same days, so positive means the
candidate is better. Then each part alone. Exit 0 means it ran; any other
exit prints `DID NOT RUN` and why. `flop-score list` names the rules.

**Gate** ([SPEC.md](SPEC.md), "Changing the mark rule"): a candidate is
recommended only if, on `--split heldout`, gain > 2 x se, and an
independent review does not veto it.

## What it charges

Each day, in log units (0.01 is about 1%):

- **jump**: how much each mark move differs from the move of the median
  listed price, between consecutive snapshots.
- **attack**: one anchor a day, four injected scenarios, each lasting 24 h
  and measured for 24 h after: a spoofed listing at ~1% of the median
  (0.5-2%) under a real provider name from elsewhere in the data, one
  provider's promo at ~0.1x (0.05-0.2x), half the providers down, one
  provider relisting at ~10x (5-20x). Loss is the mean |ln(attacked mark /
  clean mark)|. Victims and strengths are drawn from the seed.
- **follow**: a genuine move, every provider about 1.25-3x up or down
  (from the seed; each provider within +-10% of it, moving at its own time
  in the first 6 h); loss is the mean distance from where the mark should
  be (the clean mark times the move), from 6 h to 48 h, beyond the +-10% band (a rule that follows at once
  pays 0). It weighs as much
  as the four attacks together: a mark that doesn't move is not an index. Added 2026-11-11
  after the first round of candidates: a mark capped near 2.4%/day passed the gate (+0.213)
  by ignoring everything; with follow it scores clearly negative. The
  range reaches 3x because real cuts do (a counter-example showed a 24 h time
  filter lagging a 2.5x cut by a day).
- **reference**: distance from the leave-one-provider-out median.

Day loss = jump + mean of the four attacks + follow + reference, each averaged over
the models scored that day. Standard error is over days, Newey-West
corrected (attack windows span two days and rules carry state). A snapshot
is scored when it has 2+ up providers. A rule that
holds (returns no price) keeps its last mark; no mark at all costs 1.0.

The report also prints `ex-ref`: jump + mean attack + follow, paired per
day with its own standard error. It is advisory; the gate still uses the
total. The reference part is a median, so it favours median-like rules by
construction; a gain that lives only there is suspect.

**Held out:** the last 30% of scored days (`--split heldout`). Tune on
`--split tune` only. Rules replay the whole history, so the last tune day's
attack window and the first held-out jump read one day across the line.

**Daily-data caveat:** on daily pins an attack covers one snapshot, so a
time filter or a multi-day smoother clears most of the attack term almost
for free. A pass by such a rule needs confirming on recorder data, where an
attack covers 288 snapshots. The reverse holds for follow: on daily data only the
+24 h snapshot is charged, so lagging one snapshot pays the whole move; on
5-minute data it pays about 1/500 of it.

## Data

A pin is a folder with a `MANIFEST` of `sha256 loader file`. The scorer reads
only listed files and stops if any hash differs. `flop-score pin <dir>
loader:file ...` makes one and never overwrites it.

Pin 2026-11-11 (`flop-index/pins/2026-11-11/`, files not committed):

| File | Source | Used rows |
|---|---|---|
| romanshumy-price-history.csv | romanshumy/llm-prices-data @3517cbd, MIT | daily price per model per provider |
| sophie-history.csv | sophie-nguyenthuthuy/llm-pricing-history @9711a88, licence unclear: analysed, not vendored | LiteLLM per-provider price changes, replayed daily |

219 model series (one per model per source) with 2+ providers, 14,348 daily snapshots, 2024-02 to 2026-11-10
(daily from late June). Neither source has quantization or uptime: every
endpoint is "unknown" and up, so the bf16/fp16 set order is not exercised.

Spot-check, 2026-11-11, 10 pinned rows against OpenRouter's live listing: 3
match (GMI 0.45/3.2; OpenRouter's model price for qwen3.8-27b and
qwen3.6-27b), 5 differ (DeepInfra pinned 0.40/3.00, listed 0.15/1.875; also
Nebius, StreamLake, Cloudflare gateway, Alibaba), 2 are not on OpenRouter
(Groq, WandB). The LiteLLM rows lag the listings. So this data measures the
rule on realistic endpoint sets, not on OpenRouter's own history.

Recorder data (`recorder-jsonl`, 5-minute OpenRouter snapshots with
quantization and status) replaces it as it accumulates: pin a dated copy of
the day files and score the same way.

**Recording a pin.** Recorder data has no public source and pins of it are
not committed (size, and third-party listings). Record and pin your own:

    staging record-or launch/flop-compute.toml <rec dir>   # one file per UTC day
    staging record-or-status <rec dir>                     # gaps over 15 minutes
    mkdir <pin dir> && cp <rec dir>/2026-11-1*.jsonl <pin dir>/
    flop-score pin <pin dir> recorder-jsonl:2026-11-11.jsonl ...

The MANIFEST's sha256 names the data in every report. Tests use small
synthetic fixtures, not pins.

## Results, pin 2026-11-11, held out (2026-10-13 to 11-10, 29 days)

Scored against SPEC as it stood on 2026-11-11, which clamped sets of 2. Since 2026-11-12 the clamp needs 3 endpoints (SPEC, cheap-outlier clamp), so these numbers are for the old baseline until the scorer is rerun.

| Rule | gain | se | Note |
|---|---|---|---|
| spec (baseline) | 0.000 | 0.000 | itself |
| mean (no clamp, no sets) | +0.024 | 0.004 | better overall; worse on 10x relisting and jumps |
| spec-median (SPEC's `or_weighting = median`) | +0.116 | 0.003 | worse on relisting |

Seed 2 gives the same to 0.001. The baseline's own losses: jump 0.003,
spoof 0.20, promo 0.36, down 0.03, relist 0.34, reference 0.11.

So the brief's last check, "a plain mean scores clearly worse", does not
hold overall: the mean is worse only on relisting and jumps. The scorer
was not reshaped to make it hold.

The relist loss came from the clamp as it then stood. With two endpoints, one
relisting at 10x lifted the median, the half-median floor rose past the other
endpoint, and that cheap endpoint was clamped up: on SPEC's bf16 pair
(DeepInfra $0.2578, GMICloud $0.6219), GMICloud at 10x moved the mark from
$0.311 to $1.911. Since 2026-11-12 a pair is not clamped and the same
relisting gives $0.268. Test:
`the_live_bf16_pair_relisting_at_10x_no_longer_lifts_the_mark`.

## Hold candidates, recorder data so far (2026-11-13, not a gate result)

`hold-2h`, `hold-6h`, `hold-12h`: SPEC with a missing endpoint held at its
last up price (SPEC.md, "Missing endpoints"). On daily pins they equal `spec`
(snapshots are 24 h apart). On the recorder's first 539 snapshots
(2026-11-11 11:08 to 11-13, one model, `--split all`, seed 1):

| Rule | gain | se | down | jump |
|---|---|---|---|---|
| hold-2h | +0.005 | 0.007 | +0.035 | -0.0005 |
| hold-6h | +0.346 | 0.215 | +0.178 | -0.0007 |
| hold-12h | +0.366 | 0.228 | +0.188 | -0.0007 |

All with the 1 h confirm. The jump part is slightly worse for every length;
not yet explained. The "down" scenario downs endpoints at their current
price, so these gains say nothing about a provider that lists a new price
and then goes down: add that scenario before choosing.

Three days, so no rule clears 2 x se and there is no held-out split yet.
Rerun on a pin of the first full week (complete 2026-11-18) before choosing.

## Adding a candidate

Write a struct implementing `Rule` in `flop-index/src/score/candidates.rs`
and add one line to `ALL`. A rule sees every endpoint at each snapshot (down
ones too) and may keep state between snapshots (time filters, smoothing).

## Check v3

A second check for settlement rules, on 5-minute recorder data. It does not
change the default scorer above (v1) or `FLOP_SCORE_V2=1`. Run it with
`--check v3` (or `FLOP_SCORE_V3=1`):

    flop-score --check v3 --data <recorder pin> --rule <name> [--base spec] \
      [--split tune|heldout|all] [--seeds 1..8] [--k 6]

**Data.** A pin of recorder files: `recorder-jsonl` (one model, the older
recorder; uptime is read from the `raw` response by tag) or `wide-jsonl`
(the multi-model recorder, one folder per model, `<org>__<model>/<day>.jsonl`;
MANIFEST paths may now name a sub-folder, never `..`). Every model that ever
lists 5+ distinct providers is one series: rules are model-agnostic, they get
the time and that model's quotes. A rule may read each quote's `uptime_30m`
by implementing `Rule::step_v3`; rules that only implement `step` work as
before.

**Split.** By model, never by time: a model is held out if its FNV-1a hash is
odd, and the benchmark `qwen/qwen3.8-27b` always is. On the 62-model pin of
2026-11-13/14 that is 20 held-out series and 21 tune series.

**Level reference, `routed`** ("what a request pays now"): over sellers that
are up, one price per provider (its lowest 15:1 blend), weight
uptime_30m / price² (a null uptime weighs 0), no provider above 50% of the
weight (the excess spread over the others in proportion), price =
Σ(w·p) / Σw. None if fewer than two providers have weight. It is also a
baseline rule, `routed`.

**Terms** (log units; each printed for the rule and the base, with the paired
gain and its standard error):

- **phantom**: mark movement with no new listed price behind it, per day. A
  step where no row shows a new price (a seller going down or up, uptime
  moving, a row leaving and coming back at its old price) counts in full; on
  a step that also has a new price, the part the availability change made
  (the rule replayed on the previous step's availability with this step's
  prices). A hold expiring moves the mark on unchanged input, so it is
  phantom too; `stale` shows that part alone.
- **reversal**: the part of each move undone within the next K scored steps
  (default 6, 30 minutes), per day.
- **follow**: real consensus moves. Consensus is the mean log price change of
  every seller listed throughout (any precision, any status), so no rule
  defines it. An event is a step where it moves 1%+, at least half of that
  is still there 6 steps later, and it is new against 6 steps before (a
  promo ending is not an event). Score per event: 1 − |1 − mark move /
  consensus move| (0 below zero), read 6 steps later. `follow_broad` is the
  same on events where 2+ sellers moved (advisory).
- **follow_inj**: every seller moves x1.1 to x2 (seeded, each within ±3%, each
  in the first 15 minutes); scored from 30 minutes after the last move to 3 h.
- **level**: |ln(mark / routed)| per scored step; the mean is the term, the
  median is gated.
- **attack**: one anchor per 3 h per series (seeded offset, 3+ up sellers).
  For k = 1, 2, 3 colluders, each in "any" precision and in bf16: fake sellers
  at 0.5x and at 0.1x the cheapest real price with uptime 100 (6 h); real
  sellers at 0.5x for 30 minutes (promo); real sellers relisting at 10x (6
  h); "newlow-offline": a real seller lists 0.5x for one step and then goes
  down with that price still listed (6 h; a hold would keep the fake price).
  Also half the sellers down for 1 h (counted as k = 3). A seller that goes
  down loses uptime over 30 minutes. Each attack is measured until 1 h after
  it ends; its number is the largest |ln(attacked / clean mark)|. Lines give
  the max and mean per attack, k and precision.
- **distinct**: distinct marks per series (median), the benchmark's, and steps
  where the rule returned no price.

The total is phantom + reversal + level + (1 − follow) + (1 − follow_inj) +
mean attack, per series averaged over the seeds; the standard error is over
series (over seeds if there is one series).

**Gates** (constants at the top of `flop-index/src/score/v3.rs`, printed as
`GATE <name> PASS|FAIL <value> (limit <x>)`): worst attack move for k = 1 ≤
0.02, k = 2 ≤ 0.05, k = 3 ≤ 0.10; phantom ≤ 20% of all movement; follow ≥
0.5; follow_inj ≥ 0.5; median level ≤ 0.05 (a rule that targets another level
must say so and argue it); on `--split heldout`, gain over `spec` > 2 x se.

**Baselines on the 2026-11-13/14 pins** (heldout, seeds 1..8): `spec` fails
every attack gate (one bf16 fake at 0.5x the cheapest moves it up to x4.9,
x8.8 on the benchmark alone), phantom (87% of its movement), follow (0.26)
and level (0.069); 2 distinct marks on the benchmark in 3.5 days. `wide`
passes the gain gate (+1.54, se 0.50) and fails the attacks (k = 2: x10),
phantom (43%), follow (0.04) and level (0.28). `routed` passes level by
construction and fails the rest (one fake at 0.1x: x6.9). No baseline passes.

## Check v4

The current check for settlement rules:

    flop-score --check v4 --data <recorder pin> --rule <name> \
      [--split tune|heldout|all] [--seeds 1..8] \
      [--baselines spec,routed,routed-tm60,all-median,wide|none] [--threads N]

Data, split and the `routed` definition are v3's. Gates are constants at the
top of `flop-index/src/score/v4.rs`.

**Why v3's `follow` went.** v3 asked for two things at once: no single seller
may move the mark by more than 2%, and the mark must follow "real consensus
moves". In this data almost every real move is one seller repricing
(PrimeIntellect's dynamic price moves 8-70% a step; `routed` swings about 49%
high-low a day on the benchmark). So the two gates contradict each other.
Rules built to pass v3 hit this, and the ones that passed the attack gates
were nearly dead: the best moved under 1% a day on the benchmark and did not
move at all on half the models. A perp whose price does not move is a
failure even with every gate green. v3's attacks were also too narrow (six
fixed scenarios, 6 hours) and its `ok()` hid a NaN mark as a held one.

**The idea.** A seller's influence is earned and bounded, not zero. The check
sorts attackers by what they have to do:

- **U, unproven listings.** Anything with less than 24 h of observed serving
  time (status up and uptime_30m > 0), or not serving at all (down, null or
  zero uptime). Costless to make, so they may move the mark by at most 1%,
  for any count up to 8, over the whole remaining data. Behaviours:
  constant at 0.1x/0.5x/0.95x the cheapest or
  1.05x/10x the dearest (live, live in bf16, idle, null uptime); sleepers that
  list at the mark for 2 or 6 h and then jump or walk; walks of ±1% a step;
  status flicker; the vanish-and-reappear ratchet (gone, x0.9, x1.1,
  repeating, idle or live); x0.5/x2 alternation; extreme rows (1e-160 once,
  1e-160, 1e160, 0, negative, NaN with NaN uptime); one row listed 4 times;
  8 quantizations with the cheapest cycling. A rule may also never return a
  non-finite mark, and never hold because of the fakes while 2+ real sellers
  are up (gate `U_finite`; v3 treated both as a held mark). A live fake that
  has served 24 h is a proven seller from then on, so its U window ends
  there; on the 21 h pin that never happens.
- **S, a spike by proven sellers.** One or two established sellers change
  price by x0.1 to x10 for 1 or 6 snapshots (5 or 30 minutes) and revert.
  Worst move ≤ 0.12 (log) for k = 1 and 2, measured to the end of the data.
  The same spike for 18 snapshots (90 minutes) is printed as `ADVISORY
  S90_k1` / `S90_k2`, not gated. The limit was first designed as 2% (k = 1)
  and 5% (k = 2) and is set looser, at 0.12: a 30-minute spike by a seller
  that carries a large share of buyer spend does change what buyers pay,
  and a seller's own price taken as the median of its last 13 prices still
  shows a 6-snapshot spike when that seller reprices every snapshot. A
  30-minute spike can be bounded, not hidden.
- **P, a sustained move by proven sellers.** k proven sellers (k = 1, 2, 3
  and the largest k under 60% of them) hold x0.5, x0.8, x1.25, x2 or x10 to
  the end. That is partly a real market change, so the mark may move, but by
  no more than their share: worst move ≤ 0.25 x k x |ln f| (one seller
  halving for good: at most about 17%; the 0.25 share holds at every
  factor). And not fast: in the first 35 minutes no more than the S limit
  (k ≥ 3: 10%). The window was first designed as 20 minutes and is set to
  35, the length of the 13-sample median: a held move and a spike look the
  same until such a median has seen the new price 7 times, 30-35 minutes. Each factor's worst move is printed (`P
  k=.. x..`).
- **M, a market move.** 60% of proven sellers (by count, a seeded random
  set; and also all of them) move by the same factor, x1.1, x1/1.1, x1.3 or
  x1/1.3, each at its own time in the first 15 minutes. The mark must show
  80% of it from 60 minutes on (read to 3 h; score per step 1 − |1 − mark
  move / move|, floored at 0). Gated on the worst (set, factor) mean. This
  replaces v3's follow_inj.
- **Echo**. The market moves x0.5,
  x1/1.3 or x1.3 and stays there: every provider up at the anchor at once
  (`all`), or the first 60% of them by name, the j-th of k from step
  3j/k (`60%`, 15 minutes of stagger). From 3 hours after the move (after
  the stagger) to 48 hours after it, the largest 5-minute step beyond the
  clean run's step must be ≤ 0.02 (`echo_step`: two 1% steps in opposite
  directions), and |ln(mark / clean mark) − share × ln f| ≤ 0.10
  (`echo_error`, share 1 or 0.6). A rule whose market detector reads its
  own credited move as a new move the other way steps back and forth here.
  One case is printed, not gated, though first designed as gated (`ADVISORY
  echo_known_exception`): a halving by 60% of the sellers is credited to
  the other 40% as a market move, the credit fades with a 12 h time
  constant while they keep their price, and the mark gives part of the
  move back (for `clamp3`, 0.27 on the benchmark). The 40% are still listed
  at their old price, so drifting back toward them is defensible. The same moves by M's random 60% set
  at M's random times are printed (`E 60%-random`, `ADVISORY
  echo_random_set`), not gated.
- **Coalition probe** (advisory). The k dearest proven sellers
  (k = 10, 11, 12) halve their price for good; the worst move is printed as
  `ADVISORY coalition k=.. of n` (n: proven sellers up at the anchor, 20
  to 22 on the benchmark). It shows where a coalition stops being bounded
  by the weight cap and becomes a market move: for `clamp3`, 10 of 20 move
  the mark 0.18, 11 of 22 move it 0.71.
- **F, freeze.** The same move by every proven seller, an hour after 1 to 8
  unproven listings appeared at the mark and sat there (idle or live). Still
  80%, gated on the worst (k, mode) mean. A rule that lets idle listings vote
  "no move" fails it.

Attacks start at one seeded anchor per seed per series (seeds 1..8), at
least an hour after the first snapshot, and run to the end of the data (on
pin-rec up to 2.8 days); U prints the worst move at 6 h, 24 h and the end so
slow drift shows.

**Proven, in the check.** pin-wide holds 21 hours, so 24 h of serving cannot
be observed there: every seller listed in a series' first snapshot is proven
from the start, and a seller that appears later becomes proven after 24 h
of serving (at most 10 minutes credited per snapshot). U fakes appear after
the first snapshot. A rule may use the same convention: sellers seen at its
first step are proven.

**Liveliness**, on the clean data (no attack):

- **L1**, daily high-low range of the mark (max/min − 1 over 24 h windows
  from the first snapshot with 12 h of data; the whole series if none). On
  pin-wide the median over models must be ≥ `spec`'s median; on pin-rec the
  benchmark's must be ≥ `spec`'s and ≥ 2%.
- **L2**, correlation of the mark's 1-hour log moves with R60's (each series
  averaged per hour first, so 5-minute flicker inside an hour is not a move)
  ≥ 0.5: median over models listing 8+ providers on pin-wide, the benchmark
  on pin-rec.
- **L3**, models whose mark never moved, among models where routed moved
  more than 2% high-low: at most 20%.

**Level.** Median |ln(mark / R60)| ≤ 5%, where R60 is the time-median of
`routed` over the last 60 minutes (13 snapshots; 150 s slack for recorder
jitter). The target is a smoothed "what a request pays", not the
instantaneous one; the instantaneous distance is an advisory line.

**Kept from v3.** Phantom (≤ 20% of movement from availability-only changes)
and `stale` (advisory). Gain vs `spec` is advisory only (it passed rules
that were bad on other terms).

**Baselines.** `spec`, `routed`, `routed-tm60` (R60 as a rule; a yardstick,
not a candidate: it trusts any fake with uptime), `all-median`, `wide`. Each
run prints every baseline's LIVE and GATE lines (`BASE <name> ...`).

**Output.** `CHECK v4 ...`, `LIVE ...` (percent), detail lines (`U k=`,
`UB <behaviour>`, `S`, `P`, `M`, `F`, `E`), `ADVISORY ...` (S90, the echo
exception, the coalition probe among them), then one `GATE <name>
PASS|FAIL <value> (limit <x>) <where the worst case was>` per gate: U,
U_finite, S_k1, S_k2, P (worst move/bound, limit 1), P_fast (worst move in 35
min / limit, limit 1), M, F, echo_step, echo_error, L1_range (percent),
L2_corr, L3_dead (percent), level_r60, phantom_share. Deterministic (threads over series and seeds;
results land by index). A run that cannot happen exits 2.

**Baselines on the 2026-11-13/14 pins** (heldout, seeds 1..8): no baseline
passes U, S, P or M. `spec` fails U (one bf16 fake at 0.1x the cheapest moves it x27, x32 on
the benchmark),
L2 (0.24; 0.01 on the benchmark), level (0.076) and phantom; it passes L1
by definition (the bar is its own range, 13.9% a day median, 2.5% on the
benchmark). `routed` passes level (0.005) and L1-L3 (26% a day; L2 0.69,
0.58 on the benchmark) and fails every attacker. `routed-tm60` passes level,
L1-L3 on heldout (on tune its 10.6% range is under spec's 12.8%), fails U
(fakes with uptime are trusted) and S (0.39: a median of 13 snapshots moves
a whole rank when routed is mid-way through a real level change, as on
glm-5.2 at +6 h). No real multi-seller move (60%+ of 3+
sellers) occurs in either pin, so `follow_real_market` is empty.
