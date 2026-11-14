//! Settlement rule `clamp3` (proposed, off by default; docs/flop-compute/SPEC.md,
//! "Settlement rule clamp3"). It includes three guards: no market move
//! against the last one within 25 steps (echo), an explicit allow-list
//! instead of "whoever is listed at the first step", and a warm start from the
//! recorder's files.
//!
//! The rule, step by step (one step per 5-minute fetch; the step-count
//! constants below assume that spacing):
//! 1. Only proven sellers count: one on `proven_at_start` with a valid up row
//!    at the first step (names match exactly, after trimming whitespace), or
//!    one with 24 h of serving (status up and uptime_30m > 0; at most 10
//!    minutes credited per step). Rows with a non-finite, non-positive or
//!    absurd price are dropped first. A seller that is not proven and has
//!    had no valid up row for 24 h is forgotten (its state is dropped).
//! 2. A seller's price is the median of its own last 13 up prices (its
//!    lowest 15:1 blend), kept across absences.
//! 3. Market move: when 60% of the proven sellers up now (half, but at least
//!    5, in a big market) moved the same way over the last 2 hours, the
//!    smallest of those moves (at most 0.1 a step) is credited to every
//!    proven seller that has not moved that far. Sellers that reprice almost
//!    every step vote "no move". A move against the last credited one is not
//!    credited within ECHO_BLOCK_STEPS of it. Credits fade (12 h).
//! 4. Each seller's price plus credit is clamped to [mark / 3, mark x 3]
//!    around the last mark (moved by this step's market move).
//! 5. The mark is the routed formula over sellers listed up in most of the
//!    last hour: weight (median uptime) / price^2, the price for the weight
//!    being the higher of its price and its 1-hour average, no seller above
//!    24%; weight no one takes sits on a price that moves only with market
//!    moves (blended in log).
//! 6. The mark moves at most 1% (log) a step plus the market move credited in
//!    that step. It is None until some proven seller counts, and None again
//!    once no proven seller has counted for STALE_STEPS steps (15 minutes):
//!    an old mark is never passed off as a fresh one.
//!
//! Pure except `warm_start`, which reads the recorder's files.

use crate::OrEndpoint;
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

const MIN_MS: u64 = 60_000;
const HOUR_MS: u64 = 60 * MIN_MS;

/// The step the rule assumes: one fetch every 5 minutes (seconds).
pub const STEP_S: u64 = 300;

/// Band: a seller's price counts inside [mark / BAND_C, mark x BAND_C].
pub const BAND_C: f64 = 3.0;
/// Speed limit: the mark moves at most this (log) per step, plus the market
/// move credited in that step.
pub const SPEED: f64 = 0.01;
/// Serving time after which a seller counts.
pub const T_PROVEN_MS: u64 = 24 * HOUR_MS;
/// Serving time credited per step at most (a gap in the data is not service).
pub const MAX_CREDIT_MS: u64 = 10 * MIN_MS;
/// A seller's price: the median of its last MED_N up prices. It is weighed
/// once it has MIN_OBS of them and votes on market moves with VOTE_OBS.
pub const MED_N: usize = 13;
pub const MIN_OBS: usize = 13;
pub const VOTE_OBS: usize = 7;
/// Availability window (steps): a seller counts while listed up in most of
/// it; its weight factor is the median of its uptime_30m / 100 over it.
pub const AVAIL_N: usize = 13;
/// Largest weight of one seller.
pub const CAP: f64 = 0.24;
/// Market move: look-back (steps), smallest and largest credited move per
/// step (log), and the fewest movers once 60% is relaxed to half.
pub const WIN: usize = 24;
pub const FIRE_MIN: f64 = 0.002;
pub const FIRE_MAX: f64 = 0.1;
pub const QUORUM_MIN: usize = 5;
/// A seller whose median price change over VOL_N steps exceeds this (log)
/// votes "no move".
pub const VOTE_VOL: f64 = 0.001;
pub const VOL_N: usize = 36;
/// Market credit fades with this time constant (ms).
pub const CREDIT_TAU_MS: f64 = 12.0 * HOUR_MS as f64;
/// A seller's weight uses the higher of its price and a slow average of it
/// (this time constant, ms): a cut gains weight slowly, a rise loses it at once.
pub const REF_TAU_MS: f64 = HOUR_MS as f64;
/// Usable 15:1 blends, USD per 1M tokens; anything outside is no price.
pub const MIN_PRICE: f64 = 1e-6;
pub const MAX_PRICE: f64 = 1e6;
/// Echo guard: no market move in the opposite direction within this many
/// steps of the last credited one. Without it, the detector read its own
/// credited move as a new move the other way once the look-back window
/// rolled past it, and the mark stepped back.
pub const ECHO_BLOCK_STEPS: usize = WIN + 1;
/// With no proven seller counted for this many steps in a row (15 minutes),
/// the rule gives no mark until one counts again.
pub const STALE_STEPS: usize = 3;
/// A seller that is not proven and has had no valid up row for this long is
/// dropped from the state, so listings under many names cannot grow it
/// without bound. Proven sellers are never dropped.
pub const FORGET_MS: u64 = 24 * HOUR_MS;

/// Who counts from the first step.
#[derive(Clone, Debug, PartialEq)]
pub enum ProvenAtStart {
    /// The live rule: these providers, if they have a valid up row at the
    /// first step. Empty: nobody; every seller earns T_PROVEN_MS.
    List(BTreeSet<String>),
    /// Scorer only (flop-score's `clamp3`): every provider with a valid up
    /// row at the first step, the check's own convention (SCORER.md, "Proven,
    /// in the check"), since a 21-hour pin cannot show 24 h of serving.
    /// Never the live rule: a fake listed when the pusher starts would count.
    UpAtFirstStep,
}

#[derive(Clone, Debug, Default)]
struct Seller {
    proven: bool,
    /// When it last had a valid up row.
    seen_ms: u64,
    tenure_ms: u64,
    was_serving: bool,
    /// Last up prices (log), at most MED_N; their median; last |changes|.
    obs: VecDeque<f64>,
    z: Option<f64>,
    dx: VecDeque<f64>,
    /// Last AVAIL_N steps: uptime / 100 while up (else 0), and up (1/0).
    avail: VecDeque<f64>,
    upq: VecDeque<f64>,
    /// Market credit (log).
    credit: f64,
    /// z + credit - market over the last WIN + 1 steps (while proven).
    hist: VecDeque<f64>,
    /// Weight factor this step (0: not in the mark).
    wf: f64,
    /// Slow average of the counted price (log): its weight's price.
    wref: Option<f64>,
    /// The price it counts at (log), after the band.
    cq: Option<f64>,
}

/// The rule's state. `step` once per fetch, oldest first.
#[derive(Clone, Debug)]
pub struct Clamp3 {
    proven_at_start: ProvenAtStart,
    sellers: BTreeMap<String, Seller>,
    last_at: Option<u64>,
    steps: usize,
    /// Sum of credited market moves (log).
    market: f64,
    /// Price of the weight no seller takes, relative to `market` (log).
    ghost: Option<f64>,
    last_mark: Option<f64>,
    /// The last credited market move: (step, move).
    last_fire: Option<(usize, f64)>,
    /// Steps in a row with no proven seller counted.
    uncounted: usize,
    /// After the first step: the allow-list names that matched a provider
    /// with a valid up row then, and those that did not.
    allow_report: Option<(Vec<String>, Vec<String>)>,
}

fn push_cap(v: &mut VecDeque<f64>, x: f64, n: usize) {
    v.push_back(x);
    while v.len() > n {
        v.pop_front();
    }
}

fn median_of(v: &VecDeque<f64>) -> Option<f64> {
    let mut s: Vec<f64> = v.iter().copied().collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(f64::total_cmp);
    let n = s.len();
    Some(if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) })
}

/// USD per 1M tokens at the SPEC's 15:1 input:output blend.
fn blend(e: &OrEndpoint) -> f64 {
    let b = crate::SPEC.blend;
    (b * e.prompt + e.completion) / (b + 1.0) * 1e6
}

impl Clamp3 {
    pub fn new(proven_at_start: ProvenAtStart) -> Self {
        Clamp3 { proven_at_start, sellers: BTreeMap::new(), last_at: None, steps: 0, market: 0.0, ghost: None, last_mark: None, last_fire: None, uncounted: 0, allow_report: None }
    }

    /// The live rule with `proven_at_start` as the allow-list. Names match
    /// provider names exactly (case and all) after trimming whitespace.
    pub fn with_allow_list<I: IntoIterator<Item = String>>(proven_at_start: I) -> Self {
        Self::new(ProvenAtStart::List(proven_at_start.into_iter().map(|s| s.trim().to_string()).collect()))
    }

    /// The mark the last step gave (USD per 1M tokens), if any.
    pub fn last_mark(&self) -> Option<f64> {
        self.last_mark.filter(|_| self.uncounted < STALE_STEPS)
    }

    /// After the first step: (allow-list names that matched a provider with
    /// a valid up row at that step, names that did not). None before it.
    pub fn allow_list_report(&self) -> Option<&(Vec<String>, Vec<String>)> {
        self.allow_report.as_ref()
    }

    /// When the last step ran.
    pub fn last_at(&self) -> Option<u64> {
        self.last_at
    }

    /// One fetch: every endpoint listed at `at_ms` (down ones too) and each
    /// one's uptime_30m (percent, parallel to `endpoints`; None = not
    /// reported). Returns the mark in USD per 1M tokens, or None to hold.
    /// Never returns a non-finite or non-positive mark.
    pub fn step(&mut self, at_ms: u64, endpoints: &[OrEndpoint], uptime: &[Option<f64>]) -> Option<f64> {
        let first = self.last_at.is_none();
        let dt = self.last_at.map(|t| at_ms.saturating_sub(t)).unwrap_or(0);
        self.last_at = Some(at_ms);
        self.steps += 1;
        let min_obs = MIN_OBS.min(self.steps);
        let vote_obs = VOTE_OBS.min(self.steps);
        // Per provider: lowest valid up price (log) and that row's uptime.
        let mut best = BTreeMap::<&str, (f64, f64)>::new();
        for (i, e) in endpoints.iter().enumerate() {
            if e.status != 0 || !(e.prompt.is_finite() && e.completion.is_finite() && e.prompt >= 0.0 && e.completion >= 0.0) {
                continue;
            }
            let b = blend(e);
            if !(b.is_finite() && (MIN_PRICE..=MAX_PRICE).contains(&b)) {
                continue;
            }
            let x = b.ln();
            let u = uptime.get(i).copied().flatten().filter(|u| u.is_finite()).unwrap_or(0.0).clamp(0.0, 100.0) / 100.0;
            if best.get(e.provider.as_str()).is_none_or(|(y, _)| x < *y) {
                best.insert(e.provider.as_str(), (x, u));
            }
        }
        // Grandfathered: only providers with a valid up row now.
        if first {
            for pr in best.keys() {
                let ok = match &self.proven_at_start {
                    ProvenAtStart::List(l) => l.contains(*pr),
                    ProvenAtStart::UpAtFirstStep => true,
                };
                if ok {
                    self.sellers.entry(pr.to_string()).or_default().proven = true;
                }
            }
            if let ProvenAtStart::List(l) = &self.proven_at_start {
                let (hit, miss) = l.iter().cloned().partition(|n| best.contains_key(n.as_str()));
                self.allow_report = Some((hit, miss));
            }
        }
        for pr in best.keys() {
            self.sellers.entry(pr.to_string()).or_default().seen_ms = at_ms;
        }
        // Forget sellers that are not proven and long gone.
        self.sellers.retain(|_, s| s.proven || at_ms.saturating_sub(s.seen_ms) < FORGET_MS);
        // Tenure, own prices, availability, credits.
        let decay = (-(dt as f64) / CREDIT_TAU_MS).exp();
        for (pr, s) in self.sellers.iter_mut() {
            if s.was_serving {
                s.tenure_ms = s.tenure_ms.saturating_add(dt.min(MAX_CREDIT_MS));
            }
            let now = best.get(pr.as_str()).copied();
            s.was_serving = now.is_some_and(|(_, u)| u > 0.0);
            if s.tenure_ms >= T_PROVEN_MS {
                s.proven = true;
            }
            s.credit *= decay;
            push_cap(&mut s.avail, now.map(|(_, u)| u).unwrap_or(0.0), AVAIL_N);
            push_cap(&mut s.upq, if now.is_some() { 1.0 } else { 0.0 }, AVAIL_N);
            if let Some((x, _)) = now {
                if let Some(&prev) = s.obs.back() {
                    push_cap(&mut s.dx, (x - prev).abs(), VOL_N);
                }
                push_cap(&mut s.obs, x, MED_N);
                let z = median_of(&s.obs);
                if let (Some(old), Some(new)) = (s.z, z) {
                    // A move the same way as its credit uses the credit up.
                    let d = new - old;
                    if d * s.credit > 0.0 {
                        s.credit -= d.abs().min(s.credit.abs()) * s.credit.signum();
                    }
                }
                s.z = z;
            }
            let counted = s.proven && s.obs.len() >= min_obs && median_of(&s.upq).unwrap_or(0.0) > 0.5;
            s.wf = if counted { median_of(&s.avail).unwrap_or(0.0) } else { 0.0 };
        }
        // Market move over the last WIN steps, voted by proven sellers up now.
        let market0 = self.market;
        let mut moves = Vec::new();
        for (pr, s) in self.sellers.iter_mut() {
            if !(s.proven && s.obs.len() >= vote_obs) {
                s.hist.clear();
                continue;
            }
            let u = s.z.unwrap_or(0.0) + s.credit - market0;
            push_cap(&mut s.hist, u, WIN + 1);
            if best.contains_key(pr.as_str()) {
                let steady = median_of(&s.dx).unwrap_or(0.0) <= VOTE_VOL;
                moves.push(if steady { u - s.hist[0] } else { 0.0 });
            }
        }
        let n = moves.len();
        let mut fired = 0.0;
        if n >= 2 {
            moves.sort_by(f64::total_cmp);
            // 60% of the voters; in a big market half, but at least QUORUM_MIN.
            let need = (3 * n).div_ceil(5).min(n.div_ceil(2).max(QUORUM_MIN));
            let i = n - need;
            let (lo, hi) = (moves[i], moves[n - 1 - i]);
            let c = if lo > 0.0 { lo } else if hi < 0.0 { hi } else { 0.0 };
            let echo = self.last_fire.is_some_and(|(at, last)| last * c < 0.0 && self.steps <= at + ECHO_BLOCK_STEPS);
            if c.is_finite() && c.abs() >= FIRE_MIN && !echo {
                let c = c.clamp(-FIRE_MAX, FIRE_MAX);
                fired = c;
                self.market += c;
                self.last_fire = Some((self.steps, c));
                for s in self.sellers.values_mut() {
                    if let (Some(&last), Some(&h0)) = (s.hist.back(), s.hist.front()) {
                        let e = last - h0;
                        s.credit += if c > 0.0 { (c - e).clamp(0.0, c) } else { (c - e).clamp(c, 0.0) };
                        if let Some(b) = s.hist.back_mut() {
                            *b = s.z.unwrap_or(0.0) + s.credit - self.market;
                        }
                    }
                }
            }
        }
        // Band around the last mark (moved by this step's market move), on
        // the price and on its weight's price.
        let lm = self.last_mark.map(|m| m.ln() + fired);
        let band = BAND_C.ln();
        let a_ref = 1.0 - (-(dt as f64) / REF_TAU_MS).exp();
        for s in self.sellers.values_mut() {
            if s.wf > 0.0 {
                if let Some(z) = s.z {
                    let q = z + s.credit;
                    s.cq = Some(match lm {
                        Some(l) => q.clamp(l - band, l + band),
                        None => q,
                    });
                }
            }
            s.wref = match (s.wf > 0.0, s.cq, s.wref) {
                (true, Some(q), Some(r)) => Some(r + (q - r) * a_ref),
                (true, Some(q), None) => Some(q),
                _ => None,
            };
        }
        let m = self.mark(lm);
        self.uncounted = if m.is_some() { 0 } else { self.uncounted + 1 };
        self.last_mark()
    }

    /// The mark from the counted sellers (None if none counts); stores it.
    fn mark(&mut self, lm: Option<f64>) -> Option<f64> {
        // Routed formula at the counted prices, capped weights, ghost.
        let qs: Vec<(f64, f64)> = self.sellers.values().filter(|s| s.wf > 0.0).filter_map(|s| Some((s.cq?, s.wf * (-2.0 * s.wref?.max(s.cq?)).exp()))).collect();
        if qs.is_empty() {
            return None;
        }
        let q0 = qs[0].0;
        let r: Vec<f64> = qs.iter().map(|(_, a)| *a).collect();
        let rs: f64 = r.iter().sum();
        if !(rs.is_finite() && rs > 0.0) {
            return None;
        }
        let w: Vec<f64> = r.iter().map(|x| (x / rs).min(CAP)).collect();
        let ghost = match self.ghost {
            Some(g) => g,
            None => qs.iter().zip(&r).map(|((q, _), r)| r / rs * (q - q0).exp()).sum::<f64>().ln() + q0 - self.market,
        };
        // Sellers' routed mean, blended (in log) with the ghost by the weight
        // the sellers did not take.
        let sw = w.iter().sum::<f64>().min(1.0);
        let block = qs.iter().zip(&w).map(|((q, _), w)| w / sw * (q - q0).exp()).sum::<f64>().ln() + q0;
        let raw = sw * block + (1.0 - sw) * (ghost + self.market);
        // Speed limit around the last mark plus this step's market move.
        let x = match lm {
            Some(l) => l + (raw - l).clamp(-SPEED, SPEED),
            None => raw,
        };
        let m = x.exp();
        if !(m.is_finite() && m > 0.0) {
            return None;
        }
        self.ghost = Some(ghost);
        self.last_mark = Some(m);
        Some(m)
    }
}

/// One snapshot of the benchmark model from the recorder: when, every row,
/// and each row's uptime_30m.
pub type Snapshot = (u64, Vec<OrEndpoint>, Vec<Option<f64>>);

/// The snapshots of `model` in recorder files' text (`staging record-or`,
/// one JSON line per fetch), oldest first. A line that does not parse (a
/// torn write) is skipped and counted.
pub fn recorder_snapshots(texts: &[String], model: &str) -> (Vec<Snapshot>, usize) {
    let mut groups = crate::score::v3::Groups::new();
    let mut skipped = 0;
    for t in texts {
        for line in t.lines().filter(|l| !l.trim().is_empty()) {
            if crate::score::v3::parse_jsonl(line, &mut groups).is_err() {
                skipped += 1;
            }
        }
    }
    let snaps = groups
        .remove(model)
        .unwrap_or_default()
        .into_iter()
        .map(|(at, rows)| {
            let (eps, upt) = rows.into_iter().unzip();
            (at, eps, upt)
        })
        .collect();
    (snaps, skipped)
}

/// Replays `snaps` taken before `before_ms` through `rule`, oldest first;
/// returns how many ran.
pub fn replay(rule: &mut Clamp3, snaps: &[Snapshot], before_ms: u64) -> usize {
    let mut n = 0;
    for (at, eps, upt) in snaps.iter().filter(|s| s.0 < before_ms) {
        rule.step(*at, eps, upt);
        n += 1;
    }
    n
}

/// Warm start: replays every `*.jsonl` recorder file in `dir` (by name, so
/// by day) for `model`, snapshots before `before_ms` only. The live steps
/// then continue the replay, exactly so only while the pusher fetches what
/// the recorder saved, at the same times. Returns (snapshots replayed, lines
/// skipped).
pub fn warm_start(rule: &mut Clamp3, dir: &Path, model: &str, before_ms: u64) -> Result<(usize, usize)> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("warm start: read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    let texts = files.iter().map(|f| std::fs::read_to_string(f).with_context(|| format!("warm start: read {}", f.display()))).collect::<Result<Vec<_>>>()?;
    let (snaps, skipped) = recorder_snapshots(&texts, model);
    Ok((replay(rule, &snaps, before_ms), skipped))
}

#[cfg(test)]
mod tests;
