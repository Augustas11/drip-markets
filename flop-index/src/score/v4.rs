//! Check v4 (docs/flop-compute/SCORER.md, "Check v4").
//!
//! v3 asked for a mark that no single seller can move AND
//! that follows "real consensus moves", when almost every real move in the
//! data is one seller repricing. Rules tuned to v3 went dead. v4 says instead
//! that a seller's influence is earned and bounded, and tests it by what an
//! attacker must actually do:
//!
//! - U: unproven listings (under T_PROVEN of observed serving time, or not
//!   serving at all) at any price, count, behaviour: the mark may not move
//!   more than GATE_U, ever, and may never go non-finite or hold because of
//!   them;
//! - S: a proven seller spikes (0.1x to 10x) for up to 30 minutes and reverts
//!   (90 minutes reported, not gated);
//! - P: k proven sellers move for good: the mark may move, by at most their
//!   share (SHARE_CAP x k x |log move|), and not within P_DELAY_MS;
//! - M: 60%+ of proven sellers move by the same factor: the mark follows
//!   GATE_FOLLOW of it within FOLLOW_WITHIN_MS;
//! - F: the same with up to 8 idle unproven listings present;
//! - echo: after a market-wide move the mark may not step back and forth
//!   (GATE_ECHO_STEP) or drift from the move (GATE_ECHO_ERROR) over 48 h;
//! - advisory: a 90-minute spike (S90) and a coalition of the k dearest
//!   sellers halving (k = 10, 11, 12);
//! - liveliness on the clean data (L1 daily range, L2 correlation with the
//!   smoothed routed price, L3 share of dead marks), level against R60 (the
//!   60-minute time-median of routed), phantom.
//!
//! Proven, for the check: a provider listed in the series' first snapshot
//! (pin-wide is 21 h, so tenure cannot be observed), or one with T_PROVEN of
//! serving time (status up and uptime_30m > 0) before the anchor. Rules may
//! use the same convention (sellers seen at their first step are proven).
//! Attacks start after the first snapshot and run to the end of the data.

use super::data::DAY_MS;
use super::rules::{blend, Rule, RoutedTm60};
use super::v3::{self, Series3, Snap};
use super::{fnv, splitmix, Split};
use crate::OrEndpoint;
use anyhow::{anyhow, bail, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const MIN: u64 = 60_000;
const HOUR: u64 = 60 * MIN;
const STEP: u64 = 5 * MIN;

// ---- Gates. ----
//
// Three limits are looser than first designed: GATE_S (0.12, was 0.02 for
// k = 1 and 0.05 for k = 2), P_DELAY_MS (35 minutes, was 20) and the
// 60%-share x0.5 echo error (reported, not gated). The reasons are at each
// constant.

/// A seller is proven after this much observed serving time.
pub const T_PROVEN_MS: u64 = 24 * HOUR;
/// U: largest move unproven listings may cause, any k <= 8, whole data.
pub const GATE_U: f64 = 0.01;
/// S: largest move from k = 1, 2 proven sellers spiking up to 30 minutes
/// (S_GATED_STEPS snapshots). First designed as 0.02 (k = 1) and 0.05
/// (k = 2); set to 0.12 for both because a 30-minute spike by one seller
/// that carries a large share of buyer spend does change what buyers pay,
/// so the mark may show part of it. A seller's price taken as the median of
/// its last 13 up prices still shows a 6-snapshot spike by a seller that
/// reprices every snapshot: such a spike can be bounded, not hidden. The
/// 90-minute spike (18 snapshots) is reported, not gated ("S90").
pub const GATE_S: [f64; 2] = [0.12, 0.12];
/// P: k proven sellers holding a move of |ln f| may move the mark at most
/// SHARE_CAP x k x |ln f|...
pub const SHARE_CAP: f64 = 0.25;
/// ...and within the first P_DELAY_MS no more than GATE_S[k] (k = 1, 2) or
/// GATE_P_FAST_K3 (k >= 3). First designed as 20 minutes; set to 35, the
/// length of the 13-sample median: a held move and a spike look the same
/// until such a median has seen the new price 7 times (30-35 minutes), the
/// shortest time in which a rule can tell them apart.
pub const P_DELAY_MS: u64 = 35 * MIN;
pub const GATE_P_FAST_K3: f64 = 0.10;
/// M and F: a move by this share of proven sellers (by count) is the market.
pub const M_SHARE: f64 = 0.6;
/// M and F: the mark shows this share of it from FOLLOW_WITHIN_MS on.
pub const GATE_FOLLOW: f64 = 0.8;
pub const FOLLOW_WITHIN_MS: u64 = HOUR;
/// L1: the benchmark's daily range at least this (and at least spec's).
pub const GATE_L1_BENCH: f64 = 0.02;
/// L2: correlation of hourly log moves with R60 at least this.
pub const GATE_L2: f64 = 0.5;
/// L3: at most this share of live models (routed range > L3_ROUTED_MIN) with
/// a mark that never moved.
pub const GATE_L3: f64 = 0.20;
/// Median |ln(mark / R60)|.
pub const GATE_LEVEL: f64 = 0.05;
/// Phantom movement as a share of all movement.
pub const GATE_PHANTOM_SHARE: f64 = 0.20;
/// Echo: after a market-wide move (ECHO_FACTORS, every
/// proven seller or M_SHARE of them) the mark may not step back and forth.
/// From ECHO_SETTLE_MS after the move to ECHO_WINDOW_MS after it, the
/// largest 5-minute step beyond the clean run's step is at most this (two
/// 1% steps in opposite directions)...
pub const GATE_ECHO_STEP: f64 = 0.02;
/// ...and |ln(attacked / clean) - share x ln f| at most this.
pub const GATE_ECHO_ERROR: f64 = 0.10;
/// Reported, not gated: GATE_ECHO_ERROR does not apply to these (factor,
/// share of sellers), first designed as gated. A halving by 60% of the
/// sellers is credited to the other 40% as a market move; that credit fades
/// (12 h time constant) while they keep their price, so the mark gives back
/// part of the move. It is reported rather than gated because the 40% who
/// kept their price are still listed at it, so a mark drifting back toward
/// them over hours is defensible (printed as `ADVISORY
/// echo_known_exception`).
pub const ECHO_KNOWN_EXCEPTIONS: [(f64, &str); 1] = [(0.5, "60%")];

// ---- Parameters. ----

/// R60: the time window of the level reference (and `routed-tm60`).
pub const R60_MS: u64 = HOUR;
/// Recorder jitter: snapshots land 299-301 s apart, so the window reaches
/// this much further back to always hold 13 of them.
pub const R60_SLACK_MS: u64 = 150_000;
/// Colluding unproven listings.
pub const U_K: [usize; 5] = [1, 2, 3, 5, 8];
/// U horizons reported (the gate is on the whole remaining data).
pub const U_HORIZONS_MS: [u64; 2] = [6 * HOUR, 24 * HOUR];
pub const S_FACTORS: [f64; 6] = [0.1, 0.5, 0.8, 1.25, 2.0, 10.0];
/// S spike lengths in snapshots: 1 (5 min) and 6 (30 min; counted in
/// snapshots, since the recorder's 299-301 s cadence fits 7 in 30 minutes);
/// 18 (90 min) is reported only (S90).
pub const S_STEPS: [usize; 3] = [1, 6, 18];
/// The S gate covers spikes up to this many snapshots (30 minutes).
pub const S_GATED_STEPS: usize = 6;
pub const P_FACTORS: [f64; 5] = [0.5, 0.8, 1.25, 2.0, 10.0];
pub const M_FACTORS: [f64; 4] = [1.1, 1.0 / 1.1, 1.3, 1.0 / 1.3];
pub const F_FACTORS: [f64; 2] = [1.3, 1.0 / 1.3];
/// M and F movers each move at their own seeded time in the first 15 min.
pub const M_STAGGER_MS: u64 = 15 * MIN;
/// M and F are read from FOLLOW_WITHIN_MS to this long after the move starts.
pub const FOLLOW_READ_END_MS: u64 = 3 * HOUR;
/// F: the idle listings appear this long before the market moves.
pub const F_LEAD_MS: u64 = HOUR;
/// L2 counts models listing this many distinct providers.
pub const L2_MIN_PROVIDERS: usize = 8;
pub const L3_ROUTED_MIN: f64 = 0.02;
/// L1 windows: 24 h from the first snapshot; one counts with 12 h of data.
pub const L1_MIN_COVER_MS: u64 = 12 * HOUR;
/// Serving time credited per snapshot at most (a gap is not service).
pub const MAX_CREDIT_MS: u64 = 10 * MIN;
/// Attack anchors per seed per series...
pub const ANCHORS_PER_SEED: usize = 1;
/// ...no earlier than this after the first snapshot (a rule gets an hour of
/// history; the live rule never starts cold).
pub const WARMUP_MS: u64 = HOUR;
/// Real market events (advisory): read this many scored steps later.
pub const MARKET_N: usize = 12;
pub const MARKET_MIN: f64 = 0.01;
/// Echo: factors of the market-wide move; the read starts this long after
/// the move (after the M_STAGGER_MS stagger for the 60% case) and ends this
/// long after it.
pub const ECHO_FACTORS: [f64; 3] = [0.5, 1.0 / 1.3, 1.3];
/// Who moves: every provider up at the anchor; the first M_SHARE of them by
/// name, staggered evenly over ECHO_STAGGER_STEPS (both
/// gated); M's random M_SHARE set at M's random times (printed, not gated).
pub const ECHO_SETS: [&str; 3] = ["all", "60%", "60%-random"];
pub const ECHO_STAGGER_STEPS: usize = 3;
pub const ECHO_SETTLE_MS: u64 = 3 * HOUR;
pub const ECHO_WINDOW_MS: u64 = 48 * HOUR;
/// Coalition probe (advisory): the k dearest proven sellers halve their
/// price for good (k = 10, 11, 12: around half of the benchmark's 23).
pub const COAL_K: [usize; 3] = [10, 11, 12];
pub const COAL_F: f64 = 0.5;
/// Baselines reported next to the rule.
pub const BASELINES: [&str; 5] = ["spec", "routed", "routed-tm60", "all-median", "wide"];

pub const LEVEL_DEF: &str = "R60 = median of routed over the snapshots of the last 60 min (+150 s for recorder jitter: 13 snapshots; mean of the two middle values if even); routed = over up sellers, one price per provider (its lowest 15:1 blend), weight uptime_30m/price^2, no provider above 50% of the weight";
pub const PROVEN_DEF: &str = "proven = listed in the series' first snapshot, or 24 h of serving (status up, uptime_30m > 0; at most 10 min credited per snapshot) before the anchor";

pub fn median(mut v: Vec<f64>) -> Option<f64> {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) })
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Status up and uptime_30m > 0: the row is serving.
pub fn serving(e: &OrEndpoint, u: Option<f64>) -> bool {
    v3::is_up(e) && u.is_some_and(|u| u.is_finite() && u > 0.0)
}

/// Proven providers at each index in `at` (PROVEN_DEF).
pub fn proven_at(s: &Series3, at: &BTreeSet<usize>) -> BTreeMap<usize, BTreeSet<String>> {
    let first: BTreeSet<String> = s.snaps.first().map(|sn| sn.eps.iter().map(|e| e.provider.clone()).collect()).unwrap_or_default();
    let mut acc: BTreeMap<&str, u64> = BTreeMap::new();
    let mut out = BTreeMap::new();
    for (i, sn) in s.snaps.iter().enumerate() {
        if at.contains(&i) {
            let mut p = first.clone();
            p.extend(acc.iter().filter(|(_, t)| **t >= T_PROVEN_MS).map(|(p, _)| p.to_string()));
            out.insert(i, p);
        }
        if let Some(nx) = s.snaps.get(i + 1) {
            let dt = nx.at_ms.saturating_sub(sn.at_ms).min(MAX_CREDIT_MS);
            let srv: BTreeSet<&str> = sn.eps.iter().zip(&sn.uptime).filter(|(e, u)| serving(e, **u)).map(|(e, _)| e.provider.as_str()).collect();
            for p in srv {
                *acc.entry(p).or_insert(0) += dt;
            }
        }
    }
    out
}

/// What a rule's step returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Out {
    Price(f64),
    Hold,
    /// Some(x) with x non-finite or <= 0: a broken mark (v3 hid it as a hold).
    Invalid,
}

pub fn classify(m: Option<f64>) -> Out {
    match m {
        Some(x) if x.is_finite() && x > 0.0 => Out::Price(x),
        Some(_) => Out::Invalid,
        None => Out::Hold,
    }
}

fn step(r: &mut dyn Rule, last: &mut Option<f64>, at: u64, eps: &[OrEndpoint], upt: &[Option<f64>]) -> Out {
    let o = classify(r.step_v3(at, eps, upt));
    if let Out::Price(m) = o {
        *last = Some(m);
    }
    o
}

/// The clean replay.
pub struct Replay4 {
    /// Each step's effective mark (a hold or an invalid output keeps the last).
    pub marks: Vec<Option<f64>>,
    /// The rule returned a usable price at this step.
    pub priced: Vec<bool>,
    pub invalid: usize,
    pub holds: usize,
    /// As v3: the mark on counterfactual input (phantom).
    pub cf: Vec<Option<f64>>,
    pub saved: BTreeMap<usize, (Box<dyn Rule>, Option<f64>)>,
}

pub fn replay(proto: &dyn Rule, s: &Series3, save: &BTreeSet<usize>) -> Replay4 {
    let mut rule = proto.clone_box();
    let mut last = None;
    let n = s.snaps.len();
    let mut rp = Replay4 { marks: Vec::with_capacity(n), priced: Vec::with_capacity(n), invalid: 0, holds: 0, cf: vec![None; n], saved: BTreeMap::new() };
    let mut seen = v3::Seen::new();
    let (mut prev, mut changed) = (None, false);
    for (i, sn) in s.snaps.iter().enumerate() {
        if save.contains(&i) {
            rp.saved.insert(i, (rule.clone_box(), last));
        }
        let f = v3::fresh(&seen, sn);
        changed |= f.iter().any(|x| *x);
        let is_scored = v3::providers(&sn.eps) >= 2;
        let pre = match prev {
            Some(j) if is_scored && changed => Some((rule.clone_box(), last, j)),
            _ => None,
        };
        let o = step(rule.as_mut(), &mut last, sn.at_ms, &sn.eps, &sn.uptime);
        match o {
            Out::Price(_) => {}
            Out::Hold => rp.holds += 1,
            Out::Invalid => rp.invalid += 1,
        }
        rp.priced.push(matches!(o, Out::Price(_)));
        rp.marks.push(last);
        if let Some((mut r, pl, j)) = pre {
            let (eps, upt) = v3::counterfactual(&s.snaps[j], sn, &f);
            let mut l = pl;
            step(r.as_mut(), &mut l, sn.at_ms, &eps, &upt);
            rp.cf[i] = l;
        }
        v3::remember(&mut seen, sn);
        if is_scored {
            prev = Some(i);
            changed = false;
        }
    }
    rp
}

/// The level reference per snapshot: `routed-tm60` replayed (R60).
pub fn r60(s: &Series3) -> Vec<Option<f64>> {
    let mut r = RoutedTm60::default();
    s.snaps.iter().map(|sn| r.step_v3(sn.at_ms, &sn.eps, &sn.uptime)).collect()
}

/// L1: high-low range (max/min - 1) of the marks in 24 h windows from the
/// first snapshot, mean over windows with L1_MIN_COVER_MS of data; the whole
/// series if no window has that much.
pub fn daily_range(at: &[u64], marks: &[Option<f64>]) -> Option<f64> {
    let t0 = *at.first()?;
    let mut win: BTreeMap<u64, (u64, u64, f64, f64)> = BTreeMap::new();
    for (t, m) in at.iter().zip(marks) {
        let Some(m) = m else { continue };
        let w = win.entry((t - t0) / DAY_MS).or_insert((*t, *t, f64::MAX, f64::MIN));
        w.1 = *t;
        w.2 = w.2.min(*m);
        w.3 = w.3.max(*m);
    }
    let full: Vec<f64> = win.values().filter(|w| w.1 - w.0 >= L1_MIN_COVER_MS).map(|w| w.3 / w.2 - 1.0).collect();
    if !full.is_empty() {
        return Some(mean(&full));
    }
    let v: Vec<f64> = marks.iter().flatten().copied().collect();
    if v.is_empty() {
        return None;
    }
    Some(v.iter().copied().fold(f64::MIN, f64::max) / v.iter().copied().fold(f64::MAX, f64::min) - 1.0)
}

pub fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 3 {
        return None;
    }
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    Some(if saa <= 0.0 || sbb <= 0.0 { 0.0 } else { sab / (saa * sbb).sqrt() })
}

/// L2: correlation of the 1-hour log moves of `a` and `b`: each series is
/// averaged (mean log) over hourly buckets from the first snapshot, and the
/// moves are between consecutive buckets, so 5-minute noise (a seller
/// flickering inside the hour) does not count as a move. A series that never
/// moves correlates 0.
pub fn hourly_corr(at: &[u64], a: &[Option<f64>], b: &[Option<f64>]) -> Option<f64> {
    let t0 = *at.first()?;
    let mut buckets: BTreeMap<u64, (f64, usize, f64, usize)> = BTreeMap::new();
    for ((t, x), y) in at.iter().zip(a).zip(b) {
        let e = buckets.entry((t - t0) / HOUR).or_insert((0.0, 0, 0.0, 0));
        if let Some(x) = x {
            e.0 += x.ln();
            e.1 += 1;
        }
        if let Some(y) = y {
            e.2 += y.ln();
            e.3 += 1;
        }
    }
    let v: Vec<(u64, Option<f64>, Option<f64>)> = buckets.into_iter().map(|(h, e)| (h, (e.1 > 0).then(|| e.0 / e.1 as f64), (e.3 > 0).then(|| e.2 / e.3 as f64))).collect();
    let (mut xa, mut xb) = (Vec::new(), Vec::new());
    for w in v.windows(2) {
        if let (true, Some(a0), Some(a1), Some(b0), Some(b1)) = (w[1].0 == w[0].0 + 1, w[0].1, w[1].1, w[0].2, w[1].2) {
            xa.push(a1 - a0);
            xb.push(b1 - b0);
        }
    }
    pearson(&xa, &xb)
}

fn log_range(v: &[Option<f64>]) -> f64 {
    let x: Vec<f64> = v.iter().flatten().copied().collect();
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().copied().fold(f64::MIN, f64::max) / x.iter().copied().fold(f64::MAX, f64::min)).ln()
}

/// Real market events (advisory): a scored step from which, MARKET_N steps
/// later, M_SHARE+ of 3+ sellers listed throughout moved 1%+ the same way,
/// the mean move is 1%+ and new against MARKET_N steps before. Score: 1 -
/// |1 - mark move / mean move|, floored at 0. One event per MARKET_N steps.
pub fn market_events(s: &Series3, marks: &[Option<f64>]) -> Vec<f64> {
    let sc = v3::scored(s);
    let low: Vec<BTreeMap<&str, f64>> = s.snaps.iter().map(|sn| v3::lowest(&sn.eps)).collect();
    let mut out = Vec::new();
    let mut q = 1;
    while q + MARKET_N < sc.len() {
        let (b, j, e) = (sc[(q - 1).saturating_sub(MARKET_N)], sc[q - 1], sc[q + MARKET_N]);
        let common: Vec<&str> = low[j].keys().copied().filter(|p| low[e].contains_key(p) && low[b].contains_key(p)).collect();
        if common.len() >= 3 {
            let mv = |from: usize, p: &str| (low[e][p] / low[from][p]).ln();
            let cn = common.iter().map(|p| mv(j, p)).sum::<f64>() / common.len() as f64;
            let cb = common.iter().map(|p| mv(b, p)).sum::<f64>() / common.len() as f64;
            let movers = common.iter().filter(|p| mv(j, p) * cn.signum() >= MARKET_MIN).count();
            let first_step = common.iter().any(|p| low[sc[q]].get(p).is_some_and(|x| *x != low[j][*p]));
            if first_step && cn.abs() >= MARKET_MIN && movers as f64 >= M_SHARE * common.len() as f64 && cb * cn > 0.0 && cb.abs() >= 0.5 * cn.abs() {
                let score = match (marks[j], marks[e]) {
                    (Some(a), Some(b)) => (1.0 - (1.0 - (b / a).ln() / cn).abs()).max(0.0),
                    _ => 0.0,
                };
                out.push(score);
                q += MARKET_N;
                continue;
            }
        }
        q += 1;
    }
    out
}

/// One rule on one series, no attack.
#[derive(Clone, Debug, Default)]
pub struct CleanOut {
    pub model: String,
    pub providers_ever: usize,
    pub movement: f64,
    pub phantom: f64,
    pub stale: f64,
    pub days: f64,
    pub level_r60: Vec<f64>,
    pub level_now: Vec<f64>,
    pub range: Option<f64>,
    pub corr: Option<f64>,
    pub dead: bool,
    pub routed_live: bool,
    pub invalid: usize,
    pub holds: usize,
    pub steps: usize,
    pub market: Vec<f64>,
}

pub fn clean(proto: &dyn Rule, s: &Series3) -> CleanOut {
    let rp = replay(proto, s, &BTreeSet::new());
    let c3 = v3::clean_terms(&rp.marks, &rp.cf, rp.holds, s, v3::REVERSAL_K);
    let ref60 = r60(s);
    let at: Vec<u64> = s.snaps.iter().map(|sn| sn.at_ms).collect();
    let routed: Vec<Option<f64>> = s.snaps.iter().map(|sn| v3::routed(&sn.eps, &sn.uptime)).collect();
    let level_r60 = v3::scored(s).into_iter().filter(|&i| ref60[i].is_some()).map(|i| v3::dist(rp.marks[i], ref60[i])).collect();
    CleanOut {
        model: s.model.clone(),
        providers_ever: s.snaps.iter().flat_map(|sn| sn.eps.iter().map(|e| e.provider.as_str())).collect::<BTreeSet<_>>().len(),
        movement: c3.movement,
        phantom: c3.phantom,
        stale: c3.stale,
        days: c3.days,
        level_r60,
        level_now: c3.level,
        range: daily_range(&at, &rp.marks),
        corr: hourly_corr(&at, &rp.marks, &ref60),
        dead: log_range(&rp.marks) < 1e-9,
        routed_live: log_range(&routed).exp() - 1.0 > L3_ROUTED_MIN,
        invalid: rp.invalid,
        holds: rp.holds,
        steps: s.snaps.len(),
        market: market_events(s, &rp.marks),
    }
}

/// Attack anchors for a seed: ANCHORS_PER_SEED snapshots with 3+ up
/// providers, each the first at or after a seeded time from WARMUP_MS on.
pub fn anchors(s: &Series3, seed: u64) -> Vec<usize> {
    let (Some(first), Some(last)) = (s.snaps.first(), s.snaps.last()) else { return vec![] };
    let span = last.at_ms.saturating_sub(first.at_ms + WARMUP_MS);
    if span == 0 {
        return vec![];
    }
    let ok_at: Vec<usize> = (1..s.snaps.len()).filter(|&i| v3::providers(&s.snaps[i].eps) >= 3).collect();
    let mut out = BTreeSet::new();
    for j in 0..ANCHORS_PER_SEED {
        let t = first.at_ms + WARMUP_MS + splitmix(seed ^ splitmix(fnv(&s.model) ^ splitmix(j as u64 + 0x4004))) % span;
        if let Some(&i) = ok_at.iter().find(|&&i| s.snaps[i].at_ms >= t) {
            out.insert(i);
        }
    }
    out.into_iter().collect()
}

fn shuffle<T>(v: &mut [T], mut r: u64) -> u64 {
    for i in (1..v.len()).rev() {
        r = splitmix(r);
        v.swap(i, (r % (i as u64 + 1)) as usize);
    }
    r
}

// ---- U: unproven listings. ----

/// How a fake lists. Live: up, uptime 100 (a new seller saying it serves).
/// Idle: down, uptime 0. Null: up, uptime not reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Live,
    Idle,
    Null,
}

/// Price base: the cheapest up row, the dearest up row, or the clean mark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Base {
    Cheap,
    Dear,
}

/// One unproven-listing behaviour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ub {
    /// Constant at base x mult, the base row's quantization or bf16.
    Const { base: Base, mult: f64, mode: Mode, bf16: bool },
    /// Live at the mark for `quiet_ms`, then x`jump`, then walking `walk` per step.
    Sleeper { quiet_ms: u64, jump: f64, walk: f64 },
    /// Live from the mark, x(1+r) per step.
    Walk(f64),
    /// Live at the cheapest x0.5, status flipping down/up every step.
    FlickerStatus,
    /// The ratchet: gone one step, then mark x0.9, then x1.1
    /// (reverse when !up), repeating; a rule that counts the step back in and
    /// not the step out climbs.
    Ratchet { up: bool, mode: Mode },
    /// Live at mark x0.5 and x2 on alternate steps.
    Alternate,
    /// One row at 1e-160 USD/token for one step, down.
    Tiny1,
    /// 1e-160 USD/token, live, to the end.
    Tiny,
    /// 1e160 USD/token, live.
    Huge,
    /// Price 0, live.
    Zero,
    /// Negative prompt price, live.
    Neg,
    /// NaN prompt, infinite completion, NaN uptime.
    Nan,
    /// Live at the cheapest x0.5, the row listed 4 times.
    Dup,
    /// Live, 8 quantization rows around the cheapest x0.5, the cheapest cycling.
    ManyQ,
}

pub fn behaviours() -> Vec<Ub> {
    let mut v = Vec::new();
    for (base, mult) in [(Base::Cheap, 0.1), (Base::Cheap, 0.5), (Base::Cheap, 0.95), (Base::Dear, 1.05), (Base::Dear, 10.0)] {
        for (mode, bf16) in [(Mode::Live, false), (Mode::Live, true), (Mode::Idle, false), (Mode::Null, false)] {
            v.push(Ub::Const { base, mult, mode, bf16 });
        }
    }
    v.extend([
        Ub::Sleeper { quiet_ms: 2 * HOUR, jump: 0.5, walk: 0.0 },
        Ub::Sleeper { quiet_ms: 2 * HOUR, jump: 2.0, walk: 0.0 },
        Ub::Sleeper { quiet_ms: 2 * HOUR, jump: 1.0, walk: -0.01 },
        Ub::Sleeper { quiet_ms: 6 * HOUR, jump: 0.1, walk: 0.0 },
        Ub::Walk(0.01),
        Ub::Walk(-0.01),
        Ub::FlickerStatus,
        Ub::Ratchet { up: true, mode: Mode::Idle },
        Ub::Ratchet { up: false, mode: Mode::Idle },
        Ub::Ratchet { up: true, mode: Mode::Live },
        Ub::Ratchet { up: false, mode: Mode::Live },
        Ub::Alternate,
        Ub::Tiny1,
        Ub::Tiny,
        Ub::Huge,
        Ub::Zero,
        Ub::Neg,
        Ub::Nan,
        Ub::Dup,
        Ub::ManyQ,
    ]);
    v
}

fn mode_s(m: Mode) -> &'static str {
    match m {
        Mode::Live => "live",
        Mode::Idle => "idle",
        Mode::Null => "null-uptime",
    }
}

pub fn ub_name(b: &Ub) -> String {
    match b {
        Ub::Const { base, mult, mode, bf16 } => {
            format!("const-{}{mult}x-{}{}", if *base == Base::Cheap { "cheap" } else { "dear" }, mode_s(*mode), if *bf16 { "-bf16" } else { "" })
        }
        Ub::Sleeper { quiet_ms, jump, walk } => {
            if *walk != 0.0 {
                format!("sleeper-{}h-walk{walk:+}", quiet_ms / HOUR)
            } else {
                format!("sleeper-{}h-x{jump}", quiet_ms / HOUR)
            }
        }
        Ub::Walk(r) => format!("walk{r:+}"),
        Ub::FlickerStatus => "flicker-status".into(),
        Ub::Ratchet { up, mode } => format!("ratchet-{}-{}", if *up { "up" } else { "down" }, mode_s(*mode)),
        Ub::Alternate => "alternate-x0.5-x2".into(),
        Ub::Tiny1 => "extreme-1e-160-one-step".into(),
        Ub::Tiny => "extreme-1e-160".into(),
        Ub::Huge => "extreme-1e160".into(),
        Ub::Zero => "extreme-zero".into(),
        Ub::Neg => "extreme-negative".into(),
        Ub::Nan => "extreme-nan-inf".into(),
        Ub::Dup => "duplicate-rows-x4".into(),
        Ub::ManyQ => "many-quantizations-8".into(),
    }
}

/// What the fakes copy at the anchor.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub cheap: OrEndpoint,
    pub dear: OrEndpoint,
    pub mark: f64,
}

impl Ctx {
    pub fn at(sn: &Snap, mark: Option<f64>) -> Option<Ctx> {
        let cheap = super::rules::up(&sn.eps).min_by(|a, b| blend(a).total_cmp(&blend(b)))?.clone();
        let dear = super::rules::up(&sn.eps).max_by(|a, b| blend(a).total_cmp(&blend(b)))?.clone();
        let mark = mark.unwrap_or_else(|| blend(&cheap));
        Some(Ctx { cheap, dear, mark })
    }
}

fn row(name: &str, quant: &str, prompt: f64, completion: f64, mode: Mode) -> (OrEndpoint, Option<f64>) {
    let (status, u) = match mode {
        Mode::Live => (0, Some(100.0)),
        Mode::Idle => (1, Some(0.0)),
        Mode::Null => (0, None),
    };
    (OrEndpoint { provider: name.into(), quantization: quant.into(), prompt, completion, status }, u)
}

/// A row whose 15:1 blend is `b` USD per 1M tokens.
fn at_blend(name: &str, b: f64, mode: Mode) -> (OrEndpoint, Option<f64>) {
    row(name, "fp8", b * 1e-6, b * 1e-6, mode)
}

/// The rows (with uptime) one fake lists `el` steps (`el_ms`) after the anchor.
pub fn fake_rows(b: &Ub, c: &Ctx, name: &str, el: usize, el_ms: u64) -> Vec<(OrEndpoint, Option<f64>)> {
    let scaled = |e: &OrEndpoint, m: f64, q: Option<&str>, mode: Mode| row(name, q.unwrap_or(&e.quantization), e.prompt * m, e.completion * m, mode);
    match *b {
        Ub::Const { base, mult, mode, bf16 } => {
            let e = if base == Base::Cheap { &c.cheap } else { &c.dear };
            vec![scaled(e, mult, bf16.then_some("bf16"), mode)]
        }
        Ub::Sleeper { quiet_ms, jump, walk } => {
            let f = if el_ms < quiet_ms { 1.0 } else { jump * (1.0 + walk).powi(((el_ms - quiet_ms) / STEP) as i32 + 1) };
            vec![at_blend(name, c.mark * f, Mode::Live)]
        }
        Ub::Walk(r) => vec![at_blend(name, c.mark * (1.0 + r).powi(el as i32 + 1), Mode::Live)],
        Ub::FlickerStatus => {
            let (mut e, _) = scaled(&c.cheap, 0.5, None, Mode::Live);
            e.status = (el % 2) as i64;
            vec![(e, Some(100.0))]
        }
        Ub::Ratchet { up, mode } => {
            let (lo, hi) = if up { (0.9, 1.1) } else { (1.1, 0.9) };
            match el % 3 {
                0 => vec![],
                1 => vec![at_blend(name, c.mark * lo, mode)],
                _ => vec![at_blend(name, c.mark * hi, mode)],
            }
        }
        Ub::Alternate => vec![at_blend(name, c.mark * if el.is_multiple_of(2) { 0.5 } else { 2.0 }, Mode::Live)],
        Ub::Tiny1 => {
            if el == 0 {
                vec![row(name, "fp8", 0.0, 1e-160, Mode::Idle)]
            } else {
                vec![]
            }
        }
        Ub::Tiny => vec![row(name, "fp8", 1e-160, 1e-160, Mode::Live)],
        Ub::Huge => vec![row(name, "fp8", 1e160, 1e160, Mode::Live)],
        Ub::Zero => vec![row(name, "fp8", 0.0, 0.0, Mode::Live)],
        Ub::Neg => vec![row(name, "fp8", -1e-6, 1e-6, Mode::Live)],
        Ub::Nan => {
            let (e, _) = row(name, "fp8", f64::NAN, f64::INFINITY, Mode::Live);
            vec![(e, Some(f64::NAN))]
        }
        Ub::Dup => vec![scaled(&c.cheap, 0.5, None, Mode::Live); 4],
        Ub::ManyQ => (0..8).map(|q| scaled(&c.cheap, 0.5 * (1.0 + 0.01 * ((q + el) % 8) as f64), Some(&format!("q{q}")), Mode::Live)).collect(),
    }
}

// ---- Results. ----

#[derive(Clone, Debug)]
pub struct UHit {
    pub b: usize,
    pub k: usize,
    /// Worst move within each U_HORIZONS_MS, and over the whole window.
    pub mv_h: [f64; 2],
    pub mv: f64,
    /// Steps with a non-finite mark, or a hold the attack caused while 2+
    /// real sellers were up.
    pub bad: usize,
    pub at: String,
}

#[derive(Clone, Debug)]
pub struct SHit {
    pub k: usize,
    pub f: f64,
    pub steps: usize,
    pub mv: f64,
    pub at: String,
}

#[derive(Clone, Debug)]
pub struct PHit {
    pub k: usize,
    pub f: f64,
    pub mv: f64,
    pub fast: f64,
    pub at: String,
}

#[derive(Clone, Debug)]
pub struct MHit {
    pub all: bool,
    pub f: f64,
    pub score: f64,
    pub at: String,
}

#[derive(Clone, Debug)]
pub struct FHit {
    pub k: usize,
    pub mode: Mode,
    pub f: f64,
    pub score: f64,
    pub at: String,
}

/// Echo: a market-wide move by one of ECHO_SETS, read from ECHO_SETTLE_MS
/// to ECHO_WINDOW_MS after it.
#[derive(Clone, Debug)]
pub struct EHit {
    /// One of ECHO_SETS.
    pub set: &'static str,
    pub f: f64,
    /// Largest |attacked step - clean step| (log).
    pub step: f64,
    /// Largest |ln(attacked / clean) - share x ln f|.
    pub err: f64,
    pub at: String,
}

/// Coalition probe: the k dearest of n proven sellers move by COAL_F for good.
#[derive(Clone, Debug)]
pub struct CHit {
    pub k: usize,
    pub n: usize,
    pub mv: f64,
    pub at: String,
}

#[derive(Clone, Debug, Default)]
pub struct AttackOut {
    pub u: Vec<UHit>,
    pub s: Vec<SHit>,
    pub p: Vec<PHit>,
    pub m: Vec<MHit>,
    pub f: Vec<FHit>,
    pub e: Vec<EHit>,
    pub c: Vec<CHit>,
}

pub fn share_bound(k: usize, f: f64) -> f64 {
    SHARE_CAP * k as f64 * f.ln().abs()
}

pub fn fast_limit(k: usize) -> f64 {
    if k <= 2 {
        GATE_S[k - 1]
    } else {
        GATE_P_FAST_K3
    }
}

fn scale_rows(eps: &mut [OrEndpoint], who: &[String], f: f64) {
    for e in eps.iter_mut().filter(|e| who.contains(&e.provider)) {
        e.prompt *= f;
        e.completion *= f;
    }
}

/// Follow score of one step: max(0, 1 - |1 - ln(att/clean) / ln f|).
pub fn follow_score(att: Option<f64>, clean: Option<f64>, f: f64) -> f64 {
    match (att, clean) {
        (Some(a), Some(c)) => (1.0 - (1.0 - (a / c).ln() / f.ln()).abs()).max(0.0),
        _ => 0.0,
    }
}

/// An attack's input at (steps since the anchor, ms since it, clean snapshot).
type Input<'a> = dyn FnMut(usize, u64, &Snap) -> (Vec<OrEndpoint>, Vec<Option<f64>>) + 'a;
/// Sees (index, ms since the anchor, effective mark, what the rule returned);
/// false stops the replay.
type Obs<'a> = dyn FnMut(usize, u64, Option<f64>, Out) -> bool + 'a;

/// Every attack for one seed's anchors.
pub fn attacks(proto: &dyn Rule, s: &Series3, seed: u64, pool: &[String]) -> Result<AttackOut> {
    let an = anchors(s, seed);
    let save: BTreeSet<usize> = an.iter().copied().collect();
    let rp = replay(proto, s, &save);
    let proven = proven_at(s, &save);
    let listed: BTreeSet<&str> = s.snaps.iter().flat_map(|sn| sn.eps.iter().map(|e| e.provider.as_str())).collect();
    let t_first = s.snaps[0].at_ms;
    let mut out = AttackOut::default();
    let ubs = behaviours();
    let mut why: Option<String> = None;
    for &a in &an {
        let t0 = s.snaps[a].at_ms;
        let at = format!("{}@+{:.1}h", s.model, (t0 - t_first) as f64 / HOUR as f64);
        let rng = |x: u64| splitmix(seed ^ splitmix(fnv(&s.model) ^ splitmix(t0 ^ splitmix(x))));
        let (st, l0) = &rp.saved[&a];
        let run = |input: &mut Input, obs: &mut Obs, why: &mut Option<String>| {
            let mut r = st.clone_box();
            let mut last = *l0;
            for (i, sn) in s.snaps.iter().enumerate().skip(a) {
                let el_ms = sn.at_ms - t0;
                let (eps, upt) = input(i - a, el_ms, sn);
                let o = step(r.as_mut(), &mut last, sn.at_ms, &eps, &upt);
                if !obs(i, el_ms, last, o) {
                    break;
                }
            }
            if why.is_none() {
                *why = r.disqualified();
            }
        };
        // U.
        let Some(ctx) = Ctx::at(&s.snaps[a], rp.marks[a]) else { continue };
        let mut free: Vec<&String> = pool.iter().filter(|p| !listed.contains(p.as_str())).collect();
        let r = shuffle(&mut free, rng(1));
        let names: Vec<String> = (0..8).map(|n| free.get(n).map(|s| s.to_string()).unwrap_or_else(|| format!("fake{n}-{:08x}", splitmix(r ^ n as u64) >> 32))).collect();
        for (bi, ub) in ubs.iter().enumerate() {
            for &k in &U_K {
                // Serving time the fakes have earned; once it reaches
                // T_PROVEN they are proven sellers and P bounds them, so the
                // U window ends there.
                let acc = std::cell::Cell::new(0u64);
                let mut h = UHit { b: bi, k, mv_h: [0.0; 2], mv: 0.0, bad: 0, at: at.clone() };
                let mut input = |el: usize, el_ms: u64, sn: &Snap| {
                    let (mut eps, mut upt) = (sn.eps.clone(), sn.uptime.clone());
                    let mut served = false;
                    for nm in &names[..k] {
                        for (e, u) in fake_rows(ub, &ctx, nm, el, el_ms) {
                            served |= serving(&e, u);
                            eps.push(e);
                            upt.push(u);
                        }
                    }
                    if served {
                        acc.set(acc.get() + MAX_CREDIT_MS.min(STEP));
                    }
                    (eps, upt)
                };
                let mut obs = |i: usize, el_ms: u64, last: Option<f64>, o: Out| {
                    let caused_hold = o == Out::Hold && rp.priced[i] && v3::providers(&s.snaps[i].eps) >= 2;
                    if o == Out::Invalid || caused_hold {
                        h.bad += 1;
                    }
                    let d = v3::dist(last, rp.marks[i]);
                    for (x, hz) in h.mv_h.iter_mut().zip(U_HORIZONS_MS) {
                        if el_ms <= hz {
                            *x = x.max(d);
                        }
                    }
                    h.mv = h.mv.max(d);
                    acc.get() < T_PROVEN_MS
                };
                run(&mut input, &mut obs, &mut why);
                out.u.push(h);
            }
        }
        // Proven sellers up at the anchor, cheapest first, and a seeded order.
        let pv = &proven[&a];
        let low = v3::lowest(&s.snaps[a].eps);
        let ups: BTreeSet<&str> = super::rules::up(&s.snaps[a].eps).map(|e| e.provider.as_str()).collect();
        let mut cheap: Vec<(&str, f64)> = low.iter().filter(|(p, _)| ups.contains(*p) && pv.contains(**p)).map(|(p, b)| (*p, *b)).collect();
        cheap.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(y.0)));
        let cheap: Vec<String> = cheap.into_iter().map(|(p, _)| p.to_owned()).collect();
        let mut rand = cheap.clone();
        shuffle(&mut rand, rng(2));
        let n = cheap.len();
        if n < 2 {
            continue;
        }
        let picks = |k: usize| -> Vec<Vec<String>> {
            let mut v = vec![cheap[..k].to_vec()];
            if rand[..k] != cheap[..k] {
                v.push(rand[..k].to_vec());
            }
            v
        };
        // S: spike and revert.
        for k in [1, 2] {
            if k >= n {
                continue;
            }
            for who in picks(k) {
                for &f in &S_FACTORS {
                    for &steps in &S_STEPS {
                        let mut mv: f64 = 0.0;
                        let mut input = |el: usize, _: u64, sn: &Snap| {
                            let mut eps = sn.eps.clone();
                            if el < steps {
                                scale_rows(&mut eps, &who, f);
                            }
                            (eps, sn.uptime.clone())
                        };
                        let mut obs = |i: usize, _: u64, last: Option<f64>, _: Out| {
                            mv = mv.max(v3::dist(last, rp.marks[i]));
                            true
                        };
                        run(&mut input, &mut obs, &mut why);
                        out.s.push(SHit { k, f, steps, mv, at: format!("{at} {}", who.join("+")) });
                    }
                }
            }
        }
        // P: k proven sellers hold the move to the end. k < M_SHARE x n.
        let under = ((M_SHARE * n as f64).ceil() as usize).saturating_sub(1);
        let mut sets: Vec<(usize, Vec<String>)> = Vec::new();
        for k in [1, 2, 3, under] {
            if k == 0 || (k as f64) >= M_SHARE * n as f64 || sets.iter().any(|(kk, _)| *kk == k) {
                continue;
            }
            for who in if k <= 2 { picks(k) } else { vec![rand[..k].to_vec()] } {
                sets.push((k, who));
            }
        }
        for (k, who) in &sets {
            for &f in &P_FACTORS {
                let (mut mv, mut fast): (f64, f64) = (0.0, 0.0);
                let mut input = |_: usize, _: u64, sn: &Snap| {
                    let mut eps = sn.eps.clone();
                    scale_rows(&mut eps, who, f);
                    (eps, sn.uptime.clone())
                };
                let mut obs = |i: usize, el_ms: u64, last: Option<f64>, _: Out| {
                    let d = v3::dist(last, rp.marks[i]);
                    mv = mv.max(d);
                    if el_ms < P_DELAY_MS {
                        fast = fast.max(d);
                    }
                    true
                };
                run(&mut input, &mut obs, &mut why);
                out.p.push(PHit { k: *k, f, mv, fast, at: format!("{at} {}", who.join("+")) });
            }
        }
        // M and F: the market moves.
        let n60 = ((M_SHARE * n as f64).ceil() as usize).clamp(1, n);
        let movers60 = rand[..n60].to_vec();
        let start = |p: &str, f: f64| splitmix(rng(3) ^ fnv(p) ^ f.to_bits()) % (M_STAGGER_MS + 1);
        let follow_run = |who: &[String], f: f64, lead: u64, fakes: Option<(usize, Mode)>, why: &mut Option<String>| -> Option<f64> {
            let offs: Vec<u64> = who.iter().map(|p| lead + start(p, f)).collect();
            let mut sc = Vec::new();
            let mut input = |_: usize, el_ms: u64, sn: &Snap| {
                let mut eps = sn.eps.clone();
                let mut upt = sn.uptime.clone();
                for (p, off) in who.iter().zip(&offs) {
                    if el_ms >= *off {
                        scale_rows(&mut eps, std::slice::from_ref(p), f);
                    }
                }
                if let Some((k, mode)) = fakes {
                    for nm in &names[..k] {
                        let (e, u) = at_blend(nm, ctx.mark, mode);
                        eps.push(e);
                        upt.push(u);
                    }
                }
                (eps, upt)
            };
            let mut obs = |i: usize, el_ms: u64, last: Option<f64>, _: Out| {
                if el_ms >= lead + FOLLOW_READ_END_MS {
                    return false;
                }
                if el_ms >= lead + FOLLOW_WITHIN_MS {
                    sc.push(follow_score(last, rp.marks[i], f));
                }
                true
            };
            run(&mut input, &mut obs, why);
            (!sc.is_empty()).then(|| mean(&sc))
        };
        for (all, who) in [(false, &movers60), (true, &cheap)] {
            if all && n60 == n {
                continue;
            }
            for &f in &M_FACTORS {
                if let Some(score) = follow_run(who, f, 0, None, &mut why) {
                    out.m.push(MHit { all, f, score, at: at.clone() });
                }
            }
        }
        // Echo: every provider up at the
        // anchor moves at once, or the first M_SHARE of them by name, the j-th
        // of k at step j x ECHO_STAGGER_STEPS / k; read from ECHO_SETTLE_MS
        // after the move (after the stagger) to ECHO_WINDOW_MS. Also, not
        // gated, M's random 60% set at M's random times.
        let mut ups_now: Vec<String> = low.keys().filter(|p| ups.contains(**p)).map(|p| p.to_string()).collect();
        ups_now.sort();
        let k60 = ((M_SHARE * ups_now.len() as f64).ceil() as usize).clamp(1, ups_now.len().max(1));
        let first60 = ups_now[..k60.min(ups_now.len())].to_vec();
        for (set, who) in [(ECHO_SETS[0], &ups_now), (ECHO_SETS[1], &first60), (ECHO_SETS[2], &movers60)] {
            let all = set == ECHO_SETS[0];
            let share = if all { 1.0 } else { M_SHARE };
            let from = ECHO_SETTLE_MS + if all { 0 } else { M_STAGGER_MS };
            for &f in &ECHO_FACTORS {
                let (mut stp, mut err, mut seen): (f64, f64, bool) = (0.0, 0.0, false);
                let mut prev: Option<(f64, f64)> = None;
                let kk = who.len().max(1);
                let moved = |j: usize, p: &str, el: usize, el_ms: u64| match set {
                    s if s == ECHO_SETS[2] => el_ms >= start(p, f),
                    _ => el >= j * ECHO_STAGGER_STEPS / kk,
                };
                let mut input = |el: usize, el_ms: u64, sn: &Snap| {
                    let mut eps = sn.eps.clone();
                    for (j, p) in who.iter().enumerate() {
                        if all || moved(j, p, el, el_ms) {
                            scale_rows(&mut eps, std::slice::from_ref(p), f);
                        }
                    }
                    (eps, sn.uptime.clone())
                };
                let mut obs = |i: usize, el_ms: u64, last: Option<f64>, _: Out| {
                    if el_ms > ECHO_WINDOW_MS {
                        return false;
                    }
                    let now = match (last, rp.marks[i]) {
                        (Some(a), Some(c)) => Some((a.ln(), c.ln())),
                        _ => None,
                    };
                    if el_ms >= from {
                        if let Some((a, c)) = now {
                            seen = true;
                            err = err.max((a - c - share * f.ln()).abs());
                            if let Some((pa, pc)) = prev {
                                stp = stp.max(((a - pa) - (c - pc)).abs());
                            }
                        }
                    }
                    prev = now;
                    true
                };
                run(&mut input, &mut obs, &mut why);
                if seen {
                    out.e.push(EHit { set, f, step: stp, err, at: at.clone() });
                }
            }
        }
        // Coalition probe: the k dearest proven sellers halve for good.
        let dear: Vec<String> = cheap.iter().rev().cloned().collect();
        for &k in &COAL_K {
            if k >= n {
                continue;
            }
            let who = &dear[..k];
            let mut mv: f64 = 0.0;
            let mut input = |_: usize, _: u64, sn: &Snap| {
                let mut eps = sn.eps.clone();
                scale_rows(&mut eps, who, COAL_F);
                (eps, sn.uptime.clone())
            };
            let mut obs = |i: usize, _: u64, last: Option<f64>, _: Out| {
                mv = mv.max(v3::dist(last, rp.marks[i]));
                true
            };
            run(&mut input, &mut obs, &mut why);
            out.c.push(CHit { k, n, mv, at: at.clone() });
        }
        // F: every proven seller moves (so a miss is the idle listings'
        // doing, not the 60% quorum, which M tests).
        for &k in &U_K {
            for mode in [Mode::Idle, Mode::Live] {
                for &f in &F_FACTORS {
                    if let Some(score) = follow_run(&cheap, f, F_LEAD_MS, Some((k, mode)), &mut why) {
                        out.f.push(FHit { k, mode, f, score, at: at.clone() });
                    }
                }
            }
        }
    }
    if let Some(w) = why {
        bail!("disqualified: {w}");
    }
    Ok(out)
}

// ---- Pooling and the report. ----

#[derive(Clone, Debug, Default)]
pub struct RuleRun {
    pub clean: Vec<CleanOut>,
    pub att: Vec<AttackOut>,
}

/// One line per gate: (name, pass, value text, limit text, where).
#[derive(Clone, Debug)]
pub struct Gate {
    pub name: String,
    pub pass: bool,
    pub value: f64,
    pub limit: String,
    pub at: String,
}

fn fmt_gate(g: &Gate) -> String {
    format!("GATE {} {} {:.4} (limit {}) {}", g.name, if g.pass { "PASS" } else { "FAIL" }, g.value, g.limit, g.at)
}

/// Liveliness of a rule's clean runs: (L1 median, benchmark L1, L2 median over
/// 8+ provider models and their count, benchmark L2, L3 dead share, dead,
/// eligible, movement per day median).
#[derive(Clone, Debug, Default)]
pub struct Live {
    pub l1_median: f64,
    pub l1_bench: Option<f64>,
    pub l2_median: Option<f64>,
    pub l2_n: usize,
    pub l2_bench: Option<f64>,
    pub l3_share: f64,
    pub dead: usize,
    pub eligible: usize,
    pub moves_median: f64,
}

pub fn live(c: &[CleanOut]) -> Live {
    let bench = c.iter().find(|x| x.model == v3::BENCHMARK);
    let l2: Vec<f64> = c.iter().filter(|x| x.providers_ever >= L2_MIN_PROVIDERS).map(|x| x.corr.unwrap_or(0.0)).collect();
    let el: Vec<&CleanOut> = c.iter().filter(|x| x.routed_live).collect();
    let dead = el.iter().filter(|x| x.dead).count();
    Live {
        l1_median: median(c.iter().map(|x| x.range.unwrap_or(0.0)).collect()).unwrap_or(0.0),
        l1_bench: bench.map(|b| b.range.unwrap_or(0.0)),
        l2_median: median(l2.clone()),
        l2_n: l2.len(),
        l2_bench: bench.map(|b| b.corr.unwrap_or(0.0)),
        l3_share: if el.is_empty() { 0.0 } else { dead as f64 / el.len() as f64 },
        dead,
        eligible: el.len(),
        moves_median: median(c.iter().map(|x| if x.days > 0.0 { x.movement / x.days } else { 0.0 }).collect()).unwrap_or(0.0),
    }
}

fn pct(x: Option<f64>) -> String {
    x.map(|x| format!("{:.2}%", 100.0 * x)).unwrap_or_else(|| "-".into())
}

fn num(x: Option<f64>) -> String {
    x.map(|x| format!("{x:.3}")).unwrap_or_else(|| "-".into())
}

pub fn live_line(l: &Live) -> String {
    format!(
        "LIVE L1_range_per_day median {} bench {} | L2_corr_1h_vs_R60 median {} (models with {L2_MIN_PROVIDERS}+ providers: {}) bench {} | L3_dead {} ({}/{} models where routed moved > {}%) | movement_per_day median {}",
        pct(Some(l.l1_median)),
        pct(l.l1_bench),
        num(l.l2_median),
        l.l2_n,
        num(l.l2_bench),
        pct(Some(l.l3_share)),
        l.dead,
        l.eligible,
        L3_ROUTED_MIN * 100.0,
        pct(Some(l.moves_median)),
    )
}

/// The gates of one rule; `spec` is the live rule's liveliness (L1 is relative).
pub fn gates(rr: &RuleRun, spec: &Live) -> Vec<Gate> {
    let ubs = behaviours();
    let mut g = Vec::new();
    let worst = |v: Vec<(f64, String)>| v.into_iter().fold((-1.0, "none".to_owned()), |a, b| if b.0 > a.0 { b } else { a });
    // U.
    let u: Vec<&UHit> = rr.att.iter().flat_map(|a| a.u.iter()).collect();
    let (mx, at) = worst(u.iter().map(|h| (h.mv, format!("k={} {} {} (6h {:.4}, 24h {:.4})", h.k, ub_name(&ubs[h.b]), h.at, h.mv_h[0], h.mv_h[1]))).collect());
    g.push(Gate { name: "U".into(), pass: mx <= GATE_U, value: mx, limit: format!("{GATE_U}"), at: format!("worst {at}; n {}", u.len()) });
    let clean_bad: usize = rr.clean.iter().map(|c| c.invalid).sum();
    let bad_n = u.iter().filter(|h| h.bad > 0).count();
    let bad_at = u.iter().find(|h| h.bad > 0).map(|h| format!("first k={} {} {}", h.k, ub_name(&ubs[h.b]), h.at)).unwrap_or_default();
    g.push(Gate {
        name: "U_finite".into(),
        pass: bad_n == 0 && clean_bad == 0,
        value: (bad_n + clean_bad) as f64,
        limit: "0".into(),
        at: format!("attacks with a non-finite mark or an attack-caused hold {bad_n}/{}; clean non-finite steps {clean_bad} {bad_at}", u.len()),
    });
    // S.
    for k in [1, 2] {
        let (mx, at) = worst(rr.att.iter().flat_map(|a| a.s.iter()).filter(|h| h.k == k && h.steps <= S_GATED_STEPS).map(|h| (h.mv, format!("x{} {} snapshots {}", h.f, h.steps, h.at))).collect());
        let lim = GATE_S[k - 1];
        g.push(Gate { name: format!("S_k{k}"), pass: mx <= lim, value: mx, limit: format!("{lim}"), at: format!("worst {at}") });
    }
    // P.
    let p: Vec<&PHit> = rr.att.iter().flat_map(|a| a.p.iter()).collect();
    let (mx, at) = worst(p.iter().map(|h| (h.mv / share_bound(h.k, h.f), format!("k={} x{} move {:.4} bound {:.4} {}", h.k, h.f, h.mv, share_bound(h.k, h.f), h.at))).collect());
    g.push(Gate { name: "P".into(), pass: mx <= 1.0, value: mx, limit: format!("1 = {SHARE_CAP} x k x |ln f|"), at: format!("worst move/bound: {at}; n {}", p.len()) });
    let (mx, at) = worst(p.iter().map(|h| (h.fast / fast_limit(h.k), format!("k={} x{} move {:.4} in the first {} min, limit {} {}", h.k, h.f, h.fast, P_DELAY_MS / MIN, fast_limit(h.k), h.at))).collect());
    g.push(Gate { name: "P_fast".into(), pass: mx <= 1.0, value: mx, limit: "1".into(), at: format!("worst {at}") });
    // M and F: the worst group mean.
    let m: Vec<&MHit> = rr.att.iter().flat_map(|a| a.m.iter()).collect();
    let mut by: BTreeMap<(bool, u64), Vec<f64>> = BTreeMap::new();
    for h in &m {
        by.entry((h.all, h.f.to_bits())).or_default().push(h.score);
    }
    let (mn, at) = by.iter().map(|((all, f), v)| (mean(v), format!("{} x{:.3} n {}", if *all { "all" } else { "60%" }, f64::from_bits(*f), v.len()))).fold((f64::MAX, "none".to_owned()), |a, b| if b.0 < a.0 { b } else { a });
    let mn = if by.is_empty() { 0.0 } else { mn };
    let lowest = m.iter().map(|h| (h.score, &h.at)).fold((f64::MAX, None), |a, b| if b.0 < a.0 { (b.0, Some(b.1)) } else { a });
    g.push(Gate {
        name: "M".into(),
        pass: !by.is_empty() && mn >= GATE_FOLLOW,
        value: mn,
        limit: format!(">= {GATE_FOLLOW}"),
        at: format!("worst group {at}; mean all {:.4}; lowest instance {:.4} {}", mean(&m.iter().map(|h| h.score).collect::<Vec<_>>()), if lowest.1.is_some() { lowest.0 } else { 0.0 }, lowest.1.map(|s| s.as_str()).unwrap_or("")),
    });
    let f: Vec<&FHit> = rr.att.iter().flat_map(|a| a.f.iter()).collect();
    let mut by: BTreeMap<(usize, Mode), Vec<f64>> = BTreeMap::new();
    for h in &f {
        by.entry((h.k, h.mode)).or_default().push(h.score);
    }
    let (mn, at) = by.iter().map(|((k, mode), v)| (mean(v), format!("k={k} {} n {}", mode_s(*mode), v.len()))).fold((f64::MAX, "none".to_owned()), |a, b| if b.0 < a.0 { b } else { a });
    let mn = if by.is_empty() { 0.0 } else { mn };
    g.push(Gate { name: "F".into(), pass: !by.is_empty() && mn >= GATE_FOLLOW, value: mn, limit: format!(">= {GATE_FOLLOW}"), at: format!("worst group {at}") });
    // Echo: the step on every case; the error on all but the known exceptions.
    let e: Vec<&EHit> = rr.att.iter().flat_map(|a| a.e.iter()).filter(|h| h.set != ECHO_SETS[2]).collect();
    let name = |h: &EHit| format!("{} x{:.3} {}", h.set, h.f, h.at);
    let (mx, at) = worst(e.iter().map(|h| (h.step, name(h))).collect());
    // 1e-9: two full 1% steps in opposite directions sit on the limit.
    g.push(Gate { name: "echo_step".into(), pass: !e.is_empty() && mx <= GATE_ECHO_STEP + 1e-9, value: mx.max(0.0), limit: format!("{GATE_ECHO_STEP}"), at: format!("worst {at}; n {}", e.len()) });
    let known = |h: &EHit| ECHO_KNOWN_EXCEPTIONS.iter().any(|(f, set)| (h.f - f).abs() < 1e-9 && h.set == *set);
    let (mx, at) = worst(e.iter().filter(|h| !known(h)).map(|h| (h.err, name(h))).collect());
    g.push(Gate {
        name: "echo_error".into(),
        pass: !e.is_empty() && mx <= GATE_ECHO_ERROR,
        value: mx.max(0.0),
        limit: format!("{GATE_ECHO_ERROR}"),
        at: format!("worst {at}; known exceptions not gated: 60% x0.5 (see ADVISORY echo_known_exception)"),
    });
    // Liveliness. On a run of the benchmark alone (pin-rec) the benchmark
    // terms are gated; on a multi-model run the medians over models are (the
    // benchmark's 21 h there is printed, not gated).
    let l = live(&rr.clean);
    let bench_only = rr.clean.len() == 1 && l.l1_bench.is_some();
    if bench_only {
        let lim = spec.l1_bench.unwrap_or(0.0).max(GATE_L1_BENCH);
        let b = l.l1_bench.unwrap_or(0.0);
        g.push(Gate { name: "L1_range".into(), pass: b >= lim, value: 100.0 * b, limit: format!(">= {:.2} (% per day: spec's, at least {}%)", 100.0 * lim, GATE_L1_BENCH * 100.0), at: "benchmark".into() });
        let b = l.l2_bench.unwrap_or(0.0);
        g.push(Gate { name: "L2_corr".into(), pass: b >= GATE_L2, value: b, limit: format!(">= {GATE_L2}"), at: "benchmark, hourly log moves vs R60".into() });
    } else {
        g.push(Gate {
            name: "L1_range".into(),
            pass: l.l1_median >= spec.l1_median,
            value: 100.0 * l.l1_median,
            limit: format!(">= {:.2} (spec's; median over models, % per day)", 100.0 * spec.l1_median),
            at: format!("benchmark {} (gated on pin-rec)", pct(l.l1_bench)),
        });
        let m = l.l2_median.unwrap_or(0.0);
        g.push(Gate {
            name: "L2_corr".into(),
            pass: l.l2_median.is_some() && m >= GATE_L2,
            value: m,
            limit: format!(">= {GATE_L2}"),
            at: format!("median over {} models with {L2_MIN_PROVIDERS}+ providers, hourly log moves vs R60; benchmark {} (gated on pin-rec)", l.l2_n, num(l.l2_bench)),
        });
    }
    g.push(Gate {
        name: "L3_dead".into(),
        pass: l.l3_share <= GATE_L3,
        value: 100.0 * l.l3_share,
        limit: format!("{}%", GATE_L3 * 100.0),
        at: format!("{}/{} models never moved where routed moved > {}%", l.dead, l.eligible, L3_ROUTED_MIN * 100.0),
    });
    // Level, phantom.
    let lv: Vec<f64> = rr.clean.iter().flat_map(|c| c.level_r60.iter().copied()).collect();
    let lvm = median(lv).unwrap_or(0.0);
    g.push(Gate { name: "level_r60".into(), pass: lvm <= GATE_LEVEL, value: lvm, limit: format!("{GATE_LEVEL}"), at: "median |ln(mark/R60)|".into() });
    let (mv, ph): (f64, f64) = rr.clean.iter().fold((0.0, 0.0), |a, c| (a.0 + c.movement, a.1 + c.phantom));
    let share = if mv > 0.0 { ph / mv } else { 0.0 };
    g.push(Gate { name: "phantom_share".into(), pass: share <= GATE_PHANTOM_SHARE, value: share, limit: format!("{GATE_PHANTOM_SHARE}"), at: String::new() });
    g
}

/// Detail lines for the rule under test.
pub fn details(rr: &RuleRun) -> Vec<String> {
    let ubs = behaviours();
    let mut out = Vec::new();
    let u: Vec<&UHit> = rr.att.iter().flat_map(|a| a.u.iter()).collect();
    for &k in &U_K {
        let v: Vec<&&UHit> = u.iter().filter(|h| h.k == k).collect();
        let mx = |f: &dyn Fn(&UHit) -> f64| v.iter().map(|h| f(h)).fold(0.0, f64::max);
        let w = v.iter().fold(None::<&&UHit>, |a, h| if a.is_none_or(|a| h.mv > a.mv) { Some(h) } else { a });
        out.push(format!(
            "U k={k} 6h {:.4} 24h {:.4} end {:.4} bad {} n {} worst {}",
            mx(&|h| h.mv_h[0]),
            mx(&|h| h.mv_h[1]),
            mx(&|h| h.mv),
            v.iter().filter(|h| h.bad > 0).count(),
            v.len(),
            w.map(|h| format!("{} {}", ub_name(&ubs[h.b]), h.at)).unwrap_or_default()
        ));
    }
    for (bi, b) in ubs.iter().enumerate() {
        let v: Vec<&&UHit> = u.iter().filter(|h| h.b == bi).collect();
        let w = v.iter().fold(None::<&&UHit>, |a, h| if a.is_none_or(|a| h.mv > a.mv) { Some(h) } else { a });
        if let Some(w) = w {
            out.push(format!("UB {} end {:.4} 24h {:.4} 6h {:.4} bad {} worst k={} {}", ub_name(b), w.mv, w.mv_h[1], w.mv_h[0], v.iter().filter(|h| h.bad > 0).count(), w.k, w.at));
        }
    }
    let s: Vec<&SHit> = rr.att.iter().flat_map(|a| a.s.iter()).collect();
    for k in [1, 2] {
        for &f in &S_FACTORS {
            for &d in &S_STEPS {
                let v: Vec<f64> = s.iter().filter(|h| h.k == k && h.f == f && h.steps == d).map(|h| h.mv).collect();
                if !v.is_empty() {
                    out.push(format!("S k={k} x{f} {d} snapshots worst {:.4} mean {:.4} n {}", v.iter().copied().fold(0.0, f64::max), mean(&v), v.len()));
                }
            }
        }
    }
    let p: Vec<&PHit> = rr.att.iter().flat_map(|a| a.p.iter()).collect();
    let ks: BTreeSet<usize> = p.iter().map(|h| h.k).collect();
    for k in ks {
        for &f in &P_FACTORS {
            let v: Vec<&&PHit> = p.iter().filter(|h| h.k == k && h.f == f).collect();
            if !v.is_empty() {
                let mx = v.iter().map(|h| h.mv).fold(0.0, f64::max);
                out.push(format!(
                    "P k={k} x{f} worst {:.4} mean {:.4} bound {:.4} fast {:.4} n {}",
                    mx,
                    mean(&v.iter().map(|h| h.mv).collect::<Vec<_>>()),
                    share_bound(k, f),
                    v.iter().map(|h| h.fast).fold(0.0, f64::max),
                    v.len()
                ));
            }
        }
    }
    let m: Vec<&MHit> = rr.att.iter().flat_map(|a| a.m.iter()).collect();
    for all in [false, true] {
        for &f in &M_FACTORS {
            let v: Vec<f64> = m.iter().filter(|h| h.all == all && h.f == f).map(|h| h.score).collect();
            if !v.is_empty() {
                out.push(format!("M {} x{f:.3} mean {:.4} min {:.4} n {}", if all { "all" } else { "60%" }, mean(&v), v.iter().copied().fold(f64::MAX, f64::min), v.len()));
            }
        }
    }
    let f: Vec<&FHit> = rr.att.iter().flat_map(|a| a.f.iter()).collect();
    for &k in &U_K {
        for mode in [Mode::Idle, Mode::Live] {
            let v: Vec<f64> = f.iter().filter(|h| h.k == k && h.mode == mode).map(|h| h.score).collect();
            if !v.is_empty() {
                out.push(format!("F k={k} {} mean {:.4} min {:.4} n {}", mode_s(mode), mean(&v), v.iter().copied().fold(f64::MAX, f64::min), v.len()));
            }
        }
    }
    let e: Vec<&EHit> = rr.att.iter().flat_map(|a| a.e.iter()).collect();
    for set in ECHO_SETS {
        for &f in &ECHO_FACTORS {
            let v: Vec<&&EHit> = e.iter().filter(|h| h.set == set && h.f == f).collect();
            if !v.is_empty() {
                out.push(format!(
                    "E {set} x{f:.3} step worst {:.4} error worst {:.4} n {}{}",
                    v.iter().map(|h| h.step).fold(0.0, f64::max),
                    v.iter().map(|h| h.err).fold(0.0, f64::max),
                    v.len(),
                    if set == ECHO_SETS[2] { "  # not gated" } else { "" }
                ));
            }
        }
    }
    out
}

/// Advisory lines: instantaneous level, stale, real market events, gain.
pub fn advisory(rr: &RuleRun, spec: &RuleRun) -> Vec<String> {
    let mut out = Vec::new();
    let lv: Vec<f64> = rr.clean.iter().flat_map(|c| c.level_now.iter().copied()).collect();
    out.push(format!("ADVISORY level_now_median {:.4}  # median |ln(mark/routed)|, routed instantaneous", median(lv).unwrap_or(0.0)));
    let days: f64 = rr.clean.iter().map(|c| c.days).sum();
    let st: f64 = rr.clean.iter().map(|c| c.stale).sum();
    out.push(format!("ADVISORY stale {:.6}  # log/day moved on input identical to the step before (a hold expiring); inside phantom", if days > 0.0 { st / days } else { 0.0 }));
    let ev: Vec<f64> = rr.clean.iter().flat_map(|c| c.market.iter().copied()).collect();
    out.push(format!("ADVISORY follow_real_market {:.4} events {}  # real moves by {}%+ of 3+ sellers, read {} steps later", mean(&ev), ev.len(), M_SHARE * 100.0, MARKET_N));
    let holds: usize = rr.clean.iter().map(|c| c.holds).sum();
    let steps: usize = rr.clean.iter().map(|c| c.steps).sum();
    out.push(format!("ADVISORY holds {holds}/{steps}"));
    let s90 = S_STEPS[S_STEPS.len() - 1];
    for k in [1, 2] {
        let w = rr.att.iter().flat_map(|a| a.s.iter()).filter(|h| h.k == k && h.steps == s90).fold(None::<&SHit>, |a, h| if a.is_none_or(|a| h.mv > a.mv) { Some(h) } else { a });
        if let Some(h) = w {
            out.push(format!("ADVISORY S90_k{k} {:.4} worst x{} {s90} snapshots {}  # a {}-minute spike; not gated", h.mv, h.f, h.at, s90 * 5));
        }
    }
    for (f, set) in ECHO_KNOWN_EXCEPTIONS {
        let w = rr.att.iter().flat_map(|a| a.e.iter()).filter(|h| (h.f - f).abs() < 1e-9 && h.set == set).fold(None::<&EHit>, |a, h| if a.is_none_or(|a| h.err > a.err) { Some(h) } else { a });
        if let Some(h) = w {
            out.push(format!("ADVISORY echo_known_exception {set} x{f} error {:.4} (step {:.4}, gated) {}  # market credit fading after the move; the error is not gated", h.err, h.step, h.at));
        }
    }
    let r: Vec<&EHit> = rr.att.iter().flat_map(|a| a.e.iter()).filter(|h| h.set == ECHO_SETS[2]).collect();
    if let (Some(ws), Some(we)) = (r.iter().max_by(|a, b| a.step.total_cmp(&b.step)), r.iter().max_by(|a, b| a.err.total_cmp(&b.err))) {
        out.push(format!("ADVISORY echo_random_set step {:.4} x{:.3} {} error {:.4} x{:.3} {}  # M's random 60% at random times; not gated", ws.step, ws.f, ws.at, we.err, we.f, we.at));
    }
    for k in COAL_K {
        let w = rr.att.iter().flat_map(|a| a.c.iter()).filter(|h| h.k == k).fold(None::<&CHit>, |a, h| if a.is_none_or(|a| h.mv > a.mv) { Some(h) } else { a });
        if let Some(h) = w {
            out.push(format!("ADVISORY coalition k={k} of {} dearest x{COAL_F} move {:.4} (|ln f| {:.4}) {}  # not gated", h.n, h.mv, COAL_F.ln().abs(), h.at));
        }
    }
    // Gain vs spec: per series, level_r60 mean + phantom/day + mean U move + mean S move.
    let loss = |r: &RuleRun| -> Vec<f64> {
        r.clean
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let u: Vec<f64> = r.att.iter().skip(i * r.att.len() / r.clean.len().max(1)).take(r.att.len() / r.clean.len().max(1)).flat_map(|a| a.u.iter().map(|h| h.mv).chain(a.s.iter().map(|h| h.mv))).collect();
                mean(&c.level_r60) + if c.days > 0.0 { c.phantom / c.days } else { 0.0 } + mean(&u)
            })
            .collect()
    };
    let (lr, ls) = (loss(rr), loss(spec));
    let (g, se) = v3::mean_se(&ls.iter().zip(&lr).map(|(s, r)| s - r).collect::<Vec<_>>());
    out.push(format!("ADVISORY gain_vs_spec {g:+.6} se {se:.6} n {}  # level_r60 + phantom/day + mean U and S move, per series; positive = better than spec; not a gate", lr.len()));
    out
}

/// Builds rules by name (one per worker thread; rules need not be Send).
pub type Factory<'a> = &'a (dyn Fn(&str) -> Result<Box<dyn Rule>> + Sync);

enum Job {
    Clean(usize, usize),
    Attack(usize, usize, u64),
}

enum JobOut {
    Clean(CleanOut),
    Attack(AttackOut),
}

/// Runs `rules` over `series` (clean, and every seed's attacks), in parallel.
pub fn run_all(rules: &[String], factory: Factory, series: &[&Series3], seeds: &[u64], pool: &[String], threads: usize) -> Result<Vec<RuleRun>> {
    let mut jobs = Vec::new();
    // Long series first, for balance; results land by index.
    let mut order: Vec<usize> = (0..series.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(series[i].snaps.len()));
    for &si in &order {
        for ri in 0..rules.len() {
            for &sd in seeds {
                jobs.push(Job::Attack(ri, si, sd));
            }
            jobs.push(Job::Clean(ri, si));
        }
    }
    let results: Vec<Mutex<Option<Result<JobOut>>>> = jobs.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let mut protos: BTreeMap<usize, Box<dyn Rule>> = BTreeMap::new();
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs.len() {
                        break;
                    }
                    let (ri, si) = match jobs[j] {
                        Job::Clean(r, s) | Job::Attack(r, s, _) => (r, s),
                    };
                    let res = (|| -> Result<JobOut> {
                        if let std::collections::btree_map::Entry::Vacant(v) = protos.entry(ri) {
                            v.insert(factory(&rules[ri])?);
                        }
                        let p = protos[&ri].as_ref();
                        Ok(match jobs[j] {
                            Job::Clean(..) => {
                                let c = clean(p, series[si]);
                                JobOut::Clean(c)
                            }
                            Job::Attack(_, _, sd) => JobOut::Attack(attacks(p, series[si], sd, pool)?),
                        })
                    })()
                    .map_err(|e| anyhow!("{e:#} (rule {}, {})", rules[ri], series[si].model));
                    *results[j].lock().unwrap() = Some(res);
                }
            });
        }
    });
    let mut out: Vec<RuleRun> = rules.iter().map(|_| RuleRun { clean: vec![CleanOut::default(); series.len()], att: vec![AttackOut::default(); series.len() * seeds.len()] }).collect();
    for (j, r) in jobs.iter().zip(results) {
        let res = r.into_inner().unwrap().ok_or_else(|| anyhow!("a job did not run"))??;
        match (j, res) {
            (Job::Clean(ri, si), JobOut::Clean(c)) => out[*ri].clean[*si] = c,
            (Job::Attack(ri, si, sd), JobOut::Attack(a)) => {
                let k = seeds.iter().position(|x| x == sd).unwrap();
                out[*ri].att[si * seeds.len() + k] = a;
            }
            _ => unreachable!(),
        }
    }
    Ok(out)
}

/// Run everything and return the report lines.
#[allow(clippy::too_many_arguments)]
pub fn report(rule: &str, baselines: &[String], factory: Factory, series: &[Series3], manifest: &str, data: &str, split: Split, seeds: &[u64], threads: usize) -> Result<Vec<String>> {
    let pick: Vec<&Series3> = series.iter().filter(|s| v3::in_split(&s.model, split)).collect();
    if pick.is_empty() {
        bail!("{split:?} split has no series");
    }
    if seeds.is_empty() {
        bail!("no seeds");
    }
    let names: Vec<String> = series.iter().flat_map(|s| s.snaps.iter().flat_map(|sn| sn.eps.iter().map(|e| e.provider.clone()))).collect::<BTreeSet<_>>().into_iter().collect();
    let mut rules = vec![rule.to_owned()];
    for b in std::iter::once("spec".to_owned()).chain(baselines.iter().cloned()) {
        if !rules.contains(&b) {
            rules.push(b);
        }
    }
    let runs = run_all(&rules, factory, &pick, seeds, &names, threads)?;
    let si = rules.iter().position(|r| r == "spec").unwrap();
    let spec_live = live(&runs[si].clean);
    let mut out = Vec::new();
    out.push(format!(
        "CHECK v4 rule {rule} split {split:?} series {} seeds {}..{} anchors/seed {ANCHORS_PER_SEED} data {data} manifest {manifest}",
        pick.len(),
        seeds[0],
        seeds[seeds.len() - 1]
    ));
    out.push(format!("LEVEL_DEF {LEVEL_DEF}"));
    out.push(format!("PROVEN_DEF {PROVEN_DEF}"));
    out.push(live_line(&live(&runs[0].clean)));
    out.extend(details(&runs[0]));
    out.extend(advisory(&runs[0], &runs[si]));
    out.extend(gates(&runs[0], &spec_live).iter().map(fmt_gate));
    for (ri, r) in rules.iter().enumerate().skip(1) {
        if !baselines.contains(r) {
            continue;
        }
        out.push(format!("BASE {r} {}", live_line(&live(&runs[ri].clean))));
        out.extend(gates(&runs[ri], &spec_live).iter().map(|g| format!("BASE {r} {}", fmt_gate(g))));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
