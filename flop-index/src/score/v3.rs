//! Check v3 (docs/flop-compute/SCORER.md, "Check v3").
//!
//! Scores a settlement rule on 5-minute recorder data, one series per model
//! (rules are model-agnostic), and makes a reviewer's objections measurable:
//!
//! - phantom: mark movement with no new listed price behind it (a seller
//!   flickering up or down, uptime moving, a hold expiring); at a step that
//!   also has a new price, the part the availability change made;
//! - reversal: movement undone within the next K scored steps;
//! - follow: how much of a real consensus move the mark shows N steps later
//!   (consensus = the mean log price change of every listed seller, so it
//!   does not depend on any rule); also an injected all-seller move;
//! - level: distance from `routed`, what a request pays now;
//! - attacks injected on real data for k = 1, 2, 3 colluding sellers, with
//!   the fakes or victims in any precision or in bf16;
//! - stale: movement at steps whose input is identical to the step before
//!   (a hold expiring; it is inside phantom too);
//! - distinct marks.
//!
//! Gates are the constants below. Nothing here changes check v1 or v2.

use super::data::{self, DAY_MS};
use super::rules::{blend, up, Rule};
use super::{fnv, splitmix, Split};
use crate::OrEndpoint;
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

// ---- Gates. ----

/// Largest move any attack may cause, |ln(attacked / clean mark)|, for
/// k = 1, 2, 3 colluding sellers. "down" (half the sellers) counts as k = 3.
pub const GATE_ATTACK_MAX: [f64; 3] = [0.02, 0.05, 0.10];
/// Phantom movement as a share of all movement.
pub const GATE_PHANTOM_SHARE: f64 = 0.20;
/// Mean share of real consensus moves the mark shows FOLLOW_N steps later.
pub const GATE_FOLLOW: f64 = 0.5;
/// The same for the injected all-seller move.
pub const GATE_FOLLOW_INJ: f64 = 0.5;
/// Median |ln(mark / routed)| over scored steps.
pub const GATE_LEVEL_MEDIAN: f64 = 0.05;
/// Paired gain over the base must exceed this many standard errors (heldout).
pub const GATE_GAIN_SE: f64 = 2.0;

// ---- Parameters. ----

/// The benchmark model: always in the heldout split.
pub const BENCHMARK: &str = "qwen/qwen3.8-27b";
/// A model is scored if it ever lists this many distinct providers.
pub const MIN_PROVIDERS: usize = 5;
/// Reversal window in scored steps (6 = 30 min).
pub const REVERSAL_K: usize = 6;
/// Follow: steps after a consensus move at which the mark is read (30 min).
pub const FOLLOW_N: usize = 6;
/// Follow: a consensus move counts from this size (log), and only if at
/// least half of it is still there FOLLOW_N steps later and is new against
/// FOLLOW_N steps before it (a promo ending is not a move).
pub const FOLLOW_MIN: f64 = 0.01;
/// Routed: no provider gets more than this share of the weight.
pub const ROUTED_CAP: f64 = 0.5;
/// Loss when the rule has no mark where one is needed.
pub const PENALTY: f64 = 1.0;
const MIN: u64 = 60_000;
const HOUR: u64 = 60 * MIN;
/// Attack anchors: one per this period per series, at a seeded offset.
pub const ANCHOR_EVERY_MS: u64 = 3 * HOUR;
/// An anchor needs this much data after it.
const ANCHOR_TAIL_MS: u64 = 2 * HOUR;
/// Each attack is measured this long after it ends.
pub const AFTER_MS: u64 = HOUR;
/// A seller that goes down loses uptime_30m linearly over this long.
pub const UPTIME_DECAY_MS: u64 = 30 * MIN;
/// Attacks: name, duration, price multiplier. Fakes are priced off the
/// cheapest up seller; promo, relist and newlow multiply a real seller's
/// price; newlow lists its new price for one step, then goes down with it
/// still listed; down takes half the up sellers offline.
pub const SCENARIOS: [(&str, u64, f64); 6] = [
    ("fake-0.5x", 6 * HOUR, 0.5),
    ("fake-0.1x", 6 * HOUR, 0.1),
    ("promo-0.5x-30m", 30 * MIN, 0.5),
    ("relist-10x", 6 * HOUR, 10.0),
    ("newlow-offline", 6 * HOUR, 0.5),
    ("down-half-1h", HOUR, 1.0),
];
const DOWN: usize = 5;
/// "any": fakes copy the cheapest seller's quantization and victims are any
/// up seller; "bf16": fakes say bf16 and victims are bf16 sellers.
pub const QUANTS: [&str; 2] = ["any", "bf16"];
/// Injected genuine move: every seller about x1.1 to x2 (log-uniform, up or
/// down), each within +-3% of it, each at its own time in the first 15 min;
/// read from 30 min after the last seller moved to 3 h after the anchor.
pub const INJ_MOVE: (f64, f64) = (1.1, 2.0);
pub const INJ_JITTER: f64 = 0.03;
pub const INJ_STAGGER_MS: u64 = 15 * MIN;
pub const INJ_WINDOW_MS: u64 = 3 * HOUR;

pub const LEVEL_DEF: &str = "routed = over sellers that are up, one price per provider (its lowest 15:1 blend), weight uptime_30m/price^2 (null uptime: weight 0), no provider above 50% of the weight (excess spread pro rata), price = sum(w*p)/sum(w); none if fewer than 2 providers have weight";

/// One 5-minute snapshot of one model: every listed endpoint, down ones
/// too, and each one's uptime_30m (percent; None = not reported).
#[derive(Clone, Debug, PartialEq)]
pub struct Snap {
    pub at_ms: u64,
    pub eps: Vec<OrEndpoint>,
    pub uptime: Vec<Option<f64>>,
}

#[derive(Clone, Debug)]
pub struct Series3 {
    pub model: String,
    pub snaps: Vec<Snap>,
}

/// Heldout: the benchmark, and every model whose FNV-1a hash is odd. Stable
/// across data: a model never changes side.
pub fn heldout(model: &str) -> bool {
    model == BENCHMARK || fnv(model) & 1 == 1
}

pub fn in_split(model: &str, split: Split) -> bool {
    match split {
        Split::All => true,
        Split::Heldout => heldout(model),
        Split::Tune => !heldout(model),
    }
}

/// A pin of 5-minute recorder files (`recorder-jsonl`, `wide-jsonl`), one
/// series per model listing MIN_PROVIDERS+ distinct providers. Fails on any
/// other loader: v3 needs uptime.
pub fn load(dir: &Path) -> Result<(Vec<Series3>, String)> {
    let (files, manifest) = data::verified(dir)?;
    let mut groups = Groups::new();
    for (loader, file, text) in &files {
        match loader.as_str() {
            "recorder-jsonl" | "wide-jsonl" => parse_jsonl(text, &mut groups).with_context(|| format!("load {file}"))?,
            other => bail!("check v3 reads 5-minute recorder pins (recorder-jsonl, wide-jsonl), not {other} ({file}): it needs uptime"),
        }
    }
    let series: Vec<Series3> = groups
        .into_iter()
        .filter(|(_, s)| s.values().flat_map(|r| r.iter().map(|(e, _)| e.provider.as_str())).collect::<BTreeSet<_>>().len() >= MIN_PROVIDERS)
        .map(|(model, s)| Series3 {
            model,
            snaps: s
                .into_iter()
                .map(|(at_ms, rows)| {
                    let (eps, uptime) = rows.into_iter().unzip();
                    Snap { at_ms, eps, uptime }
                })
                .collect(),
        })
        .collect();
    if series.is_empty() {
        bail!("no model in {} lists {MIN_PROVIDERS}+ providers", dir.display());
    }
    Ok((series, manifest))
}

/// Recorder lines. Uptime: the row's `uptime_30m` (wide recorder), else the
/// `uptime_last_30m` of the endpoint with the row's tag in `raw` (the
/// single-model recorder keeps it only there).
/// Model -> time -> rows with uptime.
pub(crate) type Groups = BTreeMap<String, BTreeMap<u64, Vec<(OrEndpoint, Option<f64>)>>>;

pub(crate) fn parse_jsonl(text: &str, groups: &mut Groups) -> Result<()> {
    let num = |v: &serde_json::Value| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).with_context(|| format!("line {}", n + 1))?;
        let at = v["at_ms"].as_u64().with_context(|| format!("line {}: no at_ms", n + 1))?;
        let Some(rows) = v["rows"].as_array() else { continue };
        let url_model = v["url"].as_str().and_then(|u| u.split("/models/").nth(1)).and_then(|m| m.strip_suffix("/endpoints"));
        let raw_up: BTreeMap<String, Option<f64>> = v["raw"]
            .as_str()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok())
            .and_then(|r| r["data"]["endpoints"].as_array().cloned())
            .map(|eps| eps.iter().filter_map(|e| Some((e["tag"].as_str()?.to_owned(), e["uptime_last_30m"].as_f64()))).collect())
            .unwrap_or_default();
        for r in rows {
            let Some(model) = r["model"].as_str().or(v["model"].as_str()).or(url_model) else { continue };
            let (Some(provider), Some(p), Some(c)) = (r["provider"].as_str(), num(&r["prompt"]), num(&r["completion"])) else {
                continue;
            };
            let tag = r["tag"].as_str().or(r["endpoint"].as_str());
            let uptime = match r.get("uptime_30m") {
                Some(u) => u.as_f64(),
                None => tag.and_then(|t| raw_up.get(t).copied().flatten()),
            };
            let e = OrEndpoint {
                provider: provider.to_owned(),
                quantization: r["quantization"].as_str().unwrap_or("unknown").to_owned(),
                prompt: p,
                completion: c,
                status: r["status"].as_i64().unwrap_or(0),
            };
            groups.entry(model.to_owned()).or_default().entry(at).or_default().push((e, uptime));
        }
    }
    Ok(())
}

pub(crate) fn is_up(e: &OrEndpoint) -> bool {
    up(std::slice::from_ref(e)).next().is_some()
}

/// What a request pays now (LEVEL_DEF): the level reference and the
/// `routed` baseline.
pub fn routed(eps: &[OrEndpoint], uptime: &[Option<f64>]) -> Option<f64> {
    let mut best: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for (e, u) in eps.iter().zip(uptime) {
        if !is_up(e) {
            continue;
        }
        let p = blend(e);
        let u = u.filter(|u| u.is_finite()).unwrap_or(0.0).max(0.0);
        match best.get(e.provider.as_str()) {
            Some((bp, _)) if *bp <= p => {}
            _ => {
                best.insert(&e.provider, (p, u));
            }
        }
    }
    let (ps, mut ws): (Vec<f64>, Vec<f64>) = best.values().map(|(p, u)| (*p, u / (p * p))).unzip();
    if ws.iter().filter(|w| **w > 0.0).count() < 2 {
        return None;
    }
    cap_weights(&mut ws, ROUTED_CAP);
    let (num, den) = ps.iter().zip(&ws).fold((0.0, 0.0), |(n, d), (p, w)| (n + w * p, d + w));
    Some(num / den)
}

/// Normalise to sum 1, no weight above `cap`, the excess spread over the
/// rest in proportion to their weights.
fn cap_weights(w: &mut [f64], cap: f64) {
    let total: f64 = w.iter().sum();
    for x in w.iter_mut() {
        *x /= total;
    }
    let mut capped = vec![false; w.len()];
    loop {
        let free: f64 = w.iter().zip(&capped).filter(|(_, c)| !**c).map(|(x, _)| *x).sum();
        let room = 1.0 - cap * capped.iter().filter(|c| **c).count() as f64;
        if free <= 0.0 || room <= 0.0 {
            break;
        }
        for (x, c) in w.iter_mut().zip(&capped) {
            if !*c {
                *x *= room / free;
            }
        }
        let mut more = false;
        for (x, c) in w.iter_mut().zip(capped.iter_mut()) {
            if !*c && *x > cap * (1.0 + 1e-12) {
                *c = true;
                *x = cap;
                more = true;
            }
        }
        if !more {
            break;
        }
    }
}

pub(crate) fn providers(eps: &[OrEndpoint]) -> usize {
    up(eps).map(|e| e.provider.as_str()).collect::<BTreeSet<_>>().len()
}

pub(crate) type Key = (String, String);
pub(crate) type Seen = BTreeMap<Key, (u64, u64)>;

fn key(e: &OrEndpoint) -> Key {
    (e.provider.clone(), e.quantization.clone())
}

fn bits(e: &OrEndpoint) -> (u64, u64) {
    (e.prompt.to_bits(), e.completion.to_bits())
}

/// Which rows of `sn` carry a new price: a (provider, quantization) never
/// listed before, or listed last at another price. A row relisted at the
/// price it last had, a row leaving, a status or an uptime change is
/// availability, not a new price.
pub(crate) fn fresh(seen: &Seen, sn: &Snap) -> Vec<bool> {
    sn.eps.iter().map(|e| seen.get(&key(e)) != Some(&bits(e))).collect()
}

pub(crate) fn remember(seen: &mut Seen, sn: &Snap) {
    for e in &sn.eps {
        seen.insert(key(e), bits(e));
    }
}

/// Per snapshot: some row carries a new price (`fresh`).
pub fn changes(s: &Series3) -> Vec<bool> {
    let mut seen = Seen::new();
    s.snaps
        .iter()
        .map(|sn| {
            let f = fresh(&seen, sn).into_iter().any(|x| x);
            remember(&mut seen, sn);
            f
        })
        .collect()
}

/// `now`'s prices on `prev`'s availability: each row of `prev` repriced at
/// its row in `now` (when the provider has one row in that quantization in
/// both), plus `now`'s rows that are new to `prev` and carry a new price.
/// The phantom part of a step with a price change is the rule's move from
/// this input to the real one.
pub fn counterfactual(prev: &Snap, now: &Snap, fresh_now: &[bool]) -> (Vec<OrEndpoint>, Vec<Option<f64>>) {
    let count = |sn: &Snap| {
        let mut m: BTreeMap<Key, usize> = BTreeMap::new();
        for e in &sn.eps {
            *m.entry(key(e)).or_insert(0) += 1;
        }
        m
    };
    let (cp, cn) = (count(prev), count(now));
    let by: BTreeMap<Key, &OrEndpoint> = now.eps.iter().filter(|e| cn[&key(e)] == 1).map(|e| (key(e), e)).collect();
    let (mut eps, mut upt) = (Vec::new(), Vec::new());
    for (e, u) in prev.eps.iter().zip(&prev.uptime) {
        let k = key(e);
        match by.get(&k) {
            Some(n) if cp[&k] == 1 => eps.push(OrEndpoint { prompt: n.prompt, completion: n.completion, ..e.clone() }),
            _ => eps.push(e.clone()),
        }
        upt.push(*u);
    }
    for ((e, u), f) in now.eps.iter().zip(&now.uptime).zip(fresh_now) {
        if *f && !cp.contains_key(&key(e)) {
            eps.push(e.clone());
            upt.push(*u);
        }
    }
    (eps, upt)
}

/// Lowest listed blend per provider, any status (the consensus input).
pub(crate) fn lowest(eps: &[OrEndpoint]) -> BTreeMap<&str, f64> {
    let mut m: BTreeMap<&str, f64> = BTreeMap::new();
    for e in eps.iter().filter(|e| e.prompt.is_finite() && e.completion.is_finite() && e.prompt + e.completion > 0.0 && e.prompt >= 0.0 && e.completion >= 0.0) {
        let b = blend(e);
        let x = m.entry(&e.provider).or_insert(b);
        if b < *x {
            *x = b;
        }
    }
    m
}

fn ok(m: Option<f64>) -> Option<f64> {
    m.filter(|m| m.is_finite() && *m > 0.0)
}

pub(crate) fn dist(a: Option<f64>, b: Option<f64>) -> f64 {
    match (a, b) {
        (Some(a), Some(b)) => (a / b).ln().abs(),
        (None, None) => 0.0,
        _ => PENALTY,
    }
}

/// Snapshots with 2+ up providers are scored.
pub fn scored(s: &Series3) -> Vec<usize> {
    (0..s.snaps.len()).filter(|&i| providers(&s.snaps[i].eps) >= 2).collect()
}

/// The clean replay.
pub struct Replay {
    /// Each step's effective mark (a hold keeps the last).
    pub marks: Vec<Option<f64>>,
    /// At a scored step with a new price since the previous scored step: the
    /// mark on `counterfactual` input from the rule's state before the step.
    pub cf: Vec<Option<f64>>,
    /// Steps that returned no price.
    pub holds: usize,
    /// The rule's state (and effective mark) just before each index asked for.
    pub saved: BTreeMap<usize, (Box<dyn Rule>, Option<f64>)>,
}

pub fn replay(proto: &dyn Rule, s: &Series3, save: &BTreeSet<usize>) -> Replay {
    let mut rule = proto.clone_box();
    let (mut last, mut holds) = (None, 0);
    let n = s.snaps.len();
    let (mut marks, mut cf) = (Vec::with_capacity(n), vec![None; n]);
    let mut saved = BTreeMap::new();
    let mut seen = Seen::new();
    let (mut prev, mut changed) = (None, false);
    for (i, sn) in s.snaps.iter().enumerate() {
        if save.contains(&i) {
            saved.insert(i, (rule.clone_box(), last));
        }
        let f = fresh(&seen, sn);
        changed |= f.iter().any(|x| *x);
        let is_scored = providers(&sn.eps) >= 2;
        let pre = match prev {
            Some(j) if is_scored && changed => Some((rule.clone_box(), last, j)),
            _ => None,
        };
        match ok(rule.step_v3(sn.at_ms, &sn.eps, &sn.uptime)) {
            Some(m) => last = Some(m),
            None => holds += 1,
        }
        marks.push(last);
        if let Some((mut r, pl, j)) = pre {
            let (eps, upt) = counterfactual(&s.snaps[j], sn, &f);
            cf[i] = ok(r.step_v3(sn.at_ms, &eps, &upt)).or(pl);
        }
        remember(&mut seen, sn);
        if is_scored {
            prev = Some(i);
            changed = false;
        }
    }
    Replay { marks, cf, holds, saved }
}

/// Terms from the clean replay (no seed involved). Sums are in log units.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Clean {
    pub movement: f64,
    pub phantom: f64,
    pub stale: f64,
    pub reversal: f64,
    pub days: f64,
    /// |ln(mark / routed)| per scored step with a routed price.
    pub level: Vec<f64>,
    /// Per real consensus event: max(0, 1 - |1 - mark move / consensus move|).
    pub follow: Vec<f64>,
    /// The same, only events where 2+ sellers moved in the event's direction
    /// (on this data most events are one seller's dynamic price).
    pub follow_broad: Vec<f64>,
    pub distinct: usize,
    pub holds: usize,
    pub steps: usize,
}

/// `cf` as in `Replay` (may be empty: then a step with a new price adds no
/// phantom).
pub fn clean_terms(marks: &[Option<f64>], cf: &[Option<f64>], holds: usize, s: &Series3, k: usize) -> Clean {
    let sc = scored(s);
    let chg = changes(s);
    let lm: Vec<Option<f64>> = marks.iter().map(|m| m.map(f64::ln)).collect();
    let mut c = Clean { holds, steps: s.snaps.len(), ..Default::default() };
    if let (Some(a), Some(b)) = (s.snaps.first(), s.snaps.last()) {
        c.days = ((b.at_ms - a.at_ms) as f64 / DAY_MS as f64).max(1.0 / 288.0);
    }
    c.distinct = marks.iter().flatten().map(|m| (m * 1e6).round() as i64).collect::<BTreeSet<_>>().len();
    // Movement, phantom, stale, between consecutive scored steps.
    let mut d = vec![0.0; sc.len()];
    for q in 1..sc.len() {
        let (j, i) = (sc[q - 1], sc[q]);
        if let (Some(a), Some(b)) = (lm[j], lm[i]) {
            d[q] = b - a;
            let x = d[q].abs();
            c.movement += x;
            // Phantom: all of a move with no new price behind it (this takes
            // in a hold expiring); at a step with a new price, the part the
            // availability change made (real mark vs `counterfactual`).
            if !chg[j + 1..=i].iter().any(|x| *x) {
                c.phantom += x;
            } else if let Some(m) = cf.get(i).copied().flatten() {
                c.phantom += (b - m.ln()).abs();
            }
            if s.snaps[j].eps == s.snaps[i].eps && s.snaps[j].uptime == s.snaps[i].uptime {
                c.stale += x;
            }
        }
    }
    // Reversal: the part of each move undone within the next k scored steps.
    for q in 1..sc.len() {
        let (Some(here), true) = (lm[sc[q]], d[q] != 0.0) else { continue };
        let sign = d[q].signum();
        let undo = (q + 1..=(q + k).min(sc.len() - 1)).filter_map(|h| lm[sc[h]]).map(|x| -sign * (x - here)).fold(0.0, f64::max);
        c.reversal += undo.min(d[q].abs());
    }
    // Level.
    for &i in &sc {
        if let Some(r) = routed(&s.snaps[i].eps, &s.snaps[i].uptime) {
            c.level.push(dist(marks[i], Some(r)));
        }
    }
    // Follow, real consensus moves.
    let low: Vec<BTreeMap<&str, f64>> = s.snaps.iter().map(|sn| lowest(&sn.eps)).collect();
    for q in 1..sc.len().saturating_sub(FOLLOW_N) {
        let (b, j, i, e) = (sc[(q - 1).saturating_sub(FOLLOW_N)], sc[q - 1], sc[q], sc[q + FOLLOW_N]);
        let common: Vec<&str> = low[j].keys().copied().filter(|p| low[i].contains_key(p) && low[e].contains_key(p) && low[b].contains_key(p)).collect();
        if common.is_empty() {
            continue;
        }
        let mv = |from: usize, t: usize| common.iter().map(|p| (low[t][p] / low[from][p]).ln()).sum::<f64>() / common.len() as f64;
        let (c1, cn, cb) = (mv(j, i), mv(j, e), mv(b, e));
        // Real, lasting and new: half of it still there FOLLOW_N steps on, and
        // against FOLLOW_N steps before (a promo ending is not a move).
        if c1.abs() < FOLLOW_MIN || c1 * cn <= 0.0 || cn.abs() < 0.5 * c1.abs() || c1 * cb <= 0.0 || cb.abs() < 0.5 * c1.abs() {
            continue;
        }
        let score = match (lm[j], lm[e]) {
            (Some(a), Some(b)) => (1.0 - (1.0 - (b - a) / cn).abs()).max(0.0),
            _ => 0.0,
        };
        c.follow.push(score);
        if common.iter().filter(|p| (low[e][*p] / low[j][*p]).ln() * cn > 0.0).count() >= 2 {
            c.follow_broad.push(score);
        }
    }
    c
}

/// Attack anchors for a seed: one scored snapshot with 3+ up providers per
/// ANCHOR_EVERY_MS, from a seeded offset, each with ANCHOR_TAIL_MS of data after.
pub fn anchors(s: &Series3, seed: u64) -> Vec<usize> {
    let (Some(first), Some(last)) = (s.snaps.first(), s.snaps.last()) else { return vec![] };
    let ok_at: Vec<usize> = scored(s).into_iter().filter(|&i| providers(&s.snaps[i].eps) >= 3).collect();
    let mut t = first.at_ms + splitmix(seed ^ splitmix(fnv(&s.model))) % ANCHOR_EVERY_MS;
    let mut out: Vec<usize> = Vec::new();
    while t + ANCHOR_TAIL_MS <= last.at_ms {
        if let Some(&i) = ok_at.iter().find(|&&i| s.snaps[i].at_ms >= t) {
            if s.snaps[i].at_ms + ANCHOR_TAIL_MS <= last.at_ms && out.last() != Some(&i) {
                out.push(i);
            }
        }
        t += ANCHOR_EVERY_MS;
    }
    out
}

/// One injected attack.
#[derive(Clone, Debug)]
pub struct Inj {
    pub sc: usize,
    pub k: usize,
    pub q: usize,
    pub victims: Vec<String>,
    pub fakes: Vec<OrEndpoint>,
}

fn shuffle<T>(v: &mut [T], mut r: u64) -> u64 {
    for i in (1..v.len()).rev() {
        r = splitmix(r);
        v.swap(i, (r % (i as u64 + 1)) as usize);
    }
    r
}

/// Plan scenario `sc` with `k` colluders in precision mode `q` on the anchor
/// snapshot; None if the model has too few eligible sellers.
pub fn plan(sc: usize, k: usize, q: usize, at: &Snap, rng: u64, pool: &[String], listed: &BTreeSet<&str>) -> Option<Inj> {
    let mult = SCENARIOS[sc].2;
    let mut provs: Vec<String> = up(&at.eps).map(|e| e.provider.clone()).collect::<BTreeSet<_>>().into_iter().collect();
    let mut inj = Inj { sc, k, q, victims: vec![], fakes: vec![] };
    match sc {
        0 | 1 => {
            let cheap = up(&at.eps).min_by(|a, b| blend(a).total_cmp(&blend(b)))?;
            let mut free: Vec<&String> = pool.iter().filter(|p| !listed.contains(p.as_str())).collect();
            let r = shuffle(&mut free, rng);
            for n in 0..k {
                let name = free.get(n).map(|s| s.to_string()).unwrap_or_else(|| format!("fake{n}-{:08x}", splitmix(r ^ n as u64) >> 32));
                let quant = if QUANTS[q] == "bf16" { "bf16".to_owned() } else { cheap.quantization.clone() };
                inj.fakes.push(OrEndpoint { provider: name, quantization: quant, prompt: cheap.prompt * mult, completion: cheap.completion * mult, status: 0 });
            }
        }
        DOWN => {
            shuffle(&mut provs, rng);
            provs.truncate(provs.len() / 2);
            if provs.is_empty() {
                return None;
            }
            inj.victims = provs;
        }
        _ => {
            if QUANTS[q] == "bf16" {
                let b: BTreeSet<&str> = up(&at.eps).filter(|e| e.quantization == "bf16").map(|e| e.provider.as_str()).collect();
                provs.retain(|p| b.contains(p.as_str()));
            }
            if provs.len() < k {
                return None;
            }
            shuffle(&mut provs, rng);
            provs.truncate(k);
            inj.victims = provs;
        }
    }
    Some(inj)
}

fn decay(u: Option<f64>, since_ms: u64) -> Option<f64> {
    u.map(|u| u.min(100.0 * (1.0 - since_ms as f64 / UPTIME_DECAY_MS as f64)).max(0.0))
}

/// The snapshot as the attack started at `t0` shows it.
pub fn inject(inj: &Inj, t0: u64, sn: &Snap) -> (Vec<OrEndpoint>, Vec<Option<f64>>) {
    let (mut eps, mut upt) = (sn.eps.clone(), sn.uptime.clone());
    let el = sn.at_ms.saturating_sub(t0);
    let (_, dur, mult) = SCENARIOS[inj.sc];
    if el >= dur {
        return (eps, upt);
    }
    for f in &inj.fakes {
        eps.push(f.clone());
        upt.push(Some(100.0));
    }
    for (e, u) in eps.iter_mut().zip(upt.iter_mut()) {
        if !inj.victims.contains(&e.provider) {
            continue;
        }
        if inj.sc != DOWN {
            e.prompt *= mult;
            e.completion *= mult;
        }
        // newlow: listed up for one step, then down with the new price.
        if inj.sc == DOWN || (inj.sc == 4 && el > 0) {
            e.status = 1;
            *u = decay(*u, el);
        }
    }
    (eps, upt)
}

/// One attack's result: the largest |ln(attacked / clean)| in its window.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub sc: usize,
    pub k: usize,
    pub q: usize,
    pub mv: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeedRun {
    pub hits: Vec<Hit>,
    /// Per anchor: the injected genuine move's follow score.
    pub follow_inj: Vec<f64>,
}

#[allow(clippy::type_complexity)]
pub fn seed_run(s: &Series3, seed: u64, marks: &[Option<f64>], saved: &BTreeMap<usize, (Box<dyn Rule>, Option<f64>)>, pool: &[String]) -> SeedRun {
    let listed: BTreeSet<&str> = s.snaps.iter().flat_map(|sn| sn.eps.iter().map(|e| e.provider.as_str())).collect();
    let mut out = SeedRun::default();
    for a in anchors(s, seed) {
        let t0 = s.snaps[a].at_ms;
        let rng = |x: u64| splitmix(seed ^ splitmix(fnv(&s.model) ^ splitmix(t0 ^ splitmix(x))));
        for (sc, &(_, dur, _)) in SCENARIOS.iter().enumerate() {
            for k in 1..=3 {
                for q in 0..QUANTS.len() {
                    // Down takes half the sellers: one run, any precision.
                    if sc == DOWN && (k != 3 || q != 0) {
                        continue;
                    }
                    let Some(inj) = plan(sc, k, q, &s.snaps[a], rng((sc * 16 + k * 4 + q) as u64), pool, &listed) else { continue };
                    let (st, l) = &saved[&a];
                    let (mut r, mut last) = (st.clone_box(), *l);
                    let mut mv: f64 = 0.0;
                    for (i, sn) in s.snaps.iter().enumerate().skip(a) {
                        if sn.at_ms >= t0 + dur + AFTER_MS {
                            break;
                        }
                        let (eps, upt) = inject(&inj, t0, sn);
                        if let Some(m) = ok(r.step_v3(sn.at_ms, &eps, &upt)) {
                            last = Some(m);
                        }
                        mv = mv.max(dist(last, marks[i]));
                    }
                    out.hits.push(Hit { sc, k, q, mv });
                }
            }
        }
        // Injected genuine move.
        let r0 = rng(999);
        let u = (r0 >> 11) as f64 / (1u64 << 53) as f64;
        let lm = (INJ_MOVE.0.ln() + u * (INJ_MOVE.1.ln() - INJ_MOVE.0.ln())) * if r0 & 1 == 0 { 1.0 } else { -1.0 };
        let per = |p: &str| {
            let h = splitmix(r0 ^ fnv(p));
            let u = (h >> 11) as f64 / (1u64 << 53) as f64;
            ((lm + INJ_JITTER * (2.0 * u - 1.0)).exp(), t0 + splitmix(h) % (INJ_STAGGER_MS + 1))
        };
        let (st, l) = &saved[&a];
        let (mut r, mut last) = (st.clone_box(), *l);
        let mut d = (0.0, 0usize);
        for (i, sn) in s.snaps.iter().enumerate().skip(a) {
            if sn.at_ms >= t0 + INJ_WINDOW_MS {
                break;
            }
            let eps: Vec<OrEndpoint> = sn
                .eps
                .iter()
                .map(|e| {
                    let (f, at) = per(&e.provider);
                    let f = if sn.at_ms >= at { f } else { 1.0 };
                    OrEndpoint { prompt: e.prompt * f, completion: e.completion * f, ..e.clone() }
                })
                .collect();
            if let Some(m) = ok(r.step_v3(sn.at_ms, &eps, &sn.uptime)) {
                last = Some(m);
            }
            if sn.at_ms >= t0 + INJ_STAGGER_MS + FOLLOW_N as u64 * 5 * MIN {
                let score = match (last, marks[i]) {
                    (Some(x), Some(c)) => (1.0 - (1.0 - (x / c).ln() / lm).abs()).max(0.0),
                    _ => 0.0,
                };
                d.0 += score;
                d.1 += 1;
            }
        }
        if d.1 > 0 {
            out.follow_inj.push(d.0 / d.1 as f64);
        }
    }
    out
}

/// One rule on one series over all seeds.
#[derive(Clone, Debug, Default)]
pub struct SeriesRun {
    pub model: String,
    pub clean: Clean,
    pub seeds: Vec<SeedRun>,
}

pub fn run_series(proto: &dyn Rule, s: &Series3, seeds: &[u64], pool: &[String], k: usize) -> SeriesRun {
    let save: BTreeSet<usize> = seeds.iter().flat_map(|&sd| anchors(s, sd)).collect();
    let r = replay(proto, s, &save);
    let clean = clean_terms(&r.marks, &r.cf, r.holds, s, k);
    let seeds = seeds.iter().map(|&sd| seed_run(s, sd, &r.marks, &r.saved, pool)).collect();
    SeriesRun { model: s.model.clone(), clean, seeds }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// The loss parts of one series (one seed, or averaged over seeds), log
/// units, lower is better. Total = phantom + reversal (per day) + level +
/// follow deficits + mean attack move.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Loss {
    pub phantom: f64,
    pub reversal: f64,
    pub level: f64,
    pub follow: f64,
    pub follow_inj: f64,
    pub attack: f64,
}

impl Loss {
    pub fn total(&self) -> f64 {
        self.phantom + self.reversal + self.level + self.follow + self.follow_inj + self.attack
    }
    fn of(c: &Clean, sd: &SeedRun) -> Loss {
        Loss {
            phantom: c.phantom / c.days,
            reversal: c.reversal / c.days,
            level: mean(&c.level),
            follow: if c.follow.is_empty() { 0.0 } else { 1.0 - mean(&c.follow) },
            follow_inj: if sd.follow_inj.is_empty() { 0.0 } else { 1.0 - mean(&sd.follow_inj) },
            attack: mean(&sd.hits.iter().map(|h| h.mv).collect::<Vec<_>>()),
        }
    }
    fn avg(v: &[Loss]) -> Loss {
        let n = v.len().max(1) as f64;
        v.iter().fold(Loss::default(), |a, l| Loss {
            phantom: a.phantom + l.phantom / n,
            reversal: a.reversal + l.reversal / n,
            level: a.level + l.level / n,
            follow: a.follow + l.follow / n,
            follow_inj: a.follow_inj + l.follow_inj / n,
            attack: a.attack + l.attack / n,
        })
    }
}

pub const TERMS: [&str; 7] = ["total", "phantom", "reversal", "level", "follow", "follow_inj", "attack"];

fn part(l: &Loss, t: usize) -> f64 {
    [l.total(), l.phantom, l.reversal, l.level, l.follow, l.follow_inj, l.attack][t]
}

/// Units for the standard error: one per series (seeds averaged), or one
/// per seed when there is a single series.
pub fn units(runs: &[SeriesRun]) -> Vec<Loss> {
    if runs.len() >= 2 {
        runs.iter().map(|r| Loss::avg(&r.seeds.iter().map(|sd| Loss::of(&r.clean, sd)).collect::<Vec<_>>())).collect()
    } else {
        runs.iter().flat_map(|r| r.seeds.iter().map(|sd| Loss::of(&r.clean, sd))).collect()
    }
}

/// Mean and standard error (sample sd / sqrt n) of paired differences.
pub fn mean_se(d: &[f64]) -> (f64, f64) {
    let n = d.len() as f64;
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let m = d.iter().sum::<f64>() / n;
    if d.len() < 2 {
        return (m, 0.0);
    }
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
    (m, (var / n).sqrt())
}

/// Pooled numbers over a rule's runs, for the term lines and the gates.
#[derive(Clone, Debug, Default)]
pub struct Pooled {
    pub movement: f64,
    pub phantom: f64,
    pub stale: f64,
    pub reversal: f64,
    pub days: f64,
    pub phantom_share: f64,
    pub follow: (f64, usize),
    pub follow_inj: (f64, usize),
    pub follow_broad: (f64, usize),
    pub level_mean: f64,
    pub level_median: f64,
    /// (scenario, k, q) -> (max, mean, n)
    pub attack: BTreeMap<(usize, usize, usize), (f64, f64, usize)>,
    pub distinct_median: usize,
    pub distinct_bench: Option<usize>,
    pub holds: usize,
    pub steps: usize,
}

pub fn pool(runs: &[SeriesRun]) -> Pooled {
    let mut p = Pooled::default();
    let (mut lv, mut fo, mut fi, mut fb, mut ds) = (vec![], vec![], vec![], vec![], vec![]);
    let mut hits: BTreeMap<(usize, usize, usize), Vec<f64>> = BTreeMap::new();
    for r in runs {
        let c = &r.clean;
        p.movement += c.movement;
        p.phantom += c.phantom;
        p.stale += c.stale;
        p.reversal += c.reversal;
        p.days += c.days;
        p.holds += c.holds;
        p.steps += c.steps;
        lv.extend(c.level.iter().copied());
        fo.extend(c.follow.iter().copied());
        fb.extend(c.follow_broad.iter().copied());
        ds.push(c.distinct);
        if r.model == BENCHMARK {
            p.distinct_bench = Some(c.distinct);
        }
        for sd in &r.seeds {
            fi.extend(sd.follow_inj.iter().copied());
            for h in &sd.hits {
                hits.entry((h.sc, h.k, h.q)).or_default().push(h.mv);
            }
        }
    }
    p.phantom_share = if p.movement > 0.0 { p.phantom / p.movement } else { 0.0 };
    p.follow = (mean(&fo), fo.len());
    p.follow_inj = (mean(&fi), fi.len());
    p.follow_broad = (mean(&fb), fb.len());
    p.level_mean = mean(&lv);
    lv.sort_by(f64::total_cmp);
    p.level_median = if lv.is_empty() { 0.0 } else { lv[lv.len() / 2] };
    ds.sort();
    p.distinct_median = ds.get(ds.len() / 2).copied().unwrap_or(0);
    p.attack = hits.into_iter().map(|(key, v)| (key, (v.iter().copied().fold(0.0, f64::max), mean(&v), v.len()))).collect();
    p
}

/// The worst attack for k colluders ("down" counts as k = 3): (max, which).
pub fn attack_worst(p: &Pooled, k: usize) -> (f64, String) {
    let mut w = (0.0, "none".to_owned());
    for (&(sc, kk, q), &(mx, _, _)) in &p.attack {
        let kk = if sc == DOWN { 3 } else { kk };
        if kk == k && mx > w.0 {
            w = (mx, format!("{} q={}", SCENARIOS[sc].0, QUANTS[q]));
        }
    }
    w
}

/// Run everything and print the report. Returns the lines (also printed).
#[allow(clippy::too_many_arguments)]
pub fn report(cand: &dyn Rule, base: &dyn Rule, rule: &str, base_name: &str, series: &[Series3], manifest: &str, data: &str, split: Split, seeds: &[u64], k: usize) -> Result<Vec<String>> {
    let pick: Vec<&Series3> = series.iter().filter(|s| in_split(&s.model, split)).collect();
    if pick.is_empty() {
        bail!("{split:?} split has no series");
    }
    if seeds.is_empty() {
        bail!("no seeds");
    }
    // Fakes borrow names of providers listed for other models.
    let names: Vec<String> = series.iter().flat_map(|s| s.snaps.iter().flat_map(|sn| sn.eps.iter().map(|e| e.provider.clone()))).collect::<BTreeSet<_>>().into_iter().collect();
    let run = |r: &dyn Rule| -> Result<Vec<SeriesRun>> {
        let out: Vec<SeriesRun> = pick.iter().map(|s| run_series(r, s, seeds, &names, k)).collect();
        if let Some(why) = r.disqualified() {
            bail!("disqualified: {why}");
        }
        Ok(out)
    };
    let c = run(cand).map_err(|e| anyhow::anyhow!("{e:#} (candidate)"))?;
    let b = run(base).map_err(|e| anyhow::anyhow!("{e:#} (base)"))?;
    let (uc, ub) = (units(&c), units(&b));
    let (pc, pb) = (pool(&c), pool(&b));
    let mut out = Vec::new();
    let unit = if c.len() >= 2 { "series" } else { "seeds" };
    let seeds_s = format!("{}..{}", seeds[0], seeds[seeds.len() - 1]);
    out.push(format!(
        "CHECK v3 rule {rule} base {base_name} split {split:?} series {} seeds {seeds_s} reversal_k {k} data {data} manifest {manifest}",
        pick.len()
    ));
    out.push(format!("LEVEL_DEF {LEVEL_DEF}"));
    let gain = |t: usize| mean_se(&ub.iter().zip(&uc).map(|(b, c)| part(b, t) - part(c, t)).collect::<Vec<_>>());
    let (g, se) = gain(0);
    out.push(format!("METRIC total_gain_vs_{base_name} {g:+.6} se {se:.6} n {} ({unit}; positive = better than {base_name})", uc.len()));
    let cl = Loss::avg(&uc);
    let bl = Loss::avg(&ub);
    let notes = [
        "",
        "log/day of mark movement made by availability alone (no new listed price); lower better",
        "log/day of movement undone within reversal_k steps; lower better",
        "mean |ln(mark/routed)|; lower better",
        "1 - follow of real consensus moves; lower better",
        "1 - follow of the injected all-seller move; lower better",
        "mean over attacks of max |ln(attacked/clean mark)|; lower better",
    ];
    for t in 1..TERMS.len() {
        let (g, se) = gain(t);
        out.push(format!("TERM {} {:.6} base {:.6} gain {g:+.6} se {se:.6}  # {}", TERMS[t], part(&cl, t), part(&bl, t), notes[t]));
    }
    let per_day = |x: f64, p: &Pooled| if p.days > 0.0 { x / p.days } else { 0.0 };
    out.push(format!("TERM movement {:.6} base {:.6}  # log/day, all mark movement", per_day(pc.movement, &pc), per_day(pb.movement, &pb)));
    out.push(format!("TERM phantom_share {:.4} base {:.4}  # phantom / movement, pooled", pc.phantom_share, pb.phantom_share));
    out.push(format!("TERM stale {:.6} base {:.6}  # log/day moved at steps whose input is identical to the step before (hold expiry); already inside phantom", per_day(pc.stale, &pc), per_day(pb.stale, &pb)));
    out.push(format!("TERM follow_real {:.4} base {:.4} events {}  # share of real consensus moves shown {FOLLOW_N} steps later; higher better", pc.follow.0, pb.follow.0, pc.follow.1));
    out.push(format!("TERM follow_broad {:.4} base {:.4} events {}  # follow_real on events where 2+ sellers moved; advisory", pc.follow_broad.0, pb.follow_broad.0, pc.follow_broad.1));
    out.push(format!("TERM follow_inj {:.4} base {:.4} anchors {}  # share of an injected x{}-x{} all-seller move shown 30 min after; higher better", pc.follow_inj.0, pb.follow_inj.0, pc.follow_inj.1, INJ_MOVE.0, INJ_MOVE.1));
    out.push(format!("TERM level_median {:.4} base {:.4}  # median |ln(mark/routed)|", pc.level_median, pb.level_median));
    for (&(sc, kk, q), &(mx, mn, n)) in &pc.attack {
        let bm = pb.attack.get(&(sc, kk, q)).map(|x| x.0).unwrap_or(0.0);
        let ks = if sc == DOWN { "half".to_owned() } else { kk.to_string() };
        out.push(format!("ATTACK {} k={ks} q={} max {mx:.4} mean {mn:.4} n {n} base_max {bm:.4}", SCENARIOS[sc].0, QUANTS[q]));
    }
    let pf = |b: bool| if b { "PASS" } else { "FAIL" };
    for k in 1..=3 {
        let (mx, which) = attack_worst(&pc, k);
        let lim = GATE_ATTACK_MAX[k - 1];
        out.push(format!("GATE attack_k{k} {} {mx:.4} (limit {lim}) worst {which}", pf(mx <= lim)));
    }
    out.push(format!("GATE phantom_share {} {:.4} (limit {GATE_PHANTOM_SHARE})", pf(pc.phantom_share <= GATE_PHANTOM_SHARE), pc.phantom_share));
    let fr = pc.follow.1 > 0 && pc.follow.0 >= GATE_FOLLOW;
    out.push(format!("GATE follow {} {:.4} (limit {GATE_FOLLOW}) events {}", pf(fr), pc.follow.0, pc.follow.1));
    let fi = pc.follow_inj.1 > 0 && pc.follow_inj.0 >= GATE_FOLLOW_INJ;
    out.push(format!("GATE follow_inj {} {:.4} (limit {GATE_FOLLOW_INJ})", pf(fi), pc.follow_inj.0));
    out.push(format!("GATE level_median {} {:.4} (limit {GATE_LEVEL_MEDIAN})", pf(pc.level_median <= GATE_LEVEL_MEDIAN), pc.level_median));
    let note = if split == Split::Heldout { "" } else { " (counts only on --split heldout)" };
    out.push(format!("GATE gain_vs_{base_name} {} {g:+.6} (limit {GATE_GAIN_SE} x se = {:.6}){note}", pf(g > GATE_GAIN_SE * se), GATE_GAIN_SE * se));
    out.push(format!(
        "distinct {}  # median distinct marks per series; benchmark {}; holds {}/{} steps",
        pc.distinct_median,
        pc.distinct_bench.map(|d| d.to_string()).unwrap_or_else(|| "-".into()),
        pc.holds,
        pc.steps
    ));
    Ok(out)
}

#[cfg(test)]
mod tests;
