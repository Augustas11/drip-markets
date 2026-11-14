//! Check v4 on tiny synthetic series with hand-computed answers.

use super::*;
use crate::score::rules::{by_name, Mean};

const T0: u64 = 1_794_355_200_000; // 2026-11-11 00:00 UTC

/// An endpoint whose 15:1 blend is `b` USD per 1M tokens.
fn ep(p: &str, b: f64, status: i64) -> OrEndpoint {
    OrEndpoint { provider: p.into(), quantization: "bf16".into(), prompt: b * 1e-6, completion: b * 1e-6, status }
}

const U: Option<f64> = Some(100.0);

fn snap(i: u64, rows: &[(&str, f64, i64, Option<f64>)]) -> Snap {
    Snap { at_ms: T0 + i * STEP, eps: rows.iter().map(|r| ep(r.0, r.1, r.2)).collect(), uptime: rows.iter().map(|r| r.3).collect() }
}

fn series(model: &str, snaps: Vec<Snap>) -> Series3 {
    Series3 { model: model.into(), snaps }
}

/// Four sellers at 1.0, all serving, `n` snapshots.
fn flat4(n: u64) -> Series3 {
    let rows = [("A", 1.0, 0, U), ("B", 1.0, 0, U), ("C", 1.0, 0, U), ("D", 1.0, 0, U)];
    series("test/flat", (0..n).map(|i| snap(i, &rows)).collect())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn a_mark_is_a_price_a_hold_or_invalid() {
    assert_eq!(classify(Some(2.0)), Out::Price(2.0));
    assert_eq!(classify(None), Out::Hold);
    assert_eq!(classify(Some(f64::NAN)), Out::Invalid);
    assert_eq!(classify(Some(f64::INFINITY)), Out::Invalid);
    assert_eq!(classify(Some(0.0)), Out::Invalid);
    assert_eq!(classify(Some(-1.0)), Out::Invalid);
}

#[test]
fn r60_is_the_median_of_the_last_13_routed_prices() {
    // Two sellers at the same price p_i: routed is p_i (equal weights).
    let p: Vec<f64> = (0..20).map(|i| [3.0, 1.0, 2.0, 5.0, 4.0][i % 5] * (1.0 + i as f64 / 100.0)).collect();
    let s = series("test/r60", (0..20).map(|i| snap(i as u64, &[("A", p[i], 0, U), ("B", p[i], 0, U)])).collect());
    let r = r60(&s);
    for i in 0..20usize {
        let lo = i.saturating_sub(12);
        let want = median(p[lo..=i].to_vec()).unwrap();
        assert!(close(r[i].unwrap(), want), "step {i}: {:?} vs {want}", r[i]);
    }
    // Step 1: two values, the mean of them.
    assert!(close(r[1].unwrap(), (3.0 + 1.01) / 2.0));
    // The baseline rule is the same thing.
    let mut b = by_name("routed-tm60").unwrap();
    for (i, sn) in s.snaps.iter().enumerate() {
        assert_eq!(b.step_v3(sn.at_ms, &sn.eps, &sn.uptime), r[i]);
    }
    // Jitter: 301 s apart, the window still holds 13 snapshots.
    let mut b = by_name("routed-tm60").unwrap();
    let mut last = None;
    for i in 0..13u64 {
        let sn = Snap { at_ms: T0 + i * 301_000, eps: vec![ep("A", (i + 1) as f64, 0), ep("B", (i + 1) as f64, 0)], uptime: vec![U, U] };
        last = b.step_v3(sn.at_ms, &sn.eps, &sn.uptime);
    }
    assert_eq!(last, Some(7.0));
}

#[test]
fn proven_is_the_first_snapshot_or_24h_of_serving() {
    // A, B from the start; C serving from step 1; D listed from step 1 but down.
    let s = series(
        "test/proven",
        (0..300u64)
            .map(|i| if i == 0 { snap(0, &[("A", 1.0, 0, U), ("B", 1.0, 1, None)]) } else { snap(i, &[("A", 1.0, 0, U), ("B", 1.0, 0, U), ("C", 1.0, 0, U), ("D", 1.0, 1, U)]) })
            .collect(),
    );
    let at: BTreeSet<usize> = [1, 288, 289, 299].into_iter().collect();
    let p = proven_at(&s, &at);
    let names = |i: usize| p[&i].iter().cloned().collect::<Vec<_>>().join(",");
    // C has served steps 1..i-1, 5 min each: 24 h = 288 steps, so from step 289.
    assert_eq!(names(1), "A,B");
    assert_eq!(names(288), "A,B");
    assert_eq!(names(289), "A,B,C");
    assert_eq!(names(299), "A,B,C");
    // Null or zero uptime is not serving.
    assert!(!serving(&ep("X", 1.0, 0), None));
    assert!(!serving(&ep("X", 1.0, 0), Some(0.0)));
    assert!(!serving(&ep("X", 1.0, 1), U));
    assert!(serving(&ep("X", 1.0, 0), Some(0.5)));
}

#[test]
fn liveliness_terms_by_hand() {
    // L1: 30 h of marks; the first 24 h window ranges 2.0..2.5 (25%); the
    // last 6 h (under 12 h) does not count.
    let at: Vec<u64> = (0..360u64).map(|i| T0 + i * STEP).collect();
    let m: Vec<Option<f64>> = (0..360).map(|i| Some(if i < 288 { if i == 100 { 2.5 } else { 2.0 } } else { 9.0 })).collect();
    assert!(close(daily_range(&at, &m).unwrap(), 0.25));
    // Under 12 h in every window: the whole series.
    assert!(close(daily_range(&at[..10], &[Some(1.0), Some(1.2), Some(0.8), Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0)]).unwrap(), 0.5));
    // L2: hourly means; b = 3 a gives the same log moves: 1; the inverse: -1;
    // a constant mark: 0.
    let a: Vec<Option<f64>> = (0..360).map(|i| Some(1.0 + ((i / 12) % 5) as f64 + 0.01 * (i % 12) as f64)).collect();
    let b3: Vec<Option<f64>> = a.iter().map(|x| x.map(|x| 3.0 * x)).collect();
    let inv: Vec<Option<f64>> = a.iter().map(|x| x.map(|x| 1.0 / x)).collect();
    assert!(close(hourly_corr(&at, &a, &b3).unwrap(), 1.0));
    assert!(close(hourly_corr(&at, &a, &inv).unwrap(), -1.0));
    assert_eq!(hourly_corr(&at, &vec![Some(2.0); 360], &a), Some(0.0));
    // L3 and the medians.
    let co = |model: &str, range: f64, corr: f64, provs: usize, dead: bool, live: bool| CleanOut { model: model.into(), range: Some(range), corr: Some(corr), providers_ever: provs, dead, routed_live: live, ..Default::default() };
    let l = live(&[co("a/1", 0.10, 0.9, 9, false, true), co("a/2", 0.02, 0.1, 5, true, true), co("a/3", 0.0, 0.0, 9, true, false), co(v3::BENCHMARK, 0.04, 0.6, 20, false, true)]);
    // Ranges 0, 0.02, 0.04, 0.10: median (0.02 + 0.04) / 2.
    assert!(close(l.l1_median, 0.03));
    assert_eq!(l.l1_bench, Some(0.04));
    // Correlation over 8+ provider models only: 0.9, 0.0, 0.6 -> 0.6.
    assert_eq!((l.l2_median, l.l2_n), (Some(0.6), 3));
    // Dead among models where routed moved: a/2 of 3.
    assert_eq!((l.dead, l.eligible), (1, 3));
    assert!(close(l.l3_share, 1.0 / 3.0));
}

/// A rule that goes NaN for good after it sees a price under 1e-100: an
/// earlier clamp rule's bug in one line.
#[derive(Clone)]
struct Poison(bool);
impl Rule for Poison {
    fn step(&mut self, a: u64, eps: &[OrEndpoint]) -> Option<f64> {
        self.0 |= eps.iter().any(|e| e.prompt + e.completion < 1e-100);
        let m = Mean.step(a, eps);
        if self.0 {
            Some(f64::NAN)
        } else {
            m
        }
    }
    fn clone_box(&self) -> Box<dyn Rule> {
        Box::new(self.clone())
    }
}

/// The mark moves by the mean log move of the sellers listed (any status)
/// at both steps: a seller vanishing and reappearing at another price is
/// never counted on the way out. The ratchet in one rule.
#[derive(Clone, Default)]
struct Chain {
    last: BTreeMap<String, f64>,
    index: f64,
}
impl Rule for Chain {
    fn step(&mut self, _: u64, eps: &[OrEndpoint]) -> Option<f64> {
        let now: BTreeMap<String, f64> = eps.iter().map(|e| (e.provider.clone(), blend(e).ln())).collect();
        let mv: Vec<f64> = now.iter().filter_map(|(p, x)| self.last.get(p).map(|l| x - l)).collect();
        if !mv.is_empty() {
            self.index += mv.iter().sum::<f64>() / mv.len() as f64;
        }
        self.last = now;
        Some(self.index.exp())
    }
    fn clone_box(&self) -> Box<dyn Rule> {
        Box::new(self.clone())
    }
}

/// A seed whose anchor leaves `tail` of data after it.
fn seed_with_tail(s: &Series3, tail: u64) -> (u64, usize) {
    (1..200).find_map(|sd| anchors(s, sd).first().copied().filter(|&a| s.snaps.last().unwrap().at_ms - s.snaps[a].at_ms >= tail).map(|a| (sd, a))).unwrap()
}

fn ub(b: Ub) -> usize {
    behaviours().iter().position(|x| *x == b).unwrap()
}

#[test]
fn u_moves_a_mean_by_the_hand_computed_amount_and_catches_nan_and_ratchets() {
    let s = flat4(144);
    let (sd, a) = seed_with_tail(&s, 2 * HOUR);
    assert!(a >= (WARMUP_MS / STEP) as usize, "anchors leave an hour of warm-up");
    let mean = attacks(&Mean, &s, sd, &[]).unwrap();
    let hit = |o: &AttackOut, b: Ub, k: usize| o.u.iter().find(|h| h.b == ub(b) && h.k == k).unwrap().clone();
    // One live fake at the cheapest x0.5 among four sellers at 1: mean 0.9.
    let h = hit(&mean, Ub::Const { base: Base::Cheap, mult: 0.5, mode: Mode::Live, bf16: false }, 1);
    assert!(close(h.mv, (1.0f64 / 0.9).ln()), "{}", h.mv);
    // Eight: (4 + 4) / 12 -> 2/3.
    let h = hit(&mean, Ub::Const { base: Base::Cheap, mult: 0.5, mode: Mode::Live, bf16: false }, 8);
    assert!(close(h.mv, 1.5f64.ln()), "{}", h.mv);
    // Idle fakes (down) do not enter a mean of up sellers.
    assert_eq!(hit(&mean, Ub::Const { base: Base::Cheap, mult: 0.1, mode: Mode::Idle, bf16: false }, 8).mv, 0.0);
    // Extreme rows that `up` rejects move nothing and break nothing.
    for b in [Ub::Zero, Ub::Neg, Ub::Nan] {
        let h = hit(&mean, b, 3);
        assert_eq!((h.mv, h.bad), (0.0, 0), "{b:?}");
    }
    // One 1e-160 row, down, for one snapshot: Poison goes NaN for good. v3
    // held the last mark (move 0); v4 counts every step from the anchor.
    let p = attacks(&Poison(false), &s, sd, &[]).unwrap();
    let h = hit(&p, Ub::Tiny1, 1);
    assert_eq!((h.mv, h.bad), (0.0, 144 - a));
    let g = gates(&RuleRun { clean: vec![clean(&Poison(false), &s)], att: vec![p] }, &Live::default());
    let uf = g.iter().find(|g| g.name == "U_finite").unwrap();
    assert!(!uf.pass && uf.value > 0.0);
    // The ratchet: gone, x0.9, x1.1, repeating. Chain counts the x0.9 -> x1.1
    // step (5 sellers listed at both) and never the way out:
    // +ln(1.1/0.9)/5 per 3 snapshots.
    let c = attacks(&Chain::default(), &s, sd, &[]).unwrap();
    let h = hit(&c, Ub::Ratchet { up: true, mode: Mode::Idle }, 1);
    let cycles = (144 - a) / 3;
    assert!(close(h.mv, cycles as f64 * (1.1f64 / 0.9).ln() / 5.0), "{} vs {cycles} cycles", h.mv);
    assert!(h.mv > GATE_U);
}

#[test]
fn fakes_list_what_the_behaviour_says() {
    let c = Ctx { cheap: ep("A", 1.0, 0), dear: ep("B", 4.0, 0), mark: 2.0 };
    let b = |r: &[(OrEndpoint, Option<f64>)]| r.iter().map(|(e, _)| blend(e)).collect::<Vec<_>>();
    let r = Ub::Ratchet { up: true, mode: Mode::Idle };
    assert!(fake_rows(&r, &c, "Z", 3, 15 * MIN).is_empty());
    assert!(close(b(&fake_rows(&r, &c, "Z", 4, 20 * MIN))[0], 1.8));
    assert!(close(b(&fake_rows(&r, &c, "Z", 5, 25 * MIN))[0], 2.2));
    assert_eq!(fake_rows(&r, &c, "Z", 4, 0)[0].0.status, 1);
    // Sleeper: at the mark while quiet, then the jump.
    let sl = Ub::Sleeper { quiet_ms: 2 * HOUR, jump: 0.5, walk: 0.0 };
    assert!(close(b(&fake_rows(&sl, &c, "Z", 23, 115 * MIN))[0], 2.0));
    assert!(close(b(&fake_rows(&sl, &c, "Z", 24, 120 * MIN))[0], 1.0));
    // Dear x10 in bf16, live, uptime 100.
    let r = fake_rows(&Ub::Const { base: Base::Dear, mult: 10.0, mode: Mode::Live, bf16: true }, &c, "Z", 0, 0);
    assert_eq!((r[0].0.quantization.as_str(), r[0].0.status, r[0].1), ("bf16", 0, Some(100.0)));
    assert!(close(blend(&r[0].0), 40.0));
    // Extremes, duplicates, quantizations.
    assert_eq!(fake_rows(&Ub::Tiny1, &c, "Z", 1, STEP).len(), 0);
    assert!(fake_rows(&Ub::Nan, &c, "Z", 0, 0)[0].1.unwrap().is_nan());
    assert_eq!(fake_rows(&Ub::Dup, &c, "Z", 0, 0).len(), 4);
    let q = fake_rows(&Ub::ManyQ, &c, "Z", 3, 0);
    assert_eq!(q.iter().map(|(e, _)| e.quantization.clone()).collect::<BTreeSet<_>>().len(), 8);
    let cheapest = q.iter().min_by(|x, y| blend(&x.0).total_cmp(&blend(&y.0))).unwrap();
    assert_eq!(cheapest.0.quantization, "q5");
}

#[test]
fn s_p_m_f_move_a_mean_by_the_hand_computed_amount() {
    let s = flat4(144);
    let (sd, _) = seed_with_tail(&s, 5 * HOUR);
    let o = attacks(&Mean, &s, sd, &[]).unwrap();
    // S: one seller x10 for one snapshot: (10 + 3) / 4; two at x0.5: 3/4.
    let sh = |k: usize, f: f64, n: usize| o.s.iter().filter(|h| h.k == k && h.f == f && h.steps == n).map(|h| h.mv).fold(0.0, f64::max);
    assert!(close(sh(1, 10.0, 1), 3.25f64.ln()));
    assert!(close(sh(2, 0.5, 6), (4.0f64 / 3.0).ln()));
    // P: one seller x2 for good: 5/4; the bound 0.25 ln 2; it fails (ratio
    // ln 1.25 / (0.25 ln 2) = 1.288), and at once (in the first 20 min).
    let p = o.p.iter().find(|h| h.k == 1 && h.f == 2.0).unwrap();
    assert!(close(p.mv, 1.25f64.ln()) && close(p.fast, 1.25f64.ln()));
    assert!(close(share_bound(1, 2.0), 0.25 * 2f64.ln()));
    assert!(close(share_bound(1, 0.5), 0.25 * 2f64.ln()));
    // k < 60% of 4: only k = 1 and 2.
    assert_eq!(o.p.iter().map(|h| h.k).collect::<BTreeSet<_>>(), [1, 2].into_iter().collect());
    let g = gates(&RuleRun { clean: vec![clean(&Mean, &s)], att: vec![o.clone()] }, &Live::default());
    let gp = g.iter().find(|g| g.name == "P").unwrap();
    // Worst ratio: k = 1 x10: ln(13/4) / (0.25 ln 10) = 2.0477.
    assert!(close(gp.value, 3.25f64.ln() / (0.25 * 10f64.ln())), "{}", gp.value);
    // M: 3 of 4 (ceil 60%) x1.3: mean (3 x 1.3 + 1) / 4 = 1.225, shown as
    // ln 1.225 / ln 1.3 of the move; all four: all of it.
    for h in &o.m {
        let want = if h.all {
            1.0
        } else {
            let r = ((3.0 * h.f + 1.0) / 4.0).ln() / h.f.ln();
            1.0 - (1.0 - r).abs()
        };
        assert!(close(h.score, want), "{h:?} want {want}");
    }
    // F: every seller x1.3 with k fakes listed at the mark. Idle: ignored,
    // all of it. Live: (4 x 1.3 + k) / (4 + k).
    for h in &o.f {
        let want = match h.mode {
            Mode::Live => (((4.0 * h.f + h.k as f64) / (4.0 + h.k as f64)).ln() / h.f.ln()).min(1.0),
            _ => 1.0,
        };
        assert!(close(h.score, want), "{h:?} want {want}");
    }
    let gf = g.iter().find(|g| g.name == "F").unwrap();
    // Worst group: k = 8 live, (5.2 + 8) / 12 = 1.1 vs 1.3 up; down by 1/1.3
    // the same shares in log: mean of the two.
    let up = (13.2f64 / 12.0).ln() / 1.3f64.ln();
    let dn = ((4.0 / 1.3 + 8.0) / 12.0f64).ln() / (1.0 / 1.3f64).ln();
    assert!(close(gf.value, (up + dn) / 2.0) && !gf.pass, "{} vs {}", gf.value, (up + dn) / 2.0);
    // Echo: a mean takes the move as the sellers do, so 3 h later there is
    // no extra step; its error is 0 when all four move and
    // |ln((3f + 1) / 4) - 0.6 ln f| when 3 of 4 (ceil 60%) do.
    assert_eq!(o.e.len(), ECHO_SETS.len() * ECHO_FACTORS.len());
    let err = |h: &EHit| if h.set == "all" { 0.0 } else { (((3.0 * h.f + 1.0) / 4.0).ln() - 0.6 * h.f.ln()).abs() };
    for h in &o.e {
        assert!(close(h.step, 0.0) && close(h.err, err(h)), "{h:?}");
    }
    let ge = g.iter().find(|g| g.name == "echo_step").unwrap();
    assert!(ge.pass && close(ge.value, 0.0));
    // The error gate reads "all" and "60%" but not 60% x0.5 (the known
    // exception) nor the random set: worst is 60% x1.3, ln 1.225 - 0.6 ln 1.3.
    let ge = g.iter().find(|g| g.name == "echo_error").unwrap();
    let want = o.e.iter().filter(|h| h.set != "60%-random" && !(h.f == 0.5 && h.set == "60%")).map(err).fold(0.0, f64::max);
    assert!(close(ge.value, want) && close(want, err(o.e.iter().find(|h| h.set == "60%" && h.f == 1.3).unwrap())) && close(want, 1.225f64.ln() - 0.6 * 1.3f64.ln()), "{} vs {want}", ge.value);
    // Four sellers: no coalition of 10 to 12.
    assert!(o.c.is_empty());
    // S90 is reported but not gated: x10 for 18 snapshots moves the mean as
    // much as for one, and the S gate reads only spikes up to 6 snapshots.
    assert!(close(sh(1, 10.0, 18), 3.25f64.ln()));
    assert!(o.s.iter().any(|h| h.steps == 18));
}

#[test]
fn gates_l1_is_relative_to_spec_and_the_benchmark_is_gated_alone() {
    let co = |model: &str, range: f64, corr: f64| CleanOut { model: model.into(), range: Some(range), corr: Some(corr), providers_ever: 9, routed_live: true, ..Default::default() };
    let spec = Live { l1_median: 0.05, l1_bench: Some(0.015), ..Default::default() };
    let run = |c: Vec<CleanOut>| gates(&RuleRun { clean: c, att: vec![] }, &spec);
    let get = |g: &[Gate], n: &str| g.iter().find(|x| x.name == n).unwrap().clone();
    // Multi-model: median 0.06 >= spec's 0.05; corr median 0.5.
    let g = run(vec![co("a/1", 0.06, 0.5), co("a/2", 0.07, 0.6), co("a/3", 0.01, 0.4)]);
    assert!(get(&g, "L1_range").pass && close(get(&g, "L1_range").value, 6.0));
    assert!(get(&g, "L2_corr").pass);
    // Benchmark alone: spec's 1.5% is under the 2% floor, so 2% it is.
    let g = run(vec![co(v3::BENCHMARK, 0.019, 0.7)]);
    assert!(!get(&g, "L1_range").pass);
    let g = run(vec![co(v3::BENCHMARK, 0.021, 0.49)]);
    assert!(get(&g, "L1_range").pass && !get(&g, "L2_corr").pass);
    // No attacks ran: M and F fail rather than pass on nothing.
    assert!(!get(&g, "M").pass && !get(&g, "F").pass);
}

#[test]
fn the_report_is_deterministic_across_threads() {
    let mk = |m: &str, n: u64, wiggle: f64| {
        series(
            m,
            (0..n)
                .map(|i| {
                    let x = 1.0 + wiggle * ((i % 7) as f64);
                    snap(i, &[("A", x, 0, U), ("B", 1.2, 0, U), ("C", 0.9 * x, 0, U), ("D", 1.5, (i % 5 == 0) as i64, Some(80.0))])
                })
                .collect(),
        )
    };
    let ss = vec![mk("test/one", 100, 0.01), mk("test/two", 90, 0.03), mk(v3::BENCHMARK, 110, 0.02)];
    let f = |n: &str| by_name(n).ok_or_else(|| anyhow!("no rule {n}"));
    let base: Vec<String> = BASELINES.iter().map(|s| s.to_string()).collect();
    let a = report("spec-median", &base, &f, &ss, "m", "d", Split::All, &[1, 2, 3], 1).unwrap();
    let b = report("spec-median", &base, &f, &ss, "m", "d", Split::All, &[1, 2, 3], 7).unwrap();
    assert_eq!(a, b);
    assert!(a.iter().any(|l| l.starts_with("GATE U ")));
    assert!(a.iter().any(|l| l.starts_with("BASE routed-tm60 GATE level_r60 PASS")));
    // An unknown baseline is an error, not a silent skip.
    assert!(report("spec", &["nope".into()], &f, &ss, "m", "d", Split::All, &[1], 2).is_err());
}
