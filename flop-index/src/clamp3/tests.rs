//! clamp3 on small synthetic markets with hand-computed answers.

use super::*;

const T0: u64 = 1_794_355_200_000; // 2026-11-11 00:00 UTC
const STEP: u64 = 300_000;
/// Steps in 24 h.
const DAY: usize = 288;
const SIX: [&str; 6] = ["A", "B", "C", "D", "E", "F"];

/// An endpoint whose 15:1 blend is `b` USD per 1M tokens.
fn ep(p: &str, b: f64) -> OrEndpoint {
    OrEndpoint { provider: p.into(), quantization: "bf16".into(), prompt: b * 1e-6, completion: b * 1e-6, status: 0 }
}

fn six(b: f64) -> Vec<OrEndpoint> {
    SIX.iter().map(|p| ep(p, b)).collect()
}

fn up(eps: &[OrEndpoint]) -> Vec<Option<f64>> {
    vec![Some(100.0); eps.len()]
}

fn listed() -> Clamp3 {
    Clamp3::with_allow_list(SIX.iter().map(|s| s.to_string()))
}

/// Marks of `rule` over `n` steps of `input(i)` (rows, uptime).
fn run(mut rule: Clamp3, n: usize, input: impl Fn(usize) -> (Vec<OrEndpoint>, Vec<Option<f64>>)) -> Vec<Option<f64>> {
    (0..n)
        .map(|i| {
            let (eps, upt) = input(i);
            rule.step(T0 + i as u64 * STEP, &eps, &upt)
        })
        .collect()
}

fn rows(eps: Vec<OrEndpoint>) -> (Vec<OrEndpoint>, Vec<Option<f64>>) {
    let u = up(&eps);
    (eps, u)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn ln(m: Option<f64>) -> f64 {
    m.expect("a mark").ln()
}

#[test]
fn six_sellers_at_one_price_mark_that_price_from_the_first_step() {
    // Equal prices, equal weights 1/6 (under the 24% cap): the mark is the price.
    let m = run(listed(), 40, |_| rows(six(1.0)));
    for x in &m {
        assert!(close(x.unwrap(), 1.0, 1e-12), "{x:?}");
    }
}

#[test]
fn unproven_listings_and_invalid_rows_move_nothing_and_never_nan() {
    let clean = run(listed(), DAY, |_| rows(six(1.0)));
    let junk = run(listed(), DAY, |_| {
        let mut v = six(1.0);
        // Junk rows from new providers and from proven A: NaN, inf, negative,
        // zero, 1e-160, 1e160 (outside the usable 1e-6..1e6 blend).
        for (i, x) in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e-160, 1e160].into_iter().enumerate() {
            v.push(OrEndpoint { prompt: x, completion: x, ..ep(&format!("J{i}"), 1.0) });
            v.push(OrEndpoint { prompt: x, completion: x, ..ep("A", 1.0) });
        }
        // Every row again (duplicates), a down row at 0.01, and an unproven
        // seller serving at 0.01 from the first step.
        v.extend(six(1.0));
        v.push(OrEndpoint { status: 2, ..ep("Z", 0.01) });
        v.push(ep("NEW", 0.01));
        // Odd uptimes on junk rows: NaN, null.
        let mut u = up(&v);
        u[6] = Some(f64::NAN);
        u[7] = None;
        (v, u)
    });
    // NEW is proven only after 24 h of serving: until then, the same bits.
    assert_eq!(clean, junk);
    assert!(junk.iter().all(|m| m.is_some_and(|m| m.is_finite() && m > 0.0)));
}

#[test]
fn the_band_counts_a_seller_at_most_three_times_or_a_third_of_the_mark() {
    // A goes to x1000 for good at step 20, or to x1e6: both count as 3 x the
    // mark, so the two runs are the same bits.
    let far = |f: f64| move |i: usize| rows(SIX.iter().map(|p| ep(p, if *p == "A" && i >= 20 { f } else { 1.0 })).collect());
    let a = run(listed(), 400, far(1000.0));
    assert_eq!(a, run(listed(), 400, far(1e6)));
    assert_eq!(run(listed(), 400, far(1e-3)), run(listed(), 400, far(1e-5)));
    // By hand, at rest: A counts at x = ln(mark) + ln 3 with weight
    // exp(-2(x + ln 3)) (its weight's price has caught up), the five others
    // at 0 with weight 1, no cap binding (each under 1/5.1 < 24%), so
    // x = ln((5 + e^-x / 3) / (5 + e^-2x / 9)): x = 0.041744...
    let mut x = 0.0f64;
    for _ in 0..100 {
        x = ((5.0 + (-x).exp() / 3.0) / (5.0 + (-2.0 * x).exp() / 9.0)).ln();
    }
    assert!(close(x, 0.041744, 1e-6), "{x}");
    assert!(close(ln(a[399]), x, 1e-6), "{} vs {x}", ln(a[399]));
    // Unclamped, x1000 would have carried the mark to about ln(1000) / 6.
    assert!(ln(a[399]) < 0.05);
}

#[test]
fn the_mark_moves_at_most_one_percent_a_step_unless_the_market_moved() {
    // One seller x1.3 for good at step 20: every step at most 0.01; at rest,
    // A at 1.3 with weight 1/1.69 and five at 1 with weight 1:
    // ln((5 + 1.3 / 1.69) / (5 + 1 / 1.69)) = 0.031291...
    let m = run(listed(), 200, |i| rows(SIX.iter().map(|p| ep(p, if *p == "A" && i >= 20 { 1.3 } else { 1.0 })).collect()));
    let steps: Vec<f64> = m.windows(2).map(|w| ln(w[1]) - ln(w[0])).collect();
    assert!(steps.iter().all(|d| d.abs() <= SPEED + 1e-12));
    // A's median of 13 shows the new price at its 7th sighting (step 26).
    assert!(close(steps[25], SPEED, 1e-12) && steps[..25].iter().all(|d| d.abs() < 1e-12), "{:?}", &steps[20..30]);
    let want = ((5.0 + 1.0 / 1.3) / (5.0 + 1.0 / 1.69f64)).ln();
    assert!(close(ln(m[199]), want, 1e-9), "{} vs {want}", ln(m[199]));
    // Every seller x1.3 at step 30: at step 36 all six medians move, the
    // detector credits 0.1, 0.1 and 0.062364 (ln 1.3 = 0.262364 in all),
    // and the mark moves that plus at most 0.01: 0.11, 0.11, then 0.052364
    // (it overshot by 0.01 and the speed limit takes 0.01 back), then -0.01.
    let m = run(listed(), 60, |i| rows(six(if i >= 30 { 1.3 } else { 1.0 })));
    let l13 = 1.3f64.ln();
    let want = [(35, 0.0), (36, 0.11), (37, 0.22), (38, 0.272364), (39, l13), (59, l13)];
    for (i, x) in want {
        assert!(close(ln(m[i]), x, 1e-6), "step {i}: {} vs {x}", ln(m[i]));
    }
}

/// Every seller moves by `f` at step 30 for good; returns the largest step
/// and the largest |ln(mark) - ln f| from 3 h after the move to 48 h after it.
fn market_move(f: f64) -> (f64, f64) {
    let n = 30 + 2 * DAY;
    let m = run(listed(), n, |i| rows(six(if i >= 30 { f } else { 1.0 })));
    let from = 30 + 36;
    let step = (from..n).map(|i| (ln(m[i]) - ln(m[i - 1])).abs()).fold(0.0, f64::max);
    let err = (from..n).map(|i| (ln(m[i]) - f.ln()).abs()).fold(0.0, f64::max);
    (step, err)
}

#[test]
fn a_market_move_is_not_echoed_back() {
    // The check's echo gates: after a market-wide move, extra 5-minute steps
    // <= 0.02 and mark error <= 0.10 over 48 h. Here the clean mark is flat,
    // so the extra step is the step. Without the echo fix, 24 steps after a
    // halving the detector reads the credited move as a rise (+0.59) and
    // the mark steps back up by 0.11.
    for f in [0.5, 1.0 / 1.3, 1.3] {
        let (step, err) = market_move(f);
        assert!(step <= 0.02 + 1e-12 && err <= 0.10, "x{f}: step {step} error {err}");
        // At rest the mark is the move.
        assert!(err < 1e-9, "x{f}: error {err}");
    }
}

#[test]
fn a_listing_at_the_first_step_is_proven_only_from_the_allow_list() {
    let with_fake = |i: usize| {
        let mut v = six(1.0);
        v.push(ep("FAKE", 0.01));
        let _ = i;
        rows(v)
    };
    let clean = run(listed(), DAY, |_| rows(six(1.0)));
    // Allow-list of the six: FAKE, listed from step 0, moves 0.000 for 24 h.
    assert_eq!(run(listed(), DAY, with_fake), clean);
    // Empty allow-list: nobody counts for 24 h, FAKE or not.
    let empty = || Clamp3::with_allow_list(Vec::<String>::new());
    let e = run(empty(), DAY, with_fake);
    assert_eq!(e, run(empty(), DAY, |_| rows(six(1.0))));
    assert!(e.iter().all(Option::is_none));
    // ...as the old "whoever is listed at the first step" would not: FAKE
    // would count at once (24% of the weight at a 0.01 price).
    let old = run(Clamp3::new(ProvenAtStart::UpAtFirstStep), DAY, with_fake);
    assert!(ln(old[DAY - 1]) < -0.5);
    // An allowed provider down or invalid at the first step is not
    // grandfathered: G earns 24 h like anyone.
    let mut allow: Vec<String> = SIX.iter().map(|s| s.to_string()).collect();
    allow.push("G".into());
    for bad in [OrEndpoint { status: 1, ..ep("G", 0.01) }, OrEndpoint { prompt: f64::NAN, ..ep("G", 0.01) }] {
        let g = run(Clamp3::with_allow_list(allow.clone()), DAY, |i| {
            let mut v = six(1.0);
            v.push(if i == 0 { bad.clone() } else { ep("G", 0.01) });
            rows(v)
        });
        assert_eq!(g, clean);
    }
}

/// A market that moves: sellers at 1 + 0.1 j, A wiggling, F down for a
/// while, everyone x0.8 at step 200.
fn market(i: usize) -> (Vec<OrEndpoint>, Vec<Option<f64>>) {
    let v: Vec<OrEndpoint> = SIX
        .iter()
        .enumerate()
        .map(|(j, p)| {
            let mut b = (1.0 + 0.1 * j as f64) * if i >= 200 { 0.8 } else { 1.0 };
            if *p == "A" {
                b *= 1.0 + 0.02 * ((i / 5) % 3) as f64;
            }
            OrEndpoint { status: if *p == "F" && (100..150).contains(&i) { 1 } else { 0 }, ..ep(p, b) }
        })
        .collect();
    let u = v.iter().map(|e| Some(if e.status == 0 { 100.0 } else { 0.0 })).collect();
    (v, u)
}

/// The recorder's line for `market(i)`: rows with their tag, uptime only in
/// the raw reply, as `staging record-or` writes it.
fn recorder_line(i: usize) -> String {
    let (eps, upt) = market(i);
    let tag = |e: &OrEndpoint| format!("{}/bf16", e.provider.to_lowercase());
    let rows: Vec<serde_json::Value> = eps
        .iter()
        .map(|e| serde_json::json!({"model": "qwen/qwen3.8-27b", "endpoint": tag(e), "provider": e.provider, "quantization": e.quantization, "prompt": e.prompt, "completion": e.completion, "status": e.status}))
        .collect();
    let raw = serde_json::json!({"data": {"endpoints": eps.iter().zip(&upt).map(|(e, u)| serde_json::json!({"tag": tag(e), "uptime_last_30m": u})).collect::<Vec<_>>()}});
    serde_json::json!({"at_ms": T0 + i as u64 * STEP, "url": "https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints", "rows": rows, "raw": raw.to_string()}).to_string()
}

#[test]
fn a_warm_start_continues_exactly_as_a_straight_replay() {
    let n = 400;
    let straight = run(listed(), n, market);
    // Two day files, the second with a torn last line and a second model.
    let dir = std::env::temp_dir().join(format!("clamp3-warm-start-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let day1: Vec<String> = (0..150).map(recorder_line).collect();
    let mut day2: Vec<String> = (150..n).map(recorder_line).collect();
    day2.push(recorder_line(399).replace("qwen/qwen3.8-27b", "other/model"));
    day2.push("{\"at_ms\":1794".into());
    std::fs::write(dir.join("2026-11-11.jsonl"), day1.join("\n") + "\n").unwrap();
    std::fs::write(dir.join("2026-11-12.jsonl"), day2.join("\n")).unwrap();
    std::fs::write(dir.join("MANIFEST"), "not a recorder file").unwrap();
    // Live from step 300: replay what was recorded before it, then go on.
    let live_from = T0 + 300 * STEP;
    let mut r = listed();
    let got = warm_start(&mut r, &dir, "qwen/qwen3.8-27b", live_from);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(got.unwrap(), (300, 1));
    assert_eq!(r.last_mark(), straight[299]);
    let live: Vec<Option<f64>> = (300..n)
        .map(|i| {
            let (eps, upt) = market(i);
            r.step(T0 + i as u64 * STEP, &eps, &upt)
        })
        .collect();
    assert_eq!(live, straight[300..]);
    // The test market moves the mark (the replay is not trivially flat).
    assert!((ln(straight[399]) - ln(straight[199])).abs() > 0.1);
}

#[test]
fn the_rule_is_deterministic() {
    let a = run(listed(), 400, market);
    assert_eq!(a, run(listed(), 400, market));
    // Row order does not matter.
    let rev = run(listed(), 400, |i| {
        let (mut v, mut u) = market(i);
        v.reverse();
        u.reverse();
        (v, u)
    });
    assert_eq!(a, rev);
    // A clone continues as the original.
    let mut r = listed();
    for i in 0..250 {
        let (v, u) = market(i);
        r.step(T0 + i as u64 * STEP, &v, &u);
    }
    let mut c = r.clone();
    for i in 250..400 {
        let (v, u) = market(i);
        assert_eq!(r.step(T0 + i as u64 * STEP, &v, &u), c.step(T0 + i as u64 * STEP, &v, &u));
    }
}

#[test]
fn with_no_proven_seller_counting_for_15_minutes_there_is_no_mark() {
    // The six count for 100 steps, vanish for 20 (only an unproven seller is
    // up), then come back.
    let input = |i: usize| rows(if (100..120).contains(&i) { vec![ep("NEW", 0.5)] } else { six(1.0) });
    let mut r = listed();
    let mut m = vec![];
    for i in 0..200 {
        let (e, u) = input(i);
        m.push(r.step(T0 + i as u64 * STEP, &e, &u));
        if i == 107 {
            assert_eq!((r.uncounted, r.last_mark()), (2, m[107]));
        }
        if i == 108 {
            assert_eq!((r.uncounted, r.last_mark()), (STALE_STEPS, None));
        }
    }
    // A seller counts while up in most of the last 13 steps: up 7 of 13 at
    // step 105, 6 at step 106. Steps 106 and 107 still give the old mark
    // (5 and 10 minutes without a count); from step 108 (15 minutes) none.
    assert!(m[..108].iter().all(|x| x.is_some_and(|x| close(x, 1.0, 1e-12))));
    assert!(m[108..120].iter().all(Option::is_none), "{:?}", &m[100..125]);
    // Back once most of the last 13 steps are up again (step 126).
    assert!(m[120..126].iter().all(Option::is_none) && m[126..].iter().all(|x| x.is_some_and(|x| close(x, 1.0, 1e-12))), "{:?}", &m[118..130]);
}

#[test]
fn unproven_sellers_gone_24_hours_are_forgotten_proven_ones_never() {
    // Step 0..20: the six and 50 unproven names up; then only the six, and
    // from step 400 to 800 only A..E (F, proven, is gone for 33 h).
    let input = |i: usize| {
        let mut v: Vec<OrEndpoint> = six(1.0).into_iter().filter(|e| !(e.provider == "F" && (400..800).contains(&i))).collect();
        if i < 20 {
            v.extend((0..50).map(|j| ep(&format!("H{j}"), 0.5)));
        }
        rows(v)
    };
    let mut r = listed();
    let mut f_state = None;
    for i in 0..800 {
        let (e, u) = input(i);
        r.step(T0 + i as u64 * STEP, &e, &u);
        if i == 399 {
            f_state = Some(r.sellers["F"].clone());
        }
        // Up to 24 h after their last row the unproven names are kept...
        if i == 19 + DAY - 1 {
            assert_eq!(r.sellers.len(), 56);
        }
    }
    // ...then dropped; the proven F, gone 33 h, is kept with its history.
    assert_eq!(r.sellers.keys().map(String::as_str).collect::<Vec<_>>(), SIX);
    let (f0, f1) = (f_state.unwrap(), &r.sellers["F"]);
    assert!(f1.proven);
    assert_eq!((&f1.obs, f1.z), (&f0.obs, f0.z));
    // A seller that earned its 24 h is proven and never dropped either.
    let mut r = listed();
    for i in 0..(DAY + 20 + 2 * DAY) {
        let mut v = six(1.0);
        if i < DAY + 20 {
            v.push(ep("LATE", 1.0));
        }
        let u = up(&v);
        r.step(T0 + i as u64 * STEP, &v, &u);
    }
    assert!(r.sellers["LATE"].proven);
}

#[test]
fn the_allow_list_is_trimmed_matched_exactly_and_reported() {
    let mut r = Clamp3::with_allow_list([" A".to_string(), "B\t".into(), "c".into(), "Gone".into()]);
    assert!(r.allow_list_report().is_none());
    let v = six(1.0);
    r.step(T0, &v, &up(&v));
    let (hit, miss) = r.allow_list_report().unwrap();
    assert_eq!((hit.as_slice(), miss.as_slice()), (&["A".to_string(), "B".into()][..], &["Gone".to_string(), "c".into()][..]));
    assert!(r.sellers["A"].proven && r.sellers["B"].proven && !r.sellers["C"].proven);
    // The check's convention has no list to report.
    let mut u = Clamp3::new(ProvenAtStart::UpAtFirstStep);
    u.step(T0, &v, &up(&v));
    assert!(u.allow_list_report().is_none());
}
