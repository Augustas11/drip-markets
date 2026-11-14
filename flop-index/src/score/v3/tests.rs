//! Check v3 terms on tiny synthetic series with hand-computed answers.

use super::*;
use crate::score::rules::{by_name, Mean, Spec};

const T0: u64 = 1_794_355_200_000; // 2026-11-11 00:00 UTC
const STEP: u64 = 5 * MIN;

/// An endpoint whose 15:1 blend is `b` USD per 1M tokens.
fn ep(p: &str, q: &str, b: f64, status: i64) -> OrEndpoint {
    OrEndpoint { provider: p.into(), quantization: q.into(), prompt: b * 1e-6, completion: b * 1e-6, status }
}

fn snap(i: u64, rows: &[(&str, f64, i64, Option<f64>)]) -> Snap {
    Snap { at_ms: T0 + i * STEP, eps: rows.iter().map(|r| ep(r.0, "bf16", r.1, r.2)).collect(), uptime: rows.iter().map(|r| r.3).collect() }
}

fn series(snaps: Vec<Snap>) -> Series3 {
    Series3 { model: "test/m".into(), snaps }
}

/// `n` snapshots of the same rows.
fn flat(n: u64, rows: &[(&str, f64, i64, Option<f64>)]) -> Series3 {
    series((0..n).map(|i| snap(i, rows)).collect())
}

fn marks(v: &[f64]) -> Vec<Option<f64>> {
    v.iter().map(|x| Some(*x)).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

const U: Option<f64> = Some(100.0);

#[test]
fn routed_matches_hand_computed_weights() {
    let r = |rows: &[(&str, f64, i64, Option<f64>)]| {
        let s = snap(0, rows);
        routed(&s.eps, &s.uptime)
    };
    // Weights 100/1, 100/4 = 0.8, 0.2; capped 0.5, 0.5: price 1.5.
    assert!(close(r(&[("A", 1.0, 0, U), ("B", 2.0, 0, U)]).unwrap(), 1.5));
    // 0.667, 0.167, 0.167 -> A capped at 0.5, B and C 0.25 each: 1.5.
    assert!(close(r(&[("A", 1.0, 0, U), ("B", 2.0, 0, U), ("C", 2.0, 0, U)]).unwrap(), 1.5));
    // 100, 25, 6.25 -> A capped 0.5, B 0.4, C 0.1: 0.5 + 0.8 + 0.4. D has uptime 0.
    assert!(close(r(&[("A", 1.0, 0, U), ("B", 2.0, 0, U), ("C", 4.0, 0, U), ("D", 4.0, 0, Some(0.0))]).unwrap(), 1.7));
    // Null uptime weighs 0: B 0.8 -> 0.5, C 0.5: 1 + 2.
    assert!(close(r(&[("A", 1.0, 0, None), ("B", 2.0, 0, U), ("C", 4.0, 0, U)]).unwrap(), 3.0));
    // Down sellers are out; the same answer.
    assert!(close(r(&[("A", 1.0, 1, U), ("B", 2.0, 0, U), ("C", 4.0, 0, U)]).unwrap(), 3.0));
    // Uptime scales weight: 50/1 vs 100/4 = 2/3, 1/3 -> capped 0.5/0.5.
    assert!(close(r(&[("A", 1.0, 0, Some(50.0)), ("B", 2.0, 0, U)]).unwrap(), 1.5));
    // One provider, its lowest row only: A 1.0 (not 3.0) with B 4.0: 100, 6.25 -> 0.5/0.5.
    assert!(close(r(&[("A", 3.0, 0, U), ("A", 1.0, 0, U), ("B", 4.0, 0, U)]).unwrap(), 2.5));
    // Fewer than two weighted providers: none.
    assert_eq!(r(&[("A", 1.0, 0, U), ("B", 2.0, 0, None)]), None);
    // The rule: the same price, and none without uptime.
    let s = snap(0, &[("A", 1.0, 0, U), ("B", 2.0, 0, U)]);
    let mut rt = by_name("routed").unwrap();
    assert!(close(rt.step_v3(0, &s.eps, &s.uptime).unwrap(), 1.5));
    assert_eq!(rt.step(0, &s.eps), None);
}

#[test]
fn a_price_change_is_a_new_listed_price_not_availability() {
    let a = [("A", 1.0, 0, U), ("B", 2.0, 0, U)];
    let ch = |rows: &[(&str, f64, i64, Option<f64>)]| changes(&series(vec![snap(0, &a), snap(1, rows), snap(2, &a)]));
    // Leaving, then relisting at the old price: availability both ways.
    assert_eq!(ch(&[("A", 1.0, 0, U)]), vec![true, false, false]);
    // Down, then up: availability.
    assert_eq!(ch(&[("A", 1.0, 0, U), ("B", 2.0, 1, Some(10.0))]), vec![true, false, false]);
    // A cut, then back: two new prices.
    assert_eq!(ch(&[("A", 1.0, 0, U), ("B", 1.5, 0, U)]), vec![true, true, true]);
    // A new seller: a new price; leaving again: availability.
    assert_eq!(ch(&[("A", 1.0, 0, U), ("B", 2.0, 0, U), ("C", 3.0, 0, U)]), vec![true, true, false]);
}

#[test]
fn a_flicker_on_a_step_with_a_price_change_is_still_phantom() {
    // Step 2: C (3.0) goes down and A reprices 1.0 -> 1.1. Mean of the up
    // sellers: 2 -> 1.55. On step 1's availability with step 2's prices it
    // would be (1.1 + 2 + 3) / 3: the rest is the flicker.
    let r3 = [("A", 1.0, 0, U), ("B", 2.0, 0, U), ("C", 3.0, 0, U)];
    let s = series(vec![snap(0, &r3), snap(1, &r3), snap(2, &[("A", 1.1, 0, U), ("B", 2.0, 0, U), ("C", 3.0, 1, U)])]);
    let r = replay(&Mean, &s, &BTreeSet::new());
    assert_eq!(r.marks, marks(&[2.0, 2.0, 1.55]));
    assert!(close(r.cf[2].unwrap(), 6.1 / 3.0) && r.cf[1].is_none());
    let c = clean_terms(&r.marks, &r.cf, r.holds, &s, REVERSAL_K);
    assert!(close(c.movement, (2.0f64 / 1.55).ln()));
    assert!(close(c.phantom, (6.1f64 / 3.0 / 1.55).ln()), "{c:?}");
    // A delisted seller relisting at its old price is availability too.
    let s = series(vec![snap(0, &r3), snap(1, &[("A", 1.0, 0, U), ("B", 2.0, 0, U)]), snap(2, &r3)]);
    let r = replay(&Mean, &s, &BTreeSet::new());
    let c = clean_terms(&r.marks, &r.cf, r.holds, &s, REVERSAL_K);
    assert!(close(c.phantom, 2.0 * (2.0f64 / 1.5).ln()) && close(c.phantom, c.movement), "{c:?}");
}

#[test]
fn a_flickering_seller_is_all_phantom_and_reversed() {
    // C (3.0) goes down at step 2 and comes back at step 3. A mean of the up
    // sellers: 2, 2, 1.5, 2, 2.
    let (r3, r2) = ([("A", 1.0, 0, U), ("B", 2.0, 0, U), ("C", 3.0, 0, U)], [("A", 1.0, 0, U), ("B", 2.0, 0, U), ("C", 3.0, 1, Some(80.0))]);
    let s = series(vec![snap(0, &r3), snap(1, &r3), snap(2, &r2), snap(3, &r3), snap(4, &r3)]);
    let r = replay(&Mean, &s, &BTreeSet::new());
    assert_eq!(r.marks, marks(&[2.0, 2.0, 1.5, 2.0, 2.0]));
    let c = clean_terms(&r.marks, &r.cf, r.holds, &s, REVERSAL_K);
    let x = (4.0f64 / 3.0).ln();
    assert!(close(c.movement, 2.0 * x) && close(c.phantom, 2.0 * x), "{c:?}");
    // The drop is undone the next step; the rise is never undone.
    assert!(close(c.reversal, x), "{c:?}");
    assert_eq!(c.stale, 0.0, "every step's input differs from the one before");
    assert_eq!((c.distinct, c.holds), (2, 0));
    // The same move from a real price change is not phantom.
    let r6 = [("A", 1.0, 0, U), ("B", 2.0, 0, U), ("C", 6.0, 0, U)];
    let s = series(vec![snap(0, &r3), snap(1, &r3), snap(2, &r6), snap(3, &r6)]);
    let r = replay(&Mean, &s, &BTreeSet::new());
    let c = clean_terms(&r.marks, &r.cf, r.holds, &s, REVERSAL_K);
    assert!(close(c.movement, 1.5f64.ln()) && c.phantom == 0.0 && c.reversal == 0.0, "{c:?}");
}

/// Moves to 2.0 from step `at` on, whatever the input: a hold expiring.
#[derive(Clone)]
struct Expire(u64);
impl Rule for Expire {
    fn step(&mut self, at_ms: u64, _: &[OrEndpoint]) -> Option<f64> {
        Some(if at_ms >= self.0 { 2.0 } else { 1.0 })
    }
    fn clone_box(&self) -> Box<dyn Rule> {
        Box::new(self.clone())
    }
}

#[test]
fn a_move_on_identical_input_is_stale_and_phantom() {
    let s = flat(6, &[("A", 1.0, 0, U), ("B", 2.0, 0, U)]);
    let r = replay(&Expire(T0 + 3 * STEP), &s, &BTreeSet::new());
    let c = clean_terms(&r.marks, &r.cf, r.holds, &s, REVERSAL_K);
    assert!(close(c.stale, 2f64.ln()) && close(c.phantom, 2f64.ln()) && close(c.movement, 2f64.ln()), "{c:?}");
}

#[test]
fn reversal_counts_only_what_is_undone_inside_k_steps() {
    let s = flat(10, &[("A", 1.0, 0, U), ("B", 2.0, 0, U)]);
    let c = |v: &[f64], k| clean_terms(&marks(v), &[], 0, &series(s.snaps[..v.len()].to_vec()), k).reversal;
    assert!(close(c(&[1.0, 1.1, 1.0, 1.0], 6), 1.1f64.ln()));
    // Partly undone: only the part.
    assert!(close(c(&[1.0, 1.2, 1.1], 6), (1.2f64 / 1.1).ln()));
    // Undone 7 steps later: outside k = 6, inside k = 7.
    let late = [1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0];
    assert_eq!(c(&late, 6), 0.0);
    assert!(close(c(&late, 7), 1.1f64.ln()));
}

/// 12 snapshots of A 1, B 2, C 4 where `f(i)` gives each one's multiplier.
fn moving(f: impl Fn(u64) -> [f64; 3]) -> Series3 {
    series((0..12).map(|i| {
        let m = f(i);
        snap(i, &[("A", m[0], 0, U), ("B", 2.0 * m[1], 0, U), ("C", 4.0 * m[2], 0, U)])
    }).collect())
}

#[test]
fn follow_reads_the_mark_against_the_mean_seller_move() {
    // Everyone cuts 10% at step 3 and stays: one event of ln 0.9.
    let s = moving(|i| if i >= 3 { [0.9; 3] } else { [1.0; 3] });
    let f = |v: f64| {
        let m: Vec<f64> = (0..12).map(|i| if i >= 3 { v } else { 1.0 }).collect();
        clean_terms(&marks(&m), &[], 0, &s, REVERSAL_K).follow
    };
    assert_eq!(f(0.9).len(), 1);
    let c = clean_terms(&marks(&[1.0; 12]), &[], 0, &s, REVERSAL_K);
    assert_eq!(c.follow_broad.len(), 1, "all three moved");
    assert!(close(f(0.9)[0], 1.0));
    assert!(close(f(1.0)[0], 0.0), "a frozen mark shows none of it");
    assert!(close(f(0.9f64.sqrt())[0], 0.5));
    assert!(close(f(0.81)[0], 0.0), "twice the move is as wrong as none");
    // One of three cuts 30%: consensus ln(0.7)/3; a mark showing half of that scores 0.5.
    let s = moving(|i| if i >= 3 { [0.7, 1.0, 1.0] } else { [1.0; 3] });
    let m: Vec<f64> = (0..12).map(|i| if i >= 3 { (0.7f64.ln() / 6.0).exp() } else { 1.0 }).collect();
    let c = clean_terms(&marks(&m), &[], 0, &s, REVERSAL_K);
    assert!(c.follow.len() == 1 && close(c.follow[0], 0.5), "{:?}", c.follow);
    assert!(c.follow_broad.is_empty(), "one seller is not a broad move");
    // A cut undone inside FOLLOW_N steps is no consensus move.
    let s = moving(|i| if (3..5).contains(&i) { [0.5, 1.0, 1.0] } else { [1.0; 3] });
    assert!(clean_terms(&marks(&[1.0; 12]), &[], 0, &s, REVERSAL_K).follow.is_empty());
}

#[test]
fn level_is_the_log_distance_from_routed() {
    // Routed of A 1, B 2 at uptime 100 is 1.5.
    let s = flat(3, &[("A", 1.0, 0, U), ("B", 2.0, 0, U)]);
    assert_eq!(clean_terms(&marks(&[1.5; 3]), &[], 0, &s, REVERSAL_K).level.iter().map(|x| (x * 1e9).round()).collect::<Vec<_>>(), vec![0.0; 3]);
    let c = clean_terms(&marks(&[3.0; 3]), &[], 0, &s, REVERSAL_K);
    assert!(c.level.iter().all(|x| close(*x, 2f64.ln())));
    // No mark yet costs the penalty.
    let c = clean_terms(&[None, Some(1.5), Some(1.5)], &[], 0, &s, REVERSAL_K);
    assert_eq!(c.level[0], PENALTY);
}

#[test]
fn split_is_by_model_hash_and_holds_the_benchmark_out() {
    assert!(heldout(BENCHMARK) && in_split(BENCHMARK, Split::Heldout) && !in_split(BENCHMARK, Split::Tune));
    let models: Vec<String> = (0..200).map(|i| format!("org/m{i}")).collect();
    let h = models.iter().filter(|m| heldout(m)).count();
    assert!((70..=130).contains(&h), "about half: {h}");
    assert_eq!(models.iter().filter(|m| heldout(m)).count(), h, "stable");
}

/// A 6 h series (attacks need 2 h after the anchor) of 3 bf16 sellers at 1.0.
fn three_at_one() -> Series3 {
    flat(72, &[("A", 1.0, 0, U), ("B", 1.0, 0, U), ("C", 1.0, 0, U)])
}

fn hits(rule: &dyn Rule, s: &Series3, seed: u64) -> SeedRun {
    let names: Vec<String> = vec!["X".into(), "Y".into(), "Z".into()];
    let r = run_series(rule, s, &[seed], &names, REVERSAL_K);
    r.seeds.into_iter().next().unwrap()
}

fn worst(r: &SeedRun, sc: usize, k: usize, q: usize) -> f64 {
    r.hits.iter().filter(|h| (h.sc, h.k, h.q) == (sc, k, q)).map(|h| h.mv).fold(0.0, f64::max)
}

#[test]
fn attacks_move_a_mean_by_the_hand_computed_amount() {
    let s = three_at_one();
    let r = hits(&Mean, &s, 1);
    assert!(!r.hits.is_empty() && !r.follow_inj.is_empty());
    // Fakes at 0.5x the cheapest (1.0): (3 + 0.5) / 4, (3 + 1) / 5.
    assert!(close(worst(&r, 0, 1, 0), (1.0f64 / 0.875).ln()));
    assert!(close(worst(&r, 0, 2, 0), (1.0f64 / 0.8).ln()));
    assert!(close(worst(&r, 0, 1, 1), (1.0f64 / 0.875).ln()), "bf16 fakes too");
    // 0.1x: 3.1 / 4.
    assert!(close(worst(&r, 1, 1, 0), (1.0f64 / 0.775).ln()));
    // Promo 0.5x: 2.5 / 3. Relist 10x: 12 / 3.
    assert!(close(worst(&r, 2, 1, 0), (3.0f64 / 2.5).ln()));
    assert!(close(worst(&r, 3, 1, 0), 4f64.ln()));
    assert!(close(worst(&r, 3, 3, 0), 10f64.ln()));
    // Newlow: 2.5 / 3 for one step, then the seller is down: mean of two at 1.0.
    assert!(close(worst(&r, 4, 1, 0), (3.0f64 / 2.5).ln()));
    // Down: one of three, the rest are all 1.0.
    assert_eq!(worst(&r, DOWN, 3, 0), 0.0);
    // A mark that never moves is never moved, and follows nothing.
    #[derive(Clone)]
    struct Frozen;
    impl Rule for Frozen {
        fn step(&mut self, _: u64, _: &[OrEndpoint]) -> Option<f64> {
            Some(1.0)
        }
        fn clone_box(&self) -> Box<dyn Rule> {
            Box::new(Frozen)
        }
    }
    let f = hits(&Frozen, &s, 1);
    assert!(f.hits.iter().all(|h| h.mv == 0.0));
    assert!(f.follow_inj.iter().all(|x| *x == 0.0));
    // The mean follows the injected all-seller move nearly in full.
    assert!(r.follow_inj.iter().all(|x| *x > 0.6), "{:?}", r.follow_inj);
}

#[test]
fn one_bf16_fake_moves_the_live_rule_tens_of_percent() {
    // The live bf16 pair and a cheaper fp8 seller, as on qwen3.8-27b.
    let rows = vec![ep("DeepInfra", "bf16", 0.2578, 0), ep("GMICloud", "bf16", 0.6219, 0), ep("Chutes", "fp8", 0.127, 0)];
    let s = series((0..72).map(|i| Snap { at_ms: T0 + i * STEP, eps: rows.clone(), uptime: vec![U; 3] }).collect());
    let r = hits(&Spec, &s, 1);
    // By hand: the bf16 set {0.0635 fake, 0.2578, 0.6219} clamps the fake to
    // half the median, 0.1289; 1/p² weights then give 0.1702 against 0.3112.
    let want = (0.3112f64 / 0.1702).ln();
    let got = worst(&r, 0, 1, 1);
    assert!((got - want).abs() < 0.01, "{got} vs {want}");
    // An fp8 fake (copying the cheapest seller) does not touch the bf16 set.
    assert_eq!(worst(&r, 0, 1, 0), 0.0);
}

#[test]
fn newlow_offline_lists_the_new_price_then_goes_down_with_it() {
    let s = three_at_one();
    let inj = Inj { sc: 4, k: 1, q: 0, victims: vec!["A".into()], fakes: vec![] };
    let (e0, u0) = inject(&inj, T0, &s.snaps[0]);
    assert!(e0[0].status == 0 && close(blend(&e0[0]), 0.5) && u0[0] == U);
    let (e1, u1) = inject(&inj, T0, &s.snaps[1]);
    assert!(e1[0].status == 1 && close(blend(&e1[0]), 0.5));
    assert!(close(u1[0].unwrap(), 100.0 * (1.0 - 5.0 / 30.0)));
    // Past its 6 h: as recorded.
    let late = Snap { at_ms: T0 + 6 * HOUR, ..s.snaps[0].clone() };
    assert_eq!(inject(&inj, T0, &late).0, late.eps);
}

#[test]
fn the_same_seeds_give_the_same_report_and_spec_scores_zero_against_itself() {
    let s = vec![three_at_one(), Series3 { model: "test/n".into(), ..three_at_one() }];
    let a = report(&Spec, &Spec, "spec", "spec", &s, "m", "d", Split::All, &[1, 2], REVERSAL_K).unwrap();
    let b = report(&Spec, &Spec, "spec", "spec", &s, "m", "d", Split::All, &[1, 2], REVERSAL_K).unwrap();
    assert_eq!(a, b);
    assert!(a.iter().any(|l| l.starts_with("METRIC total_gain_vs_spec +0.000000 se 0.000000")), "{a:#?}");
    assert!(a.iter().any(|l| l.starts_with("GATE gain_vs_spec FAIL")));
}

#[test]
fn a_recorder_pin_with_sub_folders_loads_with_uptime() {
    let dir = std::env::temp_dir().join(format!("flop-v3-pin-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("org__m")).unwrap();
    let mut wide = String::new();
    let mut rec = String::new();
    for i in 0..3u64 {
        let rows: Vec<String> = (0..5).map(|p| format!(r#"{{"provider":"P{p}","tag":"p{p}/bf16","quantization":"bf16","prompt":1e-7,"completion":1e-6,"status":0,"uptime_30m":{}}}"#, if p == 0 { "null".to_owned() } else { format!("{}", 90 + p) })).collect();
        wide.push_str(&format!(r#"{{"at_ms":{},"model":"org/m","rows":[{}]}}"#, T0 + i * STEP, rows.join(",")));
        wide.push('\n');
        let raw = r#"{\"data\":{\"endpoints\":[{\"tag\":\"q0/fp8\",\"uptime_last_30m\":77.5}]}}"#;
        let rows: Vec<String> = (0..5).map(|p| format!(r#"{{"model":"org/r","endpoint":"q{p}/fp8","provider":"Q{p}","quantization":"fp8","prompt":1e-7,"completion":1e-6,"status":0}}"#)).collect();
        rec.push_str(&format!(r#"{{"at_ms":{},"url":"u","rows":[{}],"raw":"{raw}"}}"#, T0 + i * STEP, rows.join(",")));
        rec.push('\n');
    }
    std::fs::write(dir.join("org__m/2026-11-11.jsonl"), wide).unwrap();
    std::fs::write(dir.join("rec.jsonl"), rec).unwrap();
    assert!(data::pin(&dir, &["wide-jsonl:../x.jsonl".into()]).is_err(), "no way out of the folder");
    data::pin(&dir, &["wide-jsonl:org__m/2026-11-11.jsonl".into(), "recorder-jsonl:rec.jsonl".into()]).unwrap();
    let (s, _) = load(&dir).unwrap();
    assert_eq!(s.iter().map(|s| s.model.as_str()).collect::<Vec<_>>(), vec!["org/m", "org/r"]);
    assert_eq!(s[0].snaps.len(), 3);
    assert_eq!(s[0].snaps[0].uptime, vec![None, Some(91.0), Some(92.0), Some(93.0), Some(94.0)]);
    assert_eq!(s[1].snaps[0].uptime, vec![Some(77.5), None, None, None, None], "recorder uptime comes from raw, by tag");
    // v1 reads the wide pin too.
    let (v1, _) = data::load(&dir).unwrap();
    assert!(v1.iter().any(|s| s.group == "wide:org/m"));
    let _ = std::fs::remove_dir_all(&dir);
}
