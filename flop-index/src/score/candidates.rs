//! Candidate mark rules. To add one: write a struct implementing
//! `rules::Rule` below and add one line to `ALL`. Nothing else changes.
//! Score it with `flop-score --data <pin> --rule <name>`.

#[allow(unused_imports)]
use super::rules::{blend, up, Rule};
#[allow(unused_imports)]
use crate::{combine, or_quote_with, spec_quote, Guard, Holder, OrEndpoint, SPEC};

/// (name, constructor). Names are unique, lowercase, no spaces.
pub const ALL: &[(&str, fn() -> Box<dyn Rule>)] = &[
    ("hold-2h", || Box::new(Hold(Holder::new(2 * HOUR_MS, SPEC.hold_confirm_s * 1000)))),
    ("hold-6h", || Box::new(Hold(Holder::new(6 * HOUR_MS, SPEC.hold_confirm_s * 1000)))),
    ("hold-12h", || Box::new(Hold(Holder::new(12 * HOUR_MS, SPEC.hold_confirm_s * 1000)))),
    ("all-median", || Box::new(AllSellers { holder: Holder::new(0, 0), weighting: "median", fallback: false })),
    ("all-median-hold6h", || Box::new(AllSellers { holder: Holder::new(6 * HOUR_MS, SPEC.hold_confirm_s * 1000), weighting: "median", fallback: false })),
    ("wide", || Box::new(AllSellers { holder: Holder::new(6 * HOUR_MS, SPEC.hold_confirm_s * 1000), weighting: "median", fallback: true })),
    ("all-invsq-hold6h", || Box::new(AllSellers { holder: Holder::new(6 * HOUR_MS, SPEC.hold_confirm_s * 1000), weighting: "inverse_square", fallback: false })),
    // The proposed settlement rule (crate::clamp3, SPEC.md "Settlement rule
    // clamp3"), with the check's convention for who is proven at the start.
    ("clamp3", || Box::new(Clamp3Rule(crate::clamp3::Clamp3::new(crate::clamp3::ProvenAtStart::UpAtFirstStep)))),
];

/// Every up seller of the model, any precision, one price per provider (its
/// lowest), combined by `weighting`; with an optional hold for sellers that
/// go offline. Scratch candidate for widening the settlement price.
#[derive(Clone)]
pub struct AllSellers { holder: Holder, weighting: &'static str, fallback: bool }
impl Rule for AllSellers {
    fn step(&mut self, at_ms: u64, endpoints: &[OrEndpoint]) -> Option<f64> {
        let (eps, _) = self.holder.apply(at_ms, endpoints);
        let mut best = std::collections::BTreeMap::<String, f64>::new();
        for e in up(&eps) {
            let b = blend(e);
            let x = best.entry(e.provider.clone()).or_insert(b);
            if b < *x { *x = b; }
        }
        if best.len() < 3 {
            // Too few sellers for a median: `fallback` uses the live rule, else hold.
            return if self.fallback { crate::spec_quote(&eps, at_ms).ok().map(|q| q.usd_e6 as f64 / 1e6) } else { None };
        }
        let mut v: Vec<f64> = best.into_values().collect();
        if self.weighting != "median" {
            // the live clamp: under half the median counts as half the median
            let mut s = v.clone(); s.sort_by(f64::total_cmp);
            let floor = 0.5 * combine(&s, "median").ok()?;
            for x in v.iter_mut() { if *x < floor { *x = floor; } }
        }
        combine(&v, self.weighting).ok()
    }
    fn clone_box(&self) -> Box<dyn Rule> { Box::new(self.clone()) }
}

const HOUR_MS: u64 = 3_600_000;

/// SPEC with a missing endpoint held at its last up price (`crate::Holder`,
/// decision 2026-11-13). On daily pins snapshots are 24 h apart, so no hold here
/// bridges two of them and these score exactly as `spec`; they differ only on
/// recorder data.
#[derive(Clone)]
pub struct Hold(Holder);
impl Rule for Hold {
    fn step(&mut self, at_ms: u64, endpoints: &[OrEndpoint]) -> Option<f64> {
        let (eps, _) = self.0.apply(at_ms, endpoints);
        spec_quote(&eps, at_ms).ok().map(|q| q.usd_e6 as f64 / 1e6)
    }
    fn clone_box(&self) -> Box<dyn Rule> {
        Box::new(self.clone())
    }
}

/// `crate::clamp3` as a scorer rule. Sellers with a valid up row at the
/// first step are proven (the check's convention: a 21-hour pin cannot show
/// 24 h of serving); the live rule takes an explicit allow-list instead.
#[derive(Clone)]
pub struct Clamp3Rule(pub crate::clamp3::Clamp3);
impl Rule for Clamp3Rule {
    fn step(&mut self, at_ms: u64, endpoints: &[OrEndpoint]) -> Option<f64> {
        // No uptime in the data: nobody serves, so nobody counts.
        self.0.step(at_ms, endpoints, &vec![None; endpoints.len()])
    }
    fn step_v3(&mut self, at_ms: u64, endpoints: &[OrEndpoint], uptime: &[Option<f64>]) -> Option<f64> {
        self.0.step(at_ms, endpoints, uptime)
    }
    fn clone_box(&self) -> Box<dyn Rule> {
        Box::new(self.clone())
    }
}
