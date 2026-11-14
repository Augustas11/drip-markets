//! flop-score: score a FLOP mark rule against the SPEC baseline on pinned
//! data. See docs/flop-compute/SCORER.md.
//!
//!   flop-score --data <pin dir> --rule <name> [--base spec] [--split tune|heldout|all] [--seed N]
//!   (a rule or base may be wasm:<file.wasm>, docs/flop-compute/CHALLENGE_ABI.md)
//!   flop-score pin <dir> <loader>:<file>...
//!   flop-score list
//!   flop-score --check v3 --data <recorder pin> --rule <name> [--base spec] [--split tune|heldout|all] [--seeds 1..8] [--k 6]
//!   (check v3 also with FLOP_SCORE_V3=1; docs/flop-compute/SCORER.md, "Check v3")
//!   flop-score --check v4 --data <recorder pin> --rule <name> [--split tune|heldout|all] [--seeds 1..8] [--baselines spec,routed,...|none] [--threads N]
//!   (check v4 also with FLOP_SCORE_V4=1; docs/flop-compute/SCORER.md, "Check v4")
//!
//! Exit 0: it ran, the first line is the number. Anything else: it did not.

use anyhow::{bail, Context, Result};
use flop_index::score::{self, data, rules, Split, SCENARIOS};
use std::path::PathBuf;

const USAGE: &str = "usage: flop-score --data <pin dir> --rule <name|wasm:file> [--base spec] [--split tune|heldout|all] [--seed N]\n       flop-score --check v3 --data <recorder pin> --rule <name|wasm:file> [--base spec] [--split tune|heldout|all] [--seeds 1..8] [--k 6]\n       flop-score --check v4 --data <recorder pin> --rule <name|wasm:file> [--split tune|heldout|all] [--seeds 1..8] [--baselines spec,routed,routed-tm60,all-median,wide|none] [--threads N]\n       flop-score pin <dir> <loader>:<file>...\n       flop-score list";

fn day(d: u64) -> String {
    // Civil from days (H. Hinnant), for the report's date range.
    let z = d as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let dd = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{dd:02}")
}

fn run(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("list") => {
            for n in rules::names() {
                println!("{n}");
            }
            return Ok(());
        }
        Some("pin") => {
            let dir = PathBuf::from(args.get(1).context(USAGE)?);
            print!("{}", data::pin(&dir, &args[2..])?);
            return Ok(());
        }
        _ => {}
    }
    let v4 = std::env::var_os("FLOP_SCORE_V4").is_some_and(|v| v == "1") || args.windows(2).any(|w| w[0] == "--check" && w[1] == "v4");
    if v4 {
        return run_v4(args);
    }
    let v3 = std::env::var_os("FLOP_SCORE_V3").is_some_and(|v| v == "1") || args.windows(2).any(|w| w[0] == "--check" && w[1] == "v3");
    if v3 {
        return run_v3(args);
    }
    let (mut data_dir, mut rule, mut base, mut split, mut seed) = (None, None, "spec".to_owned(), Split::Tune, 1u64);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value\n{USAGE}"));
        match a.as_str() {
            "--data" => data_dir = Some(PathBuf::from(val()?)),
            "--rule" => rule = Some(val()?.clone()),
            "--base" => base = val()?.clone(),
            "--split" => split = val()?.parse()?,
            "--seed" => seed = val()?.parse().context("--seed is a number")?,
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    let data_dir = data_dir.context(USAGE)?;
    let rule = rule.context(USAGE)?;
    let (cand, base_rule) = (get(&rule)?, get(&base)?);
    let (series, manifest) = data::load(&data_dir)?;
    let snaps: usize = series.iter().map(|s| s.snaps.len()).sum();
    let r = score::score(cand.as_ref(), base_rule.as_ref(), &series, seed, split)?;
    let t = r.total;
    println!("gain {:+.6} se {:.6} n {} ({rule} vs {base}, split {split:?}, seed {seed})", t.gain, t.se, t.n);
    let part = |name: &str, s: score::Stat, b: f64, c: f64| println!("  {name:<10} gain {:+.6} se {:.6}   loss {base} {b:.6} {rule} {c:.6}", s.gain, s.se);
    part("jump", r.jump, r.base.jump, r.cand.jump);
    part("attack", r.attack, r.base.attack_mean(), r.cand.attack_mean());
    for k in 0..4 {
        part(&format!("  {}", SCENARIOS[k]), r.scenario[k], r.base.attack[k], r.cand.attack[k]);
    }
    part("follow", r.follow, r.base.follow, r.cand.follow);
    part("reference", r.reference, r.base.reference, r.cand.reference);
    part("ex-ref", r.ex_reference, r.base.ex_reference(), r.cand.ex_reference());
    println!("days {}..{}  series {}  snapshots {snaps}  data {} (MANIFEST sha256 {manifest})", day(r.first_day), day(r.last_day), series.len(), data_dir.display());
    let gate = if r.passes() { "above" } else { "not above" };
    let note = if split == Split::Heldout { "" } else { " (the gate counts only --split heldout)" };
    println!("gate: gain is {gate} 2 x se{note}");
    Ok(())
}

fn get(name: &str) -> Result<Box<dyn rules::Rule>> {
    match name.strip_prefix("wasm:") {
        Some(path) => Ok(Box::new(score::wasm::WasmRule::load(path.as_ref())?)),
        None => rules::by_name(name).with_context(|| format!("no rule {name:?}; flop-score list")),
    }
}

/// Check v3 (score/v3.rs): every model in a 5-minute recorder pin, seeds
/// pooled, term and GATE lines.
fn run_v3(args: &[String]) -> Result<()> {
    let (mut data_dir, mut rule, mut base, mut split, mut seeds, mut k) = (None, None, "spec".to_owned(), Split::Tune, (1u64..=8).collect::<Vec<_>>(), score::v3::REVERSAL_K);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value\n{USAGE}"));
        match a.as_str() {
            "--check" => {
                let v = val()?;
                if v != "v3" {
                    bail!("--check takes v3 (v1 is the default, v2 is FLOP_SCORE_V2=1), not {v:?}");
                }
            }
            "--data" => data_dir = Some(PathBuf::from(val()?)),
            "--rule" => rule = Some(val()?.clone()),
            "--base" => base = val()?.clone(),
            "--split" => split = val()?.parse()?,
            "--seed" => seeds = vec![val()?.parse().context("--seed is a number")?],
            "--seeds" => {
                let v = val()?;
                let (a, b) = v.split_once("..").context("--seeds is A..B")?;
                let (a, b): (u64, u64) = (a.parse().context("--seeds A..B")?, b.parse().context("--seeds A..B")?);
                if a > b {
                    bail!("--seeds {v}: empty");
                }
                seeds = (a..=b).collect();
            }
            "--k" => k = val()?.parse().context("--k is a number of steps")?,
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    let data_dir = data_dir.context(USAGE)?;
    let rule = rule.context(USAGE)?;
    let (cand, base_rule) = (get(&rule)?, get(&base)?);
    let (series, manifest) = score::v3::load(&data_dir)?;
    let lines = score::v3::report(cand.as_ref(), base_rule.as_ref(), &rule, &base, &series, &manifest, &data_dir.display().to_string(), split, &seeds, k)?;
    for l in lines {
        println!("{l}");
    }
    Ok(())
}

/// Check v4 (score/v4.rs): the rule and the baselines on every model of a
/// 5-minute recorder pin, attacker types U/S/P/M/F, liveliness, level, phantom.
fn run_v4(args: &[String]) -> Result<()> {
    let (mut data_dir, mut rule, mut split, mut seeds) = (None, None, Split::Tune, (1u64..=8).collect::<Vec<_>>());
    let mut baselines: Vec<String> = score::v4::BASELINES.iter().map(|s| s.to_string()).collect();
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value\n{USAGE}"));
        match a.as_str() {
            "--check" => {
                let v = val()?;
                if v != "v4" {
                    bail!("--check takes v3 or v4, not {v:?}");
                }
            }
            "--data" => data_dir = Some(PathBuf::from(val()?)),
            "--rule" => rule = Some(val()?.clone()),
            "--split" => split = val()?.parse()?,
            "--seed" => seeds = vec![val()?.parse().context("--seed is a number")?],
            "--seeds" => {
                let v = val()?;
                let (a, b) = v.split_once("..").context("--seeds is A..B")?;
                let (a, b): (u64, u64) = (a.parse().context("--seeds A..B")?, b.parse().context("--seeds A..B")?);
                if a > b {
                    bail!("--seeds {v}: empty");
                }
                seeds = (a..=b).collect();
            }
            "--baselines" => {
                let v = val()?;
                baselines = if v == "none" { vec![] } else { v.split(',').map(str::to_owned).collect() };
            }
            "--threads" => threads = val()?.parse().context("--threads is a number")?,
            other => bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }
    let data_dir = data_dir.context(USAGE)?;
    let rule = rule.context(USAGE)?;
    // Fail early on an unknown name.
    for r in std::iter::once(&rule).chain(&baselines) {
        get(r)?;
    }
    let (series, manifest) = score::v3::load(&data_dir)?;
    let factory = |n: &str| get(n);
    let lines = score::v4::report(&rule, &baselines, &factory, &series, &manifest, &data_dir.display().to_string(), split, &seeds, threads)?;
    for l in lines {
        println!("{l}");
    }
    Ok(())
}

fn main() {
    // A panic (in a candidate rule, say) is a run that did not happen too.
    std::panic::set_hook(Box::new(|info| {
        // A wasm rule's runtime call is caught and disqualifies the rule.
        if score::wasm::in_guard() {
            return;
        }
        // A reader that stopped early (`| head -1`) already has the number.
        let m = info.to_string();
        if m.contains("failed printing to stdout") && m.contains("Broken pipe") {
            std::process::exit(0);
        }
        eprintln!("flop-score: DID NOT RUN: panic: {info}");
        std::process::exit(2);
    }));
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("flop-score: DID NOT RUN: {e:#}");
        std::process::exit(2);
    }
}
