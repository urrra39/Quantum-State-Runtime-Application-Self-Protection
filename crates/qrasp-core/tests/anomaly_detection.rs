//! Integration tests for the Q-RASP core anomaly detector.
//!
//! These exercise `StateObserver` through its public API exactly as the PyO3
//! bridge does: feed a sequence of density matrices and assert the emitted
//! `Anomaly` classification. They complement the in-module unit tests in
//! `src/observer.rs` by covering the *temporal* detection logic (baseline drift
//! across steps), trace violations, threshold tuning, and history tracking.

use ndarray::{array, Array2};
use num_complex::Complex64;

use qrasp_core::observer::{Anomaly, ObserverConfig, StateObserver};

/// Convenience: build the pure state |0><0| (purity 1, trace 1).
fn pure_zero() -> Array2<Complex64> {
    array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

/// Convenience: the maximally mixed 1-qubit state I/2 (purity 0.5, trace 1).
fn maximally_mixed() -> Array2<Complex64> {
    array![
        [Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
    ]
}

/// A density matrix with trace != 1 (here trace = 0.9): a non-trace-preserving
/// op, i.e. a tampered / unnormalized / cloned state.
fn untraced() -> Array2<Complex64> {
    array![
        [Complex64::new(0.6, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.3, 0.0)],
    ]
}

#[test]
fn first_observation_of_a_pure_state_is_nominal() {
    let mut obs = StateObserver::new(ObserverConfig::default());
    // With no prior baseline, a well-formed pure state must be nominal.
    assert_eq!(obs.observe(0, &pure_zero()), Anomaly::Nominal);
}

#[test]
fn stable_pure_trajectory_raises_no_anomaly() {
    let mut obs = StateObserver::new(ObserverConfig::default());
    for step in 0..5 {
        assert_eq!(obs.observe(step, &pure_zero()), Anomaly::Nominal);
    }
    assert_eq!(obs.history().len(), 5);
}

#[test]
fn purity_drop_between_steps_is_flagged_with_correct_delta() {
    let mut obs = StateObserver::new(ObserverConfig::default());
    assert_eq!(obs.observe(0, &pure_zero()), Anomaly::Nominal);

    match obs.observe(1, &maximally_mixed()) {
        Anomaly::PurityDrop { from, to, delta } => {
            assert!((from - 1.0).abs() < 1e-12, "from purity should be 1.0");
            assert!((to - 0.5).abs() < 1e-12, "to purity should be 0.5");
            assert!((delta - 0.5).abs() < 1e-12, "delta should be 0.5");
        }
        other => panic!("expected PurityDrop, got {other:?}"),
    }
}

#[test]
fn purity_increase_is_not_an_anomaly() {
    // Recohering (mixed -> pure) is not a leak; only drops are anomalous.
    let mut obs = StateObserver::new(ObserverConfig::default());
    assert_eq!(obs.observe(0, &maximally_mixed()), Anomaly::Nominal);
    assert_eq!(obs.observe(1, &pure_zero()), Anomaly::Nominal);
}

#[test]
fn large_purity_drop_is_flagged_but_tolerated_under_a_loose_threshold() {
    // A pure -> maximally-mixed transition is a 0.5 drop.
    let mut obs = StateObserver::new(ObserverConfig::default());
    assert_eq!(obs.observe(0, &pure_zero()), Anomaly::Nominal);
    // 1.0 -> 0.5 is a 0.5 drop, far above the default 0.01 threshold.
    assert!(
        matches!(obs.observe(1, &maximally_mixed()), Anomaly::PurityDrop { .. }),
        "a 0.5 drop must flag under the default threshold",
    );

    // With a threshold looser than the drop, the same transition is tolerated.
    let mut tolerant =
        StateObserver::new(ObserverConfig { purity_drop_threshold: 0.6 });
    assert_eq!(tolerant.observe(0, &pure_zero()), Anomaly::Nominal);
    assert_eq!(tolerant.observe(1, &maximally_mixed()), Anomaly::Nominal);
}

#[test]
fn threshold_boundary_is_strict_greater_than() {
    // `classify` flags only when delta > threshold (strict). Use exactly
    // representable binary fractions so the boundary is exact, not a
    // floating-point coin flip: pure (purity 1.0) -> I/2 (purity 0.5) is a
    // delta of exactly 0.5.
    let mut at_boundary =
        StateObserver::new(ObserverConfig { purity_drop_threshold: 0.5 });
    assert_eq!(at_boundary.observe(0, &pure_zero()), Anomaly::Nominal);
    // delta == 0.5, threshold == 0.5, and 0.5 > 0.5 is false -> Nominal.
    assert_eq!(at_boundary.observe(1, &maximally_mixed()), Anomaly::Nominal);

    // Just below the delta, the identical transition flags.
    let mut just_under =
        StateObserver::new(ObserverConfig { purity_drop_threshold: 0.49 });
    assert_eq!(just_under.observe(0, &pure_zero()), Anomaly::Nominal);
    assert!(
        matches!(
            just_under.observe(1, &maximally_mixed()),
            Anomaly::PurityDrop { .. }
        ),
        "a 0.5 drop must flag when the threshold is 0.49",
    );
}

#[test]
fn trace_violation_takes_precedence_over_purity() {
    // A state whose trace deviates from 1.0 must be flagged as a trace
    // violation regardless of purity, and before any purity comparison.
    let mut obs = StateObserver::new(ObserverConfig::default());
    assert_eq!(obs.observe(0, &pure_zero()), Anomaly::Nominal);

    match obs.observe(1, &untraced()) {
        Anomaly::TraceViolation { trace } => {
            assert!((trace - 0.9).abs() < 1e-12, "trace should be 0.9");
        }
        other => panic!("expected TraceViolation, got {other:?}"),
    }
}

#[test]
fn trace_violation_on_first_observation_is_flagged() {
    // Even with no baseline, a non-trace-preserving state is anomalous.
    let mut obs = StateObserver::new(ObserverConfig::default());
    match obs.observe(0, &untraced()) {
        Anomaly::TraceViolation { trace } => assert!((trace - 0.9).abs() < 1e-12),
        other => panic!("expected TraceViolation, got {other:?}"),
    }
}

#[test]
fn history_records_every_observation_in_order() {
    let mut obs = StateObserver::new(ObserverConfig::default());
    obs.observe(10, &pure_zero());
    obs.observe(20, &maximally_mixed());
    obs.observe(30, &pure_zero());

    let history = obs.history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].step, 10);
    assert_eq!(history[1].step, 20);
    assert_eq!(history[2].step, 30);
    assert!((history[0].purity - 1.0).abs() < 1e-12);
    assert!((history[1].purity - 0.5).abs() < 1e-12);
}

#[test]
fn purity_is_computed_against_immediate_predecessor_not_global_min() {
    // Baseline is the *previous* step, so mixed -> mixed (no further drop) is
    // nominal even though purity is below the original pure baseline.
    let mut obs = StateObserver::new(ObserverConfig::default());
    assert_eq!(obs.observe(0, &pure_zero()), Anomaly::Nominal);
    assert!(matches!(
        obs.observe(1, &maximally_mixed()),
        Anomaly::PurityDrop { .. }
    ));
    // Step 2 is the same mixed state: no drop relative to step 1 -> Nominal.
    assert_eq!(obs.observe(2, &maximally_mixed()), Anomaly::Nominal);
}
