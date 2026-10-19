"""Practice 6. Omirali Nurdaulet, group 1, 15.10.2026.

Run with Python 3. No external packages are required.
Convention: the first observation updates prior=0.30 directly.
"""


def bayes(prior_obstacle, likelihood_o, likelihood_c):
    """P(obstacle | reading), using the two likelihoods."""
    score_o = prior_obstacle * likelihood_o
    score_c = (1.0 - prior_obstacle) * likelihood_c
    total = score_o + score_c
    if total == 0:
        raise ValueError("The observation has zero probability.")
    return score_o / total


def hmm_filter(observations, prior=0.30, false_alarm=0.20):
    """Return predictions, weights, normalizers and posteriors."""
    emission = {
        "BLOCKED": (0.80, false_alarm),
        "CLEAR": (0.20, 1.0 - false_alarm),
    }
    belief = prior
    rows = []
    for t, reading in enumerate(observations, start=1):
        predicted = belief if t == 1 else (
            0.70 * belief + 0.20 * (1.0 - belief)
        )
        likelihood_o, likelihood_c = emission[reading]
        score_o = predicted * likelihood_o
        score_c = (1.0 - predicted) * likelihood_c
        total = score_o + score_c
        belief = bayes(predicted, likelihood_o, likelihood_c)
        rows.append((t, reading, predicted, score_o, score_c,
                     total, belief, 1.0 - belief))
    return rows


def main():
    print("Practice 6 | group 1 | 15.10.2026")
    print("A: Single-observation Bayes update")
    print(f"P(O | BLOCKED) = {bayes(0.30, 0.80, 0.20):.6f}")
    print(f"P(O | CLEAR)   = {bayes(0.30, 0.20, 0.80):.6f}")

    observations = ["BLOCKED", "CLEAR", "BLOCKED", "BLOCKED"]
    rows = hmm_filter(observations)
    print("\nC / D3: HMM (no transition before t1)")
    print("t reading  predicted_O updated_O updated_C")
    for t, reading, pred, so, sc, total, po, pc in rows:
        print(f"{t} {reading:7} {pred:11.6f} "
              f"{po:9.6f} {pc:9.6f}")

    print("\nD1: Change the prior; same sensor")
    print("prior  P(O | BLOCKED)")
    for prior in (0.10, 0.30, 0.50, 0.70):
        result = bayes(prior, 0.80, 0.20)
        print(f"{prior:.2f}   {result:.6f}")

    print("\nD2: Change false alarm; prior = 0.30")
    print("P(B|C) P(CLEAR|C) P(O|BLOCKED)")
    for false_alarm in (0.20, 0.40):
        result = bayes(0.30, 0.80, false_alarm)
        print(f"{false_alarm:.2f}   {1-false_alarm:.2f}       "
              f"{result:.6f}")


if __name__ == "__main__":
    main()
