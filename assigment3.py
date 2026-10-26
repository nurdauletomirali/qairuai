from fractions import Fraction as F

states = ["HAZARD", "SAFE"]

# Жолдар — алдыңғы күй, бағандар — келесі күй.
T = [
    [F(8, 10), F(2, 10)],
    [F(1, 10), F(9, 10)]
]

emission = {
    "ALARM": [F(9, 10), F(1, 10)],
    "NORMAL": [F(1, 10), F(9, 10)]
}

readings = ["ALARM", "NORMAL", "ALARM", "ALARM"]


def bayes(prior_h, observation):
    """Бір бақылаудан кейінгі қауіп ықтималдығы."""
    likelihood_h, likelihood_s = emission[observation]

    evidence = (
        prior_h * likelihood_h
        + (1 - prior_h) * likelihood_s
    )

    if evidence == 0:
        raise ValueError("Бақылаудың ықтималдығы нөл.")

    return prior_h * likelihood_h / evidence


def predict(belief):
    """Қозғалыстан кейінгі күй үлестірімін болжау."""
    return [
        sum(belief[i] * T[i][j] for i in range(2))
        for j in range(2)
    ]


def update(predicted, observation):
    """Датчик көрсеткішімен жаңарту және нормалау."""
    weights = [
        predicted[i] * emission[observation][i]
        for i in range(2)
    ]

    total = sum(weights)

    if total == 0:
        raise ValueError("Бақылаудың ықтималдығы нөл.")

    return [weight / total for weight in weights]


def main():
    prior_h = F(1, 5)  # P(HAZARD) = 0.20

    print("A. БАЙЕС ЕСЕПТЕРІ")
    for observation in ["ALARM", "NORMAL"]:
        posterior = bayes(prior_h, observation)
        print(
            f"P(HAZARD | {observation}) = "
            f"{posterior} = {float(posterior):.6f} "
            f"({float(posterior):.2%})"
        )

    print("\nD. СЕЗІМТАЛДЫҚ ТАЛДАУЫ")
    print("Бастапқы P(H) | ALARM кейін | NORMAL кейін")

    for prior in [F(5, 100), F(20, 100), F(50, 100), F(80, 100)]:
        alarm = bayes(prior, "ALARM")
        normal = bayes(prior, "NORMAL")

        print(
            f"{float(prior):13.2%} | "
            f"{float(alarm):11.2%} | "
            f"{float(normal):12.2%}"
        )

    print("\nC. HMM СҮЗГІЛЕУІ")

    # Бастапқы үлестірім t0 сәтіне сәйкес келеді.
    belief = [prior_h, 1 - prior_h]

    for step, observation in enumerate(readings, start=1):
        predicted = predict(belief)
        belief = update(predicted, observation)

        assert sum(belief) == 1
        assert all(0 <= value <= 1 for value in belief)

        most_likely = states[belief.index(max(belief))]

        print(f"\nt{step}: {observation}")
        print(
            "Болжам:",
            [round(float(value), 6) for value in predicted]
        )
        print(
            "Жаңарту:",
            [round(float(value), 6) for value in belief]
        )
        print("Ең ықтимал күй:", most_likely)

    print(f"\nСоңғы P(HAZARD) = {float(belief[0]):.2%}")
    print(f"Соңғы P(SAFE) = {float(belief[1]):.2%}")

    assert bayes(prior_h, "ALARM") == F(9, 13)
    assert bayes(prior_h, "NORMAL") == F(1, 37)
    assert all(sum(row) == 1 for row in T)


if __name__ == "__main__":
    main()
