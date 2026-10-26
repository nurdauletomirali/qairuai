states = ["A", "B", "C"]

T = [
    [0.7, 0.3, 0.0],
    [0.2, 0.6, 0.2],
    [0.0, 0.3, 0.7]
]

emission = {
    "RED": [0.8, 0.5, 0.2],
    "BLUE": [0.2, 0.5, 0.8]
}

readings = ["RED", "BLUE", "BLUE", "RED"]


def predict(b, T):
    return [
        sum(b[i] * T[i][j] for i in range(3))
        for j in range(3)
    ]


def update(pred, likelihood):
    weights = [
        pred[i] * likelihood[i]
        for i in range(3)
    ]

    total = sum(weights)

    if total <= 0:
        raise ValueError("Бақылаудың ықтималдығы нөл.")

    return [w / total for w in weights]


def run(initial, emissions, title):
    belief = initial[:]
    print("\n" + title)

    for step, reading in enumerate(readings, start=1):
        predicted = predict(belief, T)
        belief = update(predicted, emissions[reading])

        assert abs(sum(belief) - 1) < 1e-12
        assert all(0 <= v <= 1 for v in belief)

        location = states[belief.index(max(belief))]

        print(
            step,
            reading,
            [round(v, 4) for v in belief],
            "Most likely:",
            location
        )

    return belief


baseline = run(
    [0.5, 0.3, 0.2],
    emission,
    "Негізгі модель"
)

changed_prior = run(
    [0.2, 0.2, 0.6],
    emission,
    "1-эксперимент: жаңа бастапқы үлестірім"
)

weak_emission = {
    "RED": [0.6, 0.5, 0.4],
    "BLUE": [0.4, 0.5, 0.6]
}

changed_sensor = run(
    [0.5, 0.3, 0.2],
    weak_emission,
    "2-эксперимент: әлсіздеу датчик"
)
