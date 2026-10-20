STATES = ["A", "B", "C", "D", "E", "S", "F", "H", "I"]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
TERMINALS = {"G", "X"}
GRID = [["A", "B", "C", "G"], ["D", "#", "E", "X"],
        ["S", "F", "H", "I"]]
POS = {s: (r, c) for r, row in enumerate(GRID)
       for c, s in enumerate(row) if s != "#"}
AT = {pos: s for s, pos in POS.items()}
MOVES = {"UP": (-1, 0), "DOWN": (1, 0),
         "LEFT": (0, -1), "RIGHT": (0, 1)}
DIRECTIONS = {"UP": ("UP", "LEFT", "RIGHT"),
              "DOWN": ("DOWN", "RIGHT", "LEFT"),
              "LEFT": ("LEFT", "DOWN", "UP"),
              "RIGHT": ("RIGHT", "UP", "DOWN")}
BASE = dict(gamma=0.9, p_main=0.8, goal=10.0,
            danger=-10.0, step=-0.1)
THETA = 0.001


def transitions(s, a, p):
    if s in TERMINALS:
        return [(1.0, s, 0.0)]
    side = (1.0 - p["p_main"]) / 2
    result = {}
    for direction, prob in zip(DIRECTIONS[a],
                               (p["p_main"], side, side)):
        r, c = POS[s]
        dr, dc = MOVES[direction]
        nxt = AT.get((r + dr, c + dc), s)
        result[nxt] = result.get(nxt, 0.0) + prob
    return [(prob, nxt, p["goal"] if nxt == "G" else
             p["danger"] if nxt == "X" else p["step"])
            for nxt, prob in result.items()]


def q_value(s, a, values, p):
    return sum(prob * (reward + p["gamma"] *
               values.get(nxt, 0.0))
               for prob, nxt, reward in transitions(s, a, p))


def value_iteration(p, theta=THETA):
    values = {s: 0.0 for s in STATES}
    history = [values.copy()]
    for iteration in range(1, 10001):
        new_values = {
            s: max(q_value(s, a, values, p) for a in ACTIONS)
            for s in STATES
        }
        delta = max(abs(new_values[s] - values[s]) for s in STATES)
        values = new_values
        history.append(values.copy())
        if delta < theta:
            return values, history, iteration, delta
    raise RuntimeError("Мәндер итерациясы жинақталмады")


def policy_and_q(values, p):
    q = {s: {a: q_value(s, a, values, p) for a in ACTIONS}
         for s in STATES}
    policy = {s: max(ACTIONS, key=q[s].get) for s in STATES}
    return policy, q


def show_policy(policy):
    for row in GRID:
        print(" | ".join(f"{s}: {policy[s]}" if s in policy
                         else s for s in row))


def main():
    for s, a in [("F", "RIGHT"), ("E", "RIGHT"), ("C", "RIGHT")]:
        tr = transitions(s, a, BASE)
        assert abs(sum(t[0] for t in tr) - 1.0) < 1e-12
        print(f"T({s}, {a}) = {tr}")
    for rewards, gamma in [([2, 4, 6], 0.5),
                           ([2, 4, 6], 0.9), ([0, 0, 10], 0.5)]:
        utility = sum(gamma ** k * r for k, r in enumerate(rewards))
        print(f"Марапаттар={rewards}, gamma={gamma}, U={utility:.2f}")
    values, history, n, delta = value_iteration(BASE)
    policy, q = policy_and_q(values, BASE)
    print(f"Итерация: {n}; delta={delta:.6f}")
    print("Күй         V0         V1         V2")
    for s in STATES:
        print(s, *(f"{history[k][s]:10.4f}" for k in range(3)))
    print(f"Q2(E, UP) = {q_value('E', 'UP', history[1], BASE):.4f}")
    print("Күй       V(s)        UP      DOWN      LEFT     RIGHT  Саясат")
    for s in STATES:
        qs = " ".join(f"{q[s][a]:9.4f}" for a in ACTIONS)
        print(f"{s:3} {values[s]:10.4f} {qs}  {policy[s]}")
    print("G: V=0; X: V=0; эпизод аяқталады")
    show_policy(policy)
    experiments = [
        ("gamma=0.2", {"gamma": 0.2}),
        ("p_main=0.6", {"p_main": 0.6}),
        ("danger=-30", {"danger": -30.0}),
        ("step=-1", {"step": -1.0}),
    ]
    for name, change in experiments:
        p = dict(BASE, **change)
        v, _, count, _ = value_iteration(p)
        pi, _ = policy_and_q(v, p)
        changed = [f"{s}: {policy[s]} -> {pi[s]}"
                   for s in STATES if pi[s] != policy[s]]
        print(f"\n{name}; итерация={count}; V(S)={v['S']:.4f}")
        print("Өзгерістер:", "; ".join(changed) or "жоқ")
        print("Мәндер:", {s: round(v[s], 4) for s in STATES})
        show_policy(pi)


if __name__ == "__main__":
    main()
