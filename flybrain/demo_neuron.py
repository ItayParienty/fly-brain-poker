"""Interactive demo: watch a single LIF neuron charge, leak and fire.

Run this to build intuition before wiring neurons into a brain.
"""

from flybrain import console_utf8  # noqa: F401

from flybrain.neuron import LIFNeuron


def show_trace(input_current, steps=25):
    n = LIFNeuron("demo")
    spikes = 0
    print(f"\nגירוי קבוע של {input_current} בכל צעד:")
    for t in range(steps):
        n.step(input_current)
        bar = "#" * int(n.potential * 40)
        if n.spiked:
            spikes += 1
            print(f"  t={t:2d} |{'':<40}|  <<< ירייה!")
        else:
            print(f"  t={t:2d} |{bar:<40}|  מתח={n.potential:.2f}")
    print(f"  סה\"כ {spikes} יריות ב-{steps} צעדים  ->  קצב ירי = {spikes / steps:.2f}")


def firing_rate(input_current, steps=200):
    n = LIFNeuron("x")
    return sum(n.step(input_current) or n.spiked for _ in range(steps)) / steps


if __name__ == "__main__":
    print("=" * 60)
    print("חלק 1: נוירון בודד עם גירוי חזק")
    print("=" * 60)
    show_trace(0.5)

    print("\n" + "=" * 60)
    print("חלק 2: אותו נוירון בדיוק, גירוי חלש")
    print("=" * 60)
    show_trace(0.15)

    print("\n" + "=" * 60)
    print("חלק 3: קצב הירי כפונקציה של עוצמת הגירוי")
    print("=" * 60)
    for current in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0]:
        rate = firing_rate(current)
        print(f"  גירוי {current:>4}  ->  קצב ירי {rate:.2f}  {'*' * int(rate * 50)}")
