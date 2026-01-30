# test_simulation.py - Quick test for governance_logic

from components.governance_logic import run_simple_simulation

if __name__ == "__main__":
    print("Running simple GAGS simulation test...\n")

    result = run_simple_simulation(
        bias_types=["demographic", "historical", "selection"],
        bias_factor=0.35,
        poison_rate=0.12,
        n_samples=5000  # smaller for faster testing
    )

    print("Simulation Results:")
    for key, value in result.items():
        print(f"  {key: <20}: {value}")