"""Quick test to verify simulation fix"""
from qbes.config_manager import ConfigurationManager

# Load existing config  
config_manager = ConfigurationManager()
config = config_manager.load_config("quick_photosystem_20251023_210518.yaml")

# Reduce steps for quick test
config.num_time_steps = 50

# Run simulation
from qbes.simulation_engine import SimulationEngine
engine = SimulationEngine()
init_result = engine.initialize_simulation(config)

if init_result.is_valid:
    print("[OK] Initialization successful")
    print("[RUN] Running 50-step simulation...")
    results = engine.run_simulation()
    print(f"\n[OK] Simulation completed!")
    print(f"   State trajectory length: {len(results.state_trajectory)}")
    print(f"   Energy trajectory length: {len(results.energy_trajectory)}")
    print(f"   Match: {'[OK] YES' if len(results.state_trajectory) == len(results.energy_trajectory) else '[FAIL] NO'}")
else:
    print("[FAIL] Initialization failed:", init_result.errors)
