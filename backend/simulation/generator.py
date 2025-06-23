import random
from typing import Dict, Any
from backend.ingest.input_schema import EnvironmentInput
from backend.physics.layer import AtmosphereLayer
from backend.physics.propagation import BeamSimulator

def random_valid_environment() -> Dict[str, Any]:
    # Generate random but valid environment parameters
    return {
        "lat_tx": random.uniform(-90, 90),
        "lon_tx": random.uniform(-180, 180),
        "height_tx": random.uniform(1, 50),
        "material_tx": random.choice(["concrete", "steel", "wood", "white_paint"]),
        "lat_rx": random.uniform(-90, 90),
        "lon_rx": random.uniform(-180, 180),
        "height_rx": random.uniform(1, 50),
        "material_rx": random.choice(["concrete", "steel", "wood", "white_paint"]),
        "fog_density": random.uniform(0, 1),
        "rain_rate": random.uniform(0, 20),
        "surface_temp": random.uniform(-10, 50),
        "ambient_temp": random.uniform(-10, 50),
        "wavelength_nm": random.uniform(700, 1600),
    }

def build_layers_from(inputs: Dict[str, Any]):
    # For simplicity, assume a single atmospheric layer
    dz = 1.0  # 1 meter layer
    temp_gradient = (inputs["surface_temp"] - inputs["ambient_temp"]) / inputs["height_tx"]
    layer = AtmosphereLayer(
        dz=dz,
        fog_density=inputs["fog_density"],
        rain_rate=inputs["rain_rate"],
        temp_gradient=temp_gradient
    )
    return [layer]

def generate_sample() -> Dict[str, Any]:
    inputs = random_valid_environment()
    layers = build_layers_from(inputs)
    sim = BeamSimulator(layers, power_init=1.0)
    result = sim.simulate()
    return {**inputs, **result}

def generate_dataset(n_samples: int, output_csv: str):
    import csv
    samples = [generate_sample() for _ in range(n_samples)]
    if samples:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=samples[0].keys())
            writer.writeheader()
            writer.writerows(samples)
