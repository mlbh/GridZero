from typing import Dict


def normalize_mix(mix: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure energy mix percentages sum to 100
    """

    total = sum(mix.values())

    if total == 0:
        raise ValueError("Energy mix cannot sum to zero")

    return {k: v / total for k, v in mix.items()}


def calculate_surplus(generation_mw: float, demand_mw: float):

    surplus = generation_mw - demand_mw

    return {
        "surplus_mw": max(surplus, 0),
        "deficit_mw": max(-surplus, 0)
    }


def battery_requirement(deficit_mw: float, hours: int = 4):
    """
    Estimate battery storage needed.

    Example: deficit 100 MW for 4 hours → 400 MWh battery
    """

    energy_needed = deficit_mw * hours

    return {
        "battery_required_mwh": energy_needed
    }


def simulate_grid(
    predicted_mix: Dict[str, float],
    user_mix: Dict[str, float],
    demand_mw: float,
    total_generation_mw: float
):

    user_mix = normalize_mix(user_mix)

    surplus_data = calculate_surplus(total_generation_mw, demand_mw)

    battery_data = battery_requirement(surplus_data["deficit_mw"])

    return {
        "predicted_mix": predicted_mix,
        "user_mix": user_mix,
        "surplus": surplus_data,
        "battery": battery_data
    }


