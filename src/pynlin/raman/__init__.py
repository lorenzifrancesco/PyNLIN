from .undepleted import (
    db_per_km_to_np_per_m,
    effective_raman_gain,
    pump_power_for_flat_signal,
    pump_power_coprop,
    pump_power_counterprop,
    signal_power_undepleted_coprop,
    signal_power_undepleted_counterprop,
)

__all__ = [
    "db_per_km_to_np_per_m",
    "effective_raman_gain",
    "FlatnessOptimizationConfig",
    "optimize_profile_flatness",
    "pump_power_for_flat_signal",
    "pump_power_coprop",
    "pump_power_counterprop",
    "signal_power_undepleted_coprop",
    "signal_power_undepleted_counterprop",
]


def __getattr__(name):
    if name in {"FlatnessOptimizationConfig", "optimize_profile_flatness"}:
        from .flatness_optimize import (
            FlatnessOptimizationConfig,
            optimize_profile_flatness,
        )

        exports = {
            "FlatnessOptimizationConfig": FlatnessOptimizationConfig,
            "optimize_profile_flatness": optimize_profile_flatness,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
