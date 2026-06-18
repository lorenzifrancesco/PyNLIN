IEEE_DOUBLE_COLUMN_ONE_COLUMN_WIDTH_IN = 88.9 / 25.4


def scale_figsize_to_ieee_column(width_in: float, height_in: float) -> tuple[float, float]:
    """Preserve aspect ratio while forcing the IEEEtran single-column width."""
    if width_in <= 0.0:
        raise ValueError(f"width_in must be positive, got {width_in}")
    if height_in <= 0.0:
        raise ValueError(f"height_in must be positive, got {height_in}")
    return (
        IEEE_DOUBLE_COLUMN_ONE_COLUMN_WIDTH_IN,
        IEEE_DOUBLE_COLUMN_ONE_COLUMN_WIDTH_IN * float(height_in) / float(width_in),
    )
