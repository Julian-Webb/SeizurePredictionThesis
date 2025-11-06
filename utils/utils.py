def safe_float_to_int(num: float) -> int:
    if num != int(num):
        raise ValueError(f"Number {num} has decimal values")
    return int(num)
