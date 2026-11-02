def get_clean_weight_string(weights: tuple[float, ...]) -> str:
    return "_".join(
        str(weight).replace(".", "_").replace("-", "neg") for weight in weights
    )
