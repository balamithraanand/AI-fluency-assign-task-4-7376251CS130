"""Day 4: Estimate memory requirements for open LLMs."""

BYTES_PER_PARAM = {
    "FP16": 2.00,
    "Q8_0": 1.00,
    "Q6_K": 0.81,
    "Q5_K_M": 0.68,
    "Q4_K_M": 0.57,
    "Q3_K_M": 0.43,
}

KV_GB_PER_B_PER_1K = 0.02
OVERHEAD = 1.10


def estimate(params_b, precision="Q4_K_M", context_k=8):
    if precision not in BYTES_PER_PARAM:
        raise ValueError(
            f"Unknown precision {precision}. "
            f"Choose from {list(BYTES_PER_PARAM)}"
        )

    weights_gb = params_b * BYTES_PER_PARAM[precision]
    kv_gb = params_b * context_k * KV_GB_PER_B_PER_1K
    total_gb = (weights_gb + kv_gb) * OVERHEAD

    return weights_gb, kv_gb, total_gb


def verdict(total_gb, available_gb):
    if total_gb <= available_gb * 0.7:
        return "fits comfortably"
    if total_gb <= available_gb:
        return "fits, but tight"
    return "does NOT fit"


def report(name, params_b, precision, context_k, available_gb):
    weights, kv, total = estimate(
        params_b,
        precision,
        context_k
    )

    print(
        f"{name:<22} "
        f"{precision:<7} "
        f"{params_b:>5.1f}B "
        f"ctx {context_k:>3}K "
        f"weights {weights:>6.2f} GB "
        f"kv {kv:>5.2f} GB "
        f"total {total:>6.2f} GB "
        f"-> {verdict(total, available_gb)}"
    )


if __name__ == "__main__":
    AVAILABLE_GB = 8.0

    print(f"Memory available: {AVAILABLE_GB} GB\n")

    print("=== Model Estimates ===")

    report("1.5B Q4 model", 1.5, "Q4_K_M", 8, AVAILABLE_GB)
    report("8B Q4 model", 8.0, "Q4_K_M", 8, AVAILABLE_GB)
    report("8B FP16 model", 8.0, "FP16", 8, AVAILABLE_GB)
    report("8B Q8 model", 8.0, "Q8_0", 8, AVAILABLE_GB)
    report("14B Q4 model", 14.0, "Q4_K_M", 8, AVAILABLE_GB)
    report("30B Q4 model", 30.0, "Q4_K_M", 8, AVAILABLE_GB)

    print("\n=== Context Length Experiment ===")

    for context_k in (4, 8, 16, 32):
        report(
            "8B Q4 model",
            8.0,
            "Q4_K_M",
            context_k,
            AVAILABLE_GB
        )

    print("\n=== Quantization Experiment ===")

    for precision in (
        "Q3_K_M",
        "Q4_K_M",
        "Q5_K_M",
        "Q8_0",
        "FP16"
    ):
        report(
            "8B model",
            8.0,
            precision,
            8,
            AVAILABLE_GB
        )