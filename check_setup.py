"""
check_setup.py  -  run this FIRST. It answers one question:
"which quantization methods can this machine actually run?"

    python check_setup.py

It prints your CPU, your PyTorch quantization backend, and then TESTS every
method on a tiny model. At the end it gives you the exact run_all.py command
that will work here. Nothing is downloaded and nothing is measured, so it
takes a couple of seconds.
"""
import sys

import torch

from src import config
from src.environment import collect_environment
from src.quantization import METHOD_DESCRIPTIONS, available_engines, probe_method, select_quantized_engine


def main():
    print("=" * 72)
    print("SETUP CHECK")
    print("=" * 72)

    engines = available_engines()
    if not engines:
        print("\nThis PyTorch build has NO quantized engine, so dynamic quantization")
        print("cannot run at all. Reinstall the official CPU wheel:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return 1

    engine = select_quantized_engine()
    env = collect_environment()

    print(f"\nPython {env['python']} | torch {env['torch']} | {env['os']}")
    print(f"CPU   : {env['cpu_model']}")
    print(f"Cores : {env['cpu_physical_cores']} physical / {env['cpu_logical_cores']} logical"
          f" | RAM {env['ram_total_gb']} GB")
    print(f"Quantized engines available : {engines}")
    print(f"Engine selected             : {engine}")

    capability = env.get("torch_cpu_capability", "unknown")
    print(f"Vector instructions PyTorch uses : {capability}")
    flags = env.get("cpu_flags")
    if isinstance(flags, dict):
        have = [f for f, ok in flags.items() if ok is True]
        if have:
            print(f"CPU flags reported by the OS     : {', '.join(have)}")

    has_fbgemm = "fbgemm" in engines or "x86" in engines
    if not has_fbgemm:
        print("\nNOTE: FBGEMM is not in this PyTorch build.")
        print("      FBGEMM is the classic x86 INT8 backend, and it is the only one that")
        print("      implements FP16 dynamic quantization. Its absence is a property of")
        print("      the BUILD (PyTorch is retiring torch.ao.quantization), NOT of your CPU:")
        print(f"      PyTorch reports this CPU as '{capability}'.")
        print("      INT8 still runs through oneDNN, which uses the same AVX-512/VNNI")
        print("      instructions, so the INT8 speedup should still be real.")
        print("      Describe it that way in the report - do not blame the hardware.")

    print("\nTesting each quantization method on a tiny model:")
    supported, unsupported = [], []
    for method in config.ALL_METHODS:
        ok, reason = probe_method(method)
        default_marker = " (in the default run)" if method in config.METHODS else ""
        if ok:
            supported.append(method)
            print(f"  [ OK ]   {method}{default_marker}")
        else:
            unsupported.append(method)
            print(f"  [ FAIL ] {method}{default_marker}")
            print(f"           {reason}")

    print("\n" + "=" * 72)
    if len(supported) < 3:
        print("Fewer than 2 quantization methods work here, so the task cannot be done")
        print("on this machine as-is. Please send the output above for help.")
        return 1

    runnable_default = [m for m in config.METHODS if m in supported]
    if len(runnable_default) == len(config.METHODS):
        print("Everything in the default run works. Next:")
        print("  python run_all.py --smoke")
        print("  python run_all.py")
    else:
        # Fall back in an order that keeps the COMPARISON interesting: the two
        # chosen methods should differ as much as possible, not just in one knob.
        fallback_order = ["dq_int8_per_tensor", "dq_int8_linear_emb",
                          "dq_int8_per_channel", "dq_fp16"]
        chosen = ["fp32"] + [m for m in fallback_order if m in supported][:2]
        print("Some default methods do not work here. Use this command instead:")
        print(f"  python run_all.py --smoke --methods {' '.join(chosen)}")
        print(f"  python run_all.py --methods {' '.join(chosen)}")
        print("\nThat is still FP32 + 2 ways of dynamic quantization, as the task asks.")
        print("These two differ in WHICH layers are quantized, so they give a clear")
        print("contrast: INT8 Linear only vs INT8 Linear + 8-bit embeddings.")
        print("The method that failed is worth one sentence in the report.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
