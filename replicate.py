"""Master replication script — run everything in < 10 seconds.

Usage:
    pip install -r requirements.txt
    python replicate.py
"""
import sys
import pathlib
import subprocess
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def import_or_run(name):
    """Import a script module and run its main function if it has one."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main()
    return mod


def run_module(name):
    """Run a script as a subprocess (captures all stdout to terminal)."""
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {name}")
    print(f"{'=' * 70}")
    result = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py")],
                            capture_output=False, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    steps = [
        ("01_replicate_prevalence", "Prevalence tables (bills + CREC)"),
        ("02_replicate_regressions", "119th Congress regression models"),
        ("03_replicate_mechanisms", "Sector mechanism tests"),
        ("04_generate_figures", "Effort-style figures"),
    ]

    all_ok = True
    for name, desc in steps:
        print(f"\n{'=' * 70}")
        print(f"  STEP: {desc}")
        print(f"{'=' * 70}")
        rc = run_module(name)
        if rc != 0:
            print(f"  ⚠  {name} failed with code {rc}")
            all_ok = False

    if all_ok:
        print(f"\n{'=' * 70}")
        print("  ✓ All replication steps completed successfully.")
        print(f"  Figures saved to: {ROOT / 'figures' / 'fig1_prevalence_quarterly.png'}")
        print(f"                    {ROOT / 'figures' / 'fig2_crec_extensions.png'}")
        print(f"                    {ROOT / 'figures' / 'fig3_predictor_forest.png'}")
        print(f"{'=' * 70}")
    else:
        print(f"\n⚠  Some steps failed. Check output above.")
        sys.exit(1)