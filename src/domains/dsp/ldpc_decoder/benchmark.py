#!/usr/bin/env python3
"""Run the opt-in CPU module benchmark and save a machine-labelled JSON report.

Usage: python benchmark.py --build /path/to/build --output report.json
Build the LDPC module test target first. This runner never starts a build.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    root = Path(__file__).resolve().parents[4]
    executable = build / "tests/modules/jetstream-modules-dsp-ldpc_decoder-module-tests"
    if os.name == "nt":
        executable = executable.with_suffix(".exe")
    metadata = {
        "hostname": socket.gethostname(), "os": platform.platform(),
        "architecture": platform.machine(), "logical_cpus": os.cpu_count(),
        "cpu_model": platform.processor() or None,
    }
    if platform.system() == "Linux":
        metadata["cpuinfo"] = Path("/proc/cpuinfo").read_text()
        metadata["meminfo"] = Path("/proc/meminfo").read_text()
        metadata["cpu_model"] = next((line.split(":", 1)[1].strip()
            for line in metadata["cpuinfo"].splitlines()
            if line.startswith("model name")), None)
    elif os.name == "nt":
        metadata["windows_hardware"] = command([
            "powershell", "-NoProfile", "-Command",
            "@{cpu=Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors;"
            "ram=Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory;"
            "gpu=Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion} | ConvertTo-Json -Depth 4"])
    introspection = {}
    for name in ("buildoptions", "compilers", "machines"):
        path = build / "meson-info" / f"intro-{name}.json"
        introspection[name] = json.loads(path.read_text()) if path.exists() else None
    targets = build / "meson-info" / "intro-targets.json"
    if targets.exists():
        introspection["target_compile_settings"] = [
            target for target in json.loads(targets.read_text())
            if target["name"] in ("jetstream", "jetstream-modules-dsp-ldpc_decoder-module-tests")]
    environment = os.environ.copy()
    environment["PATH"] = str(build) + os.pathsep + environment.get("PATH", "")
    environment["LD_LIBRARY_PATH"] = str(build) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory(prefix="ldpc-benchmark-") as temporary:
        raw = Path(temporary) / "timings.json"
        environment["LDPC_BENCHMARK_OUTPUT"] = str(raw)
        subprocess.run([str(executable), "[ldpc_benchmark]"],
                       cwd=root, env=environment, check=True)
        results = json.loads(raw.read_text())
    report = {
        "schema_version": 1,
        "output_format": "U8, one byte per hard-decision bit (0/1), 2560 bytes per frame",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "machine": metadata, "build": introspection,
        "git_revision": command(["git", "rev-parse", "HEAD"], root),
        "git_status": command(["git", "status", "--porcelain"], root),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256((Path(__file__).parent / "test_data/ldpc_corpus.hh").read_bytes()).hexdigest(),
        "measurement_scope": "TestContext CPU compute including output snapshot; setup and correctness checks excluded",
        "workload": "64-frame corpus; clean sigma=0 (100 dB sentinel), -1/1/2/2.5/3/4/6 dB; no failure filtering; rotating inputs; batch 1 and 64 sequential CPU lanes; group -1 mixes all frames",
        "limitations": "warm-cache finite corpus; iteration/convergence counts are reference expectations, not module telemetry; early-stop only, not a fixed-work comparison",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
