"""
Runs baseline_longhealth.py on Modal (ICL baseline via external vLLM API).

Usage:
    modal run infra/modal_baseline_longhealth.py

Configurable via environment variables:
    MODEL       Model to evaluate: "llama" or "qwen" (default: llama)
    BRANCH      Git branch to use (default: main)
"""

import os
import subprocess

import modal

MODEL = os.environ.get("MODEL", "llama")
BRANCH = os.environ.get("BRANCH", "main")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://$GITHUB_TOKEN@github.com/JonathanWenger/cartridges.git /root/cartridges",
        secrets=[modal.Secret.from_name("jw-api-keys")],
    )
    .run_commands("cd /root/cartridges && git fetch --all")
    .run_commands("cd /root/cartridges && pip install -e .[dev]")
)
if BRANCH != "main":
    image = image.run_commands(
        f"cd /root/cartridges && git fetch --all && git checkout --track origin/{BRANCH}"
    )
image = image.run_commands("cd /root/cartridges && git pull")

output_vol = modal.Volume.from_name("cartridges-output", create_if_missing=True)

app = modal.App(f"jw-longhealth-baseline")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("jw-api-keys")],
    volumes={
        "/root/outputs": output_vol,
    },
    timeout=3600 * 4,
)
def baseline():
    cmd = ["python", "examples/benchmarks/longhealth/baseline_longhealth.py"]

    env = {
        **os.environ,
        "MODEL": MODEL,
        "CARTRIDGES_OUTPUT_DIR": "/root/outputs",
    }

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd="/root/cartridges", env=env, check=True)


@app.local_entrypoint()
def main():
    print(f"Launching longhealth ICL baseline, model: {MODEL}, branch: {BRANCH}")
    baseline.remote()