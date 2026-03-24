"""
Runs longhealth_train.py on Modal with distributed training across multiple GPUs.

Usage (32 GPUs = 4 nodes × 8 H100s):
    modal run infra/modal_train_longhealth.py

Configurable via environment variables:
    GPU_TYPE        GPU type (default: H100)
    GPU_COUNT       GPUs per node (default: 8)
    NUM_NODES       Number of nodes (default: 4, so 32 GPUs total)
    MODEL           Model to train: "llama" or "qwen" (default: llama)
    NUM_TOKENS      KV cache size in tokens (default: 2048)
    BRANCH          Git branch to use (default: main)
    MASTER_PORT     Port for torchrun rendezvous (default: 12355)

Example (8 GPUs on a single node):
    GPU_COUNT=8 NUM_NODES=1 modal run infra/modal_train_longhealth.py
"""

import os
import subprocess

import modal

GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 8))
NUM_NODES = int(os.environ.get("NUM_NODES", 4))
MODEL = os.environ.get("MODEL", "llama")
NUM_TOKENS = os.environ.get("NUM_TOKENS", "2048")
BRANCH = os.environ.get("BRANCH", "main")
MASTER_PORT = os.environ.get("MASTER_PORT", "12355")


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


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("cartridges-output", create_if_missing=True)

app = modal.App(f"jw-longhealth-train-{NUM_NODES * GPU_COUNT}x{GPU_TYPE}")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    secrets=[modal.Secret.from_name("jw-api-keys")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/outputs": output_vol,
    },
    timeout=3600 * 12,
    _experimental_cluster_size=NUM_NODES,
)
def train():
    node_rank = int(os.environ.get("MODAL_RANK", 0))
    master_addr = os.environ.get("MODAL_MASTER_ADDR", "localhost")

    cmd = [
        "torchrun",
        f"--nproc_per_node={GPU_COUNT}",
        f"--nnodes={NUM_NODES}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={MASTER_PORT}",
        "examples/benchmarks/longhealth/longhealth_train.py",
    ]

    env = {
        **os.environ,
        "MODEL": MODEL,
        "NUM_TOKENS": NUM_TOKENS,
        "CARTRIDGES_OUTPUT_DIR": "/root/outputs",
    }

    print(f"[Node {node_rank}/{NUM_NODES}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd="/root/cartridges", env=env, check=True)


@app.local_entrypoint()
def main():
    total_gpus = NUM_NODES * GPU_COUNT
    print(f"Launching longhealth training on {NUM_NODES} node(s) × {GPU_COUNT} {GPU_TYPE}s = {total_gpus} GPUs total")
    print(f"Model: {MODEL}, NUM_TOKENS: {NUM_TOKENS}, Branch: {BRANCH}")
    train.remote()