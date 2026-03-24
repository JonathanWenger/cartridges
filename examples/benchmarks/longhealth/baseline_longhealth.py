import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import OpenAIClient
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.evaluate import (
    GenerationEvalConfig,
    GenerationEvalRunConfig,
    ICLBaseline,
)
from cartridges.utils.wandb import WandBConfig

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "llama":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    base_url = os.environ.get("CARTRIDGES_VLLM_LLAMA_3B_URL", "http://localhost:8000")
elif MODEL == "qwen":
    model_name = "Qwen/Qwen3-4b"
    base_url = os.environ.get("CARTRIDGES_VLLM_QWEN3_4B_URL", "http://localhost:8000")
else:
    raise ValueError(f"Invalid model: {MODEL}")

client = OpenAIClient.Config(
    base_url=os.path.join(base_url, "v1"),
    model_name=model_name,
)


SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>

Do not think for too long (only a few sentences, you only have 512 tokens to work with).
"""


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = [
    GenerationEvalRunConfig(
        name=f"longhealth_mc_{patients_str}",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3,
            max_completion_tokens=2048,
            context=LongHealthResource.Config(
                patient_ids=patient_ids,
            ),
        ),
        eval=GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids, 
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=1,
            temperature=0.3,
        ),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(
            tags=[f"longhealth", "genbaseline", f"patients_{patients_str}", "icl"],
        ),
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
