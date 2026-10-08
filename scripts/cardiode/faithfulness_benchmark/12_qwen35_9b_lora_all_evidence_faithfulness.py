#!/usr/bin/env python
from _faithfulness_common import FaithfulnessConfig, run_faithfulness

run_faithfulness(
    FaithfulnessConfig(
        run_name="qwen35_9b_lora_all_evidence",
        model_name="Qwen/Qwen3.5-9B",
        is_lora=True,
    )
)