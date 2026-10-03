# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Decide child process: imajev typed decisions, plain torch (no vLLM).

Invoked by ``DecideAdapter`` via ``<imajev python> -u -m
giq_imajev_child._decide_child --model <recipe> --weights <adapter dir>
--base <Qwen checkpoint>``. Loads the base plus PEFT LoRA and the trained
readout once, with the hub disabled before transformers is imported, so
nothing is ever fetched.

Per task: compile the payload (``to_request_with_plan``), score one forward
per question per rotation (``TorchDecision.candidate_logits``), average
rotations (``combine_rotations``), apply the adapter's shipped temperature
(``TemperatureCalibrator``), and return Jev's answer shapes via
``to_response``. A task carries ``state`` (record, string or dict),
``questions`` (1-8, noul/choice/score/multi), ``images_b64`` (0-2) and an
optional ``rotations`` int (default 1; 4 is the shipped benchmark config);
``calibration`` selects the artifact (default ``calibration.json``).

Wire protocol: see giq_child (``run_ipc_child_loop``). This module is a
top-level package importing no giq: the imajev interpreter carries
torch/PEFT, never giq's venv.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

# At most 2 images per task. State size is enforced by the Request contract
# (compact UTF-8 JSON, MAX_STATE_BYTES) during to_request_with_plan.
MAX_IMAGES = 2


def _load(base: str, adapter: str):
    import torch
    from torch_decision import TorchDecision

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = TorchDecision(base, device, dtype=torch.bfloat16, max_length=4096)
    if adapter:
        from peft import PeftModel

        engine.model = PeftModel.from_pretrained(engine.model, adapter).eval()
        engine.enable_readout(adapter, trainable=False)
    return engine


def _decode_images(blobs):
    import base64

    from vision_decision.images import load_image_bytes

    images = []
    for position, blob in enumerate(blobs or []):
        try:
            image, _ = load_image_bytes(base64.b64decode(blob))
        except ValueError as exc:
            raise ValueError(f"images[{position}]: {exc}") from exc
        images.append(image)
    return images


def _calibrator(adapter: str, name: str | None):
    from vision_decision.calibration import TemperatureCalibrator

    if not name:
        name = "calibration.json"
    path = os.path.join(adapter, os.path.basename(str(name)))
    if not os.path.isfile(path):
        raise ValueError(f"calibration artifact is missing: {path}")
    return TemperatureCalibrator.load(path)


def _run(models, task: dict) -> dict:
    import torch
    from vision_decision.jev_api import to_request_with_plan, to_response
    from vision_decision.scoring import (
        combine_rotations,
        compile_question,
        cyclic_offsets,
        result_from_logits,
        rotate,
    )

    engine, adapter, model_name = models
    if not isinstance(task.get("questions"), dict) or not task["questions"]:
        raise ValueError("task needs a non-empty 'questions' object")
    state = task.get("state", {})
    blobs = task.get("images_b64") or []
    if len(blobs) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per task, got {len(blobs)}")
    images = _decode_images(blobs)
    payload = {"state": state, "questions": task["questions"]}
    rotations = max(1, int(task.get("rotations", 1)))
    request, plan = to_request_with_plan(payload, request_id=str(task.get("id", "decide")))
    calibrator = _calibrator(adapter, task.get("calibration"))

    results = []
    tokens = 0
    for field in request.fields:
        header, choices, texts = compile_question(field, request.state, engine.prompt_layout)
        labels = engine.labels(len(choices), len(images))
        passes = []
        for offset in cyclic_offsets(len(choices), rotations):
            order = rotate(texts, offset)
            prompt = header + "\n".join(
                f"{label}: {text}" for label, text in zip(labels, order, strict=True)
            )
            with torch.no_grad():
                _, inputs, token_ids = engine.prepare(images, prompt, labels)
                tokens = max(tokens, int(inputs["input_ids"].shape[-1]))
                scored = engine.candidate_logits(inputs, token_ids).cpu().tolist()
                logits = [float(x) for x in scored]
                passes.append((offset, logits, token_ids))
        if len(passes) == 1:
            result = result_from_logits(choices, passes[0][1], token_ids=passes[0][2])
        else:
            result = combine_rotations(choices, [(offset, logits) for offset, logits, _ in passes])
        photo_only = bool(images) and not request.state
        result = calibrator.calibrate_result(
            result, field.type, len(result.scores) - 1, image=bool(images), photo_only=photo_only
        )
        results.append(result)
    body = to_response(request, results, model=model_name, plan=plan)
    return {
        "id": task.get("id", "unknown"),
        "answers": body["answers"],
        "tokens_in": tokens,
        "images": len(images),
        "rotations": rotations,
    }


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    # --model names the recipe for log lines; --weights/--base is what loads.
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    try:
        models = (_load(args.base, args.weights), args.weights, args.model)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(models, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
