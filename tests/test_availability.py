# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Whether a recipe runs here, and the plan before a fetch (ADR-005).

Recipes over a temporary models directory, cards and the Hub faked: what is
under test is the judgement, not this machine.
"""

from types import SimpleNamespace

import pytest

from giq import availability, plan
from giq.registry import get_recipe, reload_registry

DEPTH = (
    "name: {name}\nmodalities: [depth]\nengine: transformers\n"
    "weights: {{path: {path}{source}}}\nvram: {{gb: {gb}}}\n"
)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """Four depth recipes: on disk, fetchable, manual, and too large."""
    models = tmp_path / "models"
    (models / "here").mkdir(parents=True)
    (models / "here" / "config.json").write_text("{}")
    operator = tmp_path / "recipes"
    operator.mkdir()
    rows = [
        ("d-ready", "here", ", source: 'hf:org/here'", 1.0),
        ("d-fetch", "gone", ", source: 'hf:org/gone', revision: " + "a" * 40, 1.0),
        ("d-manual", "lost", "", 1.0),
        ("d-huge", "gone", ", source: 'hf:org/gone'", 90.0),
    ]
    for name, path, source, gb in rows:
        (operator / f"{name}.yaml").write_text(
            DEPTH.format(name=name, path=path, source=source, gb=gb)
        )
    monkeypatch.setenv("GIQ_MODELS_DIR", str(models))
    monkeypatch.setenv("GIQ_DEPTH_MODELS_DIR", str(models))
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(operator))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    card = SimpleNamespace(uuid="GPU-a", index=0, name="card", vram_total_gb=32.0)
    monkeypatch.setattr("giq.gpus.get_gpus", lambda *a, **k: [card])
    monkeypatch.setattr("giq.gpus.compute_capabilities", lambda: {"gpu-a": "12.0"})
    reload_registry()
    yield models
    monkeypatch.undo()
    reload_registry()


def _judged(name: str) -> str:
    recipe = get_recipe(name)
    assert recipe is not None
    return availability.availability(recipe)[0]


def test_each_recipe_is_judged_against_this_machine(machine):
    assert _judged("d-ready") == "ready"
    assert _judged("d-fetch") == "fetchable"
    assert _judged("d-manual") == "manual"
    assert _judged("d-huge") == "unfit"
    _, found = availability.availability(get_recipe("d-huge"))
    card = next(c for c in found if c.check == "card")
    assert card.status == "fail" and "largest card has 32.0 GB" in card.message


def test_a_missing_engine_binary_makes_it_unfit(machine, monkeypatch, tmp_path):
    from giq.engines import reload_engines

    (machine / "m.gguf").write_bytes(b"x")
    recipe_file = tmp_path / "recipes" / "l.yaml"
    recipe_file.write_text(
        "name: l\nmodalities: [llm]\nengine: llama.cpp\nweights: {path: m.gguf}\nvram: {gb: 1}\n"
    )
    monkeypatch.setenv("GIQ_LLAMA_BINARY", str(tmp_path / "no-llama-server"))
    reload_engines()
    reload_registry()
    try:
        assert _judged("l") == "unfit"
        _, found = availability.availability(get_recipe("l"))
        assert "binary is not at" in next(c for c in found if c.check == "engine").message
    finally:
        monkeypatch.delenv("GIQ_LLAMA_BINARY")
        reload_engines()


def test_an_engine_check_can_rule_out_every_card(machine, monkeypatch):
    from dataclasses import replace

    from giq import plugins

    engine = plugins.engine("transformers")
    refusing = replace(engine, check=lambda recipe, cap: f"not on {cap}")
    monkeypatch.setitem(plugins._get().engines, "transformers", refusing)
    assert _judged("d-ready") == "unfit"


def test_the_vllm_floor():
    from giq.builtins import _vllm_check

    assert _vllm_check(None, "12.0") is None and _vllm_check(None, "7.0") is None
    assert "7.0 or newer" in str(_vllm_check(None, "6.1"))


def test_ready_recipes_come_kept_warm_first(machine, monkeypatch):
    from giq.policy import OFF, PINNED, reset_policy_store
    from giq.stats import get_stats

    (machine / "here2").mkdir()
    reload_registry()
    store = reset_policy_store()
    store.load()
    try:
        assert availability.ready("depth") == ["d-ready"]
        assert availability.default_recipe("depth") == "d-ready"
        store.set("d-ready", OFF)
        assert availability.default_recipe("depth") is None, "a switched-off recipe is no default"
        store.set("d-ready", PINNED)
        assert availability.ready("depth")[0] == "d-ready"
    finally:
        for name in list(get_stats().load_policies()):
            get_stats().delete_policy(name)
        reset_policy_store()


# --- the plan ----------------------------------------------------------------


class FakeHub:
    """The Hub's answers for one repository."""

    def __init__(self, files, gated=False, licence="apache-2.0"):
        self.files, self.gated, self.licence = files, gated, licence

    def model_info(self, repo, revision=None, files_metadata=False):
        siblings = [SimpleNamespace(rfilename=n, size=s) for n, s in self.files]
        card = SimpleNamespace(to_dict=lambda: {"license": self.licence})
        return SimpleNamespace(siblings=siblings, gated=self.gated, card_data=card)


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    def install(fake: FakeHub, token: str | None = "tok"):
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: token)
        monkeypatch.setattr(huggingface_hub, "auth_check", lambda repo, token=None: None)

    return install


def test_a_plan_sizes_the_fetch_from_the_hub(machine, hub):
    hub(FakeHub([("config.json", 10), ("model.safetensors", 4_000_000_000)]))
    p = plan.plan("d-fetch")
    assert p.availability == "fetchable" and p.can_fetch
    assert p.download_bytes == 4_000_000_010
    assert p.command == "giq add d-fetch"
    (move,) = p.transfers
    assert (move.repo, move.revision, move.file) == ("org/gone", "a" * 40, None)
    assert move.dest == machine / "gone"
    assert {c.check for c in p.checks} >= {"disk", "licence"}


def test_a_gated_repository_without_a_token_fails_the_plan(machine, hub):
    hub(FakeHub([("w.safetensors", 1)], gated="auto"), token=None)
    p = plan.plan("d-fetch")
    access = next(c for c in p.checks if c.check == "access")
    assert access.status == "fail" and "HF_TOKEN" in access.message
    assert not p.can_fetch


def test_a_custom_licence_is_shown_before_the_fetch(machine, hub):
    hub(FakeHub([("w.safetensors", 1)], licence="other"))
    licence = next(c for c in plan.plan("d-fetch").checks if c.check == "licence")
    assert licence.status == "warn" and "other" in licence.message


def test_without_the_hub_the_plan_still_stands(machine, monkeypatch):
    import huggingface_hub

    class Offline:
        def model_info(self, *a, **k):
            raise ConnectionError("no network")

    monkeypatch.setattr(huggingface_hub, "HfApi", Offline)
    p = plan.plan("d-fetch")
    access = next(c for c in p.checks if c.check == "access")
    assert access.status == "warn" and "unknown" in access.message
    assert p.can_fetch, "an unknown size does not stop a fetch"


def test_too_little_disk_fails_the_plan(machine, hub, monkeypatch):
    hub(FakeHub([("w.safetensors", 10**15)]))
    disk = next(c for c in plan.plan("d-fetch").checks if c.check == "disk")
    assert disk.status == "fail" and "free under" in disk.message


def test_an_unpinned_fetch_says_so(machine, hub):
    hub(FakeHub([("w.safetensors", 1)]))
    pinned = [c for c in plan.plan("d-huge").checks if c.check == "pinned"]
    assert pinned and pinned[0].status == "warn"


@pytest.mark.parametrize(
    ("file", "expected"),
    [
        (None, ["a.gguf", "b-00001-of-00002.gguf", "b-00002-of-00002.gguf", "c.json"]),
        ("a.gguf", ["a.gguf"]),
        ("b-00001-of-00002.gguf", ["b-00001-of-00002.gguf", "b-00002-of-00002.gguf"]),
    ],
)
def test_one_file_or_the_whole_repository(file, expected):
    repository = ["a.gguf", "b-00001-of-00002.gguf", "b-00002-of-00002.gguf", "c.json"]
    siblings = [(name, 1) for name in repository]
    assert sorted(n for n, _ in plan._wanted(siblings, file)) == expected
