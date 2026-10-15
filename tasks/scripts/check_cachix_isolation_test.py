# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the checked-in credential boundary; GitHub environment rules are external."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITER = REPO_ROOT / ".github/workflows/cachix.yml"


def load_yaml(path: Path) -> dict:
    # BaseLoader preserves the Actions `on` key instead of treating it as a
    # YAML 1.1 boolean. Scalars remain strings, including true/false.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_cachix_credentials_are_not_forwarded_to_other_workflows_or_actions():
    for path in (REPO_ROOT / ".github").rglob("*"):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        content = path.read_text().upper()
        assert "CACHIX_AUTH_TOKEN" not in content, path
        if path != WRITER:
            assert "CACHIX_WRITE_TOKEN" not in content, path
            assert "CACHIX-AUTH-TOKEN" not in content, path


def test_shared_nix_setup_only_reads_the_cache():
    setup = load_yaml(REPO_ROOT / ".github/actions/setup-nix/action.yml")
    cache_steps = [
        step
        for step in setup["runs"]["steps"]
        if step.get("uses", "").startswith("cachix/cachix-action@")
    ]
    assert len(cache_steps) == 1
    assert cache_steps[0]["with"] == {"name": "openshell", "skipPush": "true"}
    assert "cachix-auth-token" not in setup.get("inputs", {})


def check_cache_writer(writer: dict) -> None:
    assert writer["on"] == {"push": {"branches": ["main"]}}
    assert writer["permissions"] == {"contents": "read"}
    assert set(writer["jobs"]) == {"populate"}
    job = writer["jobs"]["populate"]
    assert job["environment"] == "cachix-write"
    assert job["if"] == (
        "github.repository == 'NVIDIA/OpenShell' && "
        "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )
    assert job["runs-on"] == "${{ matrix.runner }}"
    assert job["strategy"]["matrix"]["include"] == [
        {"system": "x86_64-linux", "runner": "ubuntu-24.04"},
        {"system": "aarch64-linux", "runner": "ubuntu-24.04-arm"},
        {"system": "aarch64-darwin", "runner": "macos-15-xlarge"},
    ]
    # Review any new step in this credential-bearing job. Preserve order and
    # duplicates: a later main checkout must not hide an earlier candidate build.
    steps = job["steps"]
    assert [step.get("uses", "run").split("@")[0] for step in steps] == [
        "run",
        "actions/checkout",
        "DeterminateSystems/determinate-nix-action",
        "cachix/cachix-action",
        "run",
    ]
    credential, checkout, install_nix, cache, build = steps
    assert "run" in credential and "run" in build
    for step in (checkout, install_nix, cache):
        assert re.fullmatch(r"[0-9a-f]{40}", step["uses"].partition("@")[2])
    assert checkout["with"] == {
        "persist-credentials": "false",
        "ref": "${{ github.sha }}",
    }
    assert cache["with"] == {
        "name": "openshell",
        "authToken": "${{ secrets.CACHIX_WRITE_TOKEN }}",
    }


def test_cache_writer_uses_only_the_main_push_revision():
    check_cache_writer(load_yaml(WRITER))


def test_cache_writer_rejects_candidate_checkout_followed_by_main_restore():
    writer = load_yaml(WRITER)
    steps = writer["jobs"]["populate"]["steps"]
    checkout = steps[1]
    candidate_checkout = {
        **checkout,
        "with": {**checkout["with"], "ref": "refs/pull/123/head"},
    }
    steps.insert(-1, candidate_checkout)
    steps.append(checkout)

    with pytest.raises(AssertionError):
        check_cache_writer(writer)


@pytest.mark.parametrize(
    ("source", "destination"),
    [(1, 4), (4, 1)],
    ids=["checkout-after-build", "build-before-checkout"],
)
def test_cache_writer_rejects_reordered_steps(source: int, destination: int):
    writer = load_yaml(WRITER)
    steps = writer["jobs"]["populate"]["steps"]
    steps.insert(destination, steps.pop(source))

    with pytest.raises(AssertionError):
        check_cache_writer(writer)


def test_cache_writer_rejects_mutable_action_refs():
    writer = load_yaml(WRITER)
    writer["jobs"]["populate"]["steps"][1]["uses"] = "actions/checkout@main"

    with pytest.raises(AssertionError):
        check_cache_writer(writer)


def test_cache_writer_requires_its_environment_credential():
    step = load_yaml(WRITER)["jobs"]["populate"]["steps"][0]
    assert step["env"] == {"CACHIX_WRITE_TOKEN": "${{ secrets.CACHIX_WRITE_TOKEN }}"}
    for value, expected in (("", 1), ("local-test-sentinel", 0)):
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eu", "-c", step["run"]],
            env={"PATH": os.defpath, "CACHIX_WRITE_TOKEN": value},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        assert "local-test-sentinel" not in result.stdout + result.stderr
