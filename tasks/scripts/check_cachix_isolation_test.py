# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the checked-in credential boundary; GitHub environment rules are external."""

import os
import subprocess
from pathlib import Path

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


def test_cache_writer_uses_only_the_main_push_revision():
    writer = load_yaml(WRITER)
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
    # No PR artifacts, local actions or caller-supplied source enter this job.
    actions = {
        step["uses"].split("@")[0]: step for step in job["steps"] if "uses" in step
    }
    assert set(actions) == {
        "actions/checkout",
        "DeterminateSystems/determinate-nix-action",
        "cachix/cachix-action",
    }
    assert actions["actions/checkout"]["with"] == {
        "persist-credentials": "false",
        "ref": "${{ github.sha }}",
    }
    assert actions["cachix/cachix-action"]["with"] == {
        "name": "openshell",
        "authToken": "${{ secrets.CACHIX_WRITE_TOKEN }}",
    }


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
