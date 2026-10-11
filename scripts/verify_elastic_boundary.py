#!/usr/bin/env python3
"""Fail closed when the versioned SLHAv2/Elastic ownership boundary drifts."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs/integration/slhav2-elastic-boundary-v1.json"
EXTERNAL_PACKAGE_RE = re.compile(
    r"\bmemorithm-elastic v([^\s]+) \((git\+[^)]+)\)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"SLHAv2 Elastic boundary mismatch: {message}")


def load_toml(path: str) -> dict:
    with (ROOT / path).open("rb") as handle:
        return tomllib.load(handle)


def verify_manifests_and_lock(matrix: dict) -> None:
    require(matrix.get("schema") == "slhav2-elastic-boundary/v1", "schema")
    require(matrix.get("msrv") == "1.89.0", "declared MSRV")

    consumer = matrix["consumer"]
    manifest = load_toml(consumer["manifest"])
    require(
        manifest["package"]["rust-version"] == matrix["msrv"].removesuffix(".0"),
        "consumer MSRV",
    )
    dependencies = manifest["dependencies"]
    features = manifest["features"]
    boundaries = {item["id"]: item for item in matrix["boundaries"]}

    local = boundaries["local-incubator"]
    local_dep = dependencies.get(local["dependency_key"])
    require(isinstance(local_dep, dict), "local elastic-core dependency shape")
    require(local_dep.get("version") == local["version"], "local elastic-core version")
    require(local_dep.get("path") == local["path"], "local elastic-core path")
    require(
        local_dep.get("package", local["dependency_key"]) == local["package"],
        "local elastic-core package",
    )

    external = boundaries["external-elasticxxx"]
    external_dep = dependencies.get(external["dependency_key"])
    require(isinstance(external_dep, dict), "external ElasticXxx dependency shape")
    require(external_dep.get("package") == external["package"], "external package")
    require(external_dep.get("version") == external["version"], "external version")
    require(external_dep.get("git") == external["repository"], "external repository")
    require(external_dep.get("rev") == external["revision"], "external revision")
    require(external_dep.get("optional") is external["optional"], "external optionality")
    require(
        features.get(external["feature"]) == [f"dep:{external['dependency_key']}"],
        "external feature binding",
    )

    root_manifest = load_toml("Cargo.toml")
    require(
        local["workspace_excluded_from_root"] in root_manifest["workspace"]["exclude"],
        "nested Elastic workspace exclusion",
    )
    nested = load_toml(local["workspace_manifest"])
    require(
        nested["workspace"]["package"]["rust-version"]
        == matrix["msrv"].removesuffix(".0"),
        "nested Elastic workspace MSRV",
    )
    require(
        local["package"] in nested["workspace"]["members"],
        "local elastic-core absent from nested workspace",
    )

    lock = load_toml("Cargo.lock")
    packages = lock["package"]
    local_locked = [
        package for package in packages if package["name"] == local["package"]
    ]
    require(len(local_locked) == 1, "local elastic-core lock identity")
    require(local_locked[0]["version"] == local["version"], "local lock version")
    require("source" not in local_locked[0], "local elastic-core unexpectedly external")

    external_locked = [
        package for package in packages if package["name"] == external["package"]
    ]
    require(len(external_locked) == 1, "external ElasticXxx lock identity")
    source = external_locked[0].get("source", "")
    require(external_locked[0]["version"] == external["version"], "external lock version")
    require(f"rev={external['revision']}" in source, "external lock requested revision")
    require(source.endswith(f"#{external['revision']}"), "external lock resolved revision")


def verify_graphs(
    matrix: dict,
    default_tree: Path | None,
    feature_tree: Path | None,
    nested_tree: Path | None,
) -> None:
    boundaries = {item["id"]: item for item in matrix["boundaries"]}
    local = boundaries["local-incubator"]
    external = boundaries["external-elasticxxx"]
    local_line = f"{local['package']} v{local['version']}"

    if default_tree is not None:
        graph = default_tree.read_text(encoding="utf-8")
        require(local_line in graph, "default graph missing local elastic-core")
        require(
            not EXTERNAL_PACKAGE_RE.search(graph),
            "default graph unexpectedly activates external ElasticXxx",
        )

    if feature_tree is not None:
        graph = feature_tree.read_text(encoding="utf-8")
        require(local_line in graph, "feature graph missing local elastic-core")
        observed = EXTERNAL_PACKAGE_RE.findall(graph)
        require(observed, "feature graph missing external ElasticXxx")
        unexpected = sorted(
            (version, source)
            for version, source in observed
            if version != external["version"]
            or f"rev={external['revision']}" not in source
            or not source.endswith(f"#{external['revision']}")
        )
        require(
            not unexpected,
            f"feature graph contains unqualified external ElasticXxx {unexpected!r}",
        )

    if nested_tree is not None:
        graph = nested_tree.read_text(encoding="utf-8")
        require(local_line in graph, "nested graph missing local elastic-core")
        require("slhav2-vram v" not in graph, "nested graph imports SLHAv2 product")
        require(
            not EXTERNAL_PACKAGE_RE.search(graph),
            "nested graph imports external ElasticXxx runtime",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-tree", type=Path)
    parser.add_argument("--feature-tree", type=Path)
    parser.add_argument("--nested-tree", type=Path)
    args = parser.parse_args()

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    verify_manifests_and_lock(matrix)
    verify_graphs(matrix, args.default_tree, args.feature_tree, args.nested_tree)
    print("SLHAv2 Elastic boundary matches manifests, lockfile and supplied graphs")


if __name__ == "__main__":
    main()
