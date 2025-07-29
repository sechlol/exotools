import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from astropy.table import QTable
from tabulate import tabulate

from exotools.io import Hdf5Storage, EcsvStorage, FeatherStorage, MemoryStorage
from exotools.utils.qtable_utils import QTableHeader
from tests.conftest import load_all_test_qtables_and_headers

NUM_ITERATIONS = 5
TARGET_ROWS = 10_000


def bootstrap_qtable(qtable: QTable, rows: int) -> QTable:
    if len(qtable) >= rows:
        return qtable
    copies = rows // len(qtable)
    remainder = rows % len(qtable)
    parts = [qtable] * copies
    if remainder:
        indices = np.random.choice(len(qtable), remainder, replace=True)
        parts.append(qtable[indices])
    return QTable(np.concatenate(parts))


def generate_sample_json(n: int) -> dict:
    base_fields = [
        ("id", 123),
        ("name", "example"),
        ("values", [1, 2, 3]),
        ("meta", {"a": 1, "b": 2}),
        ("flag", True),
        ("timestamp", "2025-07-28T19:17:20+03:00"),
    ]
    result = {f"field_{i}": base_fields[i % len(base_fields)][1] for i in range(n)}
    return result


def load_and_prepare_data(target_rows: int) -> dict[str, tuple[QTable, QTableHeader]]:
    raw_data = load_all_test_qtables_and_headers()
    augmented_data = {}
    for name, (qtable, header) in raw_data.items():
        print(f"Augmenting {name}")
        augmented_data[name] = (bootstrap_qtable(qtable, target_rows), header)
    return augmented_data


def run_benchmark(
    storage_class,
    test_data: dict[str, tuple[QTable, QTableHeader]],
    json_data: dict[str, Any],
) -> dict[str, float]:
    timings = {
        "write_qtable": 0.0,
        "read_qtable": 0.0,
        "write_json": 0.0,
        "read_json": 0.0,
    }

    for _ in range(NUM_ITERATIONS):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "storage"
            if storage_class == Hdf5Storage:
                storage = storage_class(path.with_suffix(".h5"))
            else:
                path.mkdir(exist_ok=True)
                storage = storage_class(path)

            t0 = perf_counter()
            for name, (qtable, header) in test_data.items():
                storage.write_qtable(qtable, header, name, override=True)
            timings["write_qtable"] += (perf_counter() - t0) / len(test_data)

            t0 = perf_counter()
            for name in test_data:
                storage.read_qtable(name)
            timings["read_qtable"] += (perf_counter() - t0) / len(test_data)

            t0 = perf_counter()
            storage.write_json(json_data, "meta")
            timings["write_json"] += perf_counter() - t0

            t0 = perf_counter()
            storage.read_json("meta")
            timings["read_json"] += perf_counter() - t0

    for k in timings:
        if not np.isnan(timings[k]):
            timings[k] /= NUM_ITERATIONS

    return timings


def collect_benchmark_results(
    storages: dict[str, type],
    data: dict[str, tuple[QTable, QTableHeader]],
    json_data: dict[str, Any],
) -> dict[str, dict[str, float]]:
    results = {}
    for name, cls in storages.items():
        print(f"Running benchmark for {name}")
        results[name] = run_benchmark(cls, data, json_data)

    return results


def display_results(all_results: dict[str, dict[str, float]]) -> None:
    operations = list(next(iter(all_results.values())).keys())
    headers = ["Operation"] + list(all_results.keys())
    table = []
    for op in operations:
        row = [op]
        for storage in all_results:
            val = all_results[storage][op]
            row.append(f"{val:.6f}" if not np.isnan(val) else "")
        table.append(row)
    print("\nResults (seconds per operation):")
    print(tabulate(table, headers=headers, tablefmt="github"))


def display_relative_results(all_results: dict[str, dict[str, float]]) -> None:
    operations = list(next(iter(all_results.values())).keys())
    headers = ["Operation"] + list(all_results.keys())
    table = []
    for op in operations:
        min_val = min((v[op] for v in all_results.values() if not np.isnan(v[op])), default=float("inf"))
        row = [op]
        for storage in all_results:
            val = all_results[storage][op]
            rel = val / min_val if not np.isnan(val) and min_val != 0 else ""
            row.append(f"{rel:.2f}" if rel != "" else "")
        table.append(row)
    print("\nRelative Performance (lower is better, 1.0 = fastest):")
    print(tabulate(table, headers=headers, tablefmt="github"))


def main():
    qtable_data = load_and_prepare_data(TARGET_ROWS)
    print("Preparing json data...")
    json_data = generate_sample_json(TARGET_ROWS)

    storages = {
        "Hdf5Storage": Hdf5Storage,
        "EcsvStorage": EcsvStorage,
        "FeatherStorage": FeatherStorage,
        "MemoryStorage": MemoryStorage,
    }

    results = collect_benchmark_results(storages, qtable_data, json_data)
    display_results(results)
    display_relative_results(results)


if __name__ == "__main__":
    main()
