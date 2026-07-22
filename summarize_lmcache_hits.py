#!/usr/bin/env python3

import re
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python summarize_lmcache_hits.py JOB_LOG.out"
        )

    log_path = sys.argv[1]

    phase = None
    results = {
        "cold": {},
        "warm": {},
    }

    pattern = re.compile(
        r"Reqid:\s+(\S+).*LMCache hit tokens:\s+(\d+)"
    )

    with open(
        log_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as log_file:
        for line in log_file:
            if "Cold run starting" in line:
                phase = "cold"
                continue

            if "Warm run starting" in line:
                phase = "warm"
                continue

            if phase not in results:
                continue

            match = pattern.search(line)
            if match is None:
                continue

            request_id = match.group(1)
            hit_tokens = int(match.group(2))

            # Tensor-parallel workers may print duplicates.
            # Keep the largest value seen for each request.
            previous = results[phase].get(request_id, 0)
            results[phase][request_id] = max(
                previous,
                hit_tokens,
            )

    for phase_name in ("cold", "warm"):
        values = list(results[phase_name].values())

        print()
        print("=" * 45)
        print(f"{phase_name.upper()} LMCache SUMMARY")
        print("=" * 45)

        if not values:
            print("No LMCache lookup rows found.")
            continue

        zero_hits = sum(value == 0 for value in values)
        nonzero_hits = sum(value > 0 for value in values)
        average_hits = sum(values) / len(values)
        maximum_hits = max(values)

        print(f"Requests found:    {len(values)}")
        print(f"Requests with hit: {nonzero_hits}")
        print(f"Zero-hit requests: {zero_hits}")
        print(f"Average hit tokens:{average_hits:.1f}")
        print(f"Maximum hit tokens:{maximum_hits}")


if __name__ == "__main__":
    main()