#!/usr/bin/env python3
"""Remove an occlusion prefix from metadata image names and paths."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip an image filename prefix from metadata JSON.")
    parser.add_argument("--input", required=True, help="Input metadata JSON file.")
    parser.add_argument("--output", required=True, help="Output metadata JSON file.")
    parser.add_argument("--prefix", default="fire_L_", help="Filename prefix to remove.")
    return parser.parse_args()


def strip_prefix(data: list[dict], prefix: str) -> list[dict]:
    path_token = f"/{prefix}"
    for item in data:
        if item.get("image_filename", "").startswith(prefix):
            item["image_filename"] = item["image_filename"].replace(prefix, "", 1)

        if path_token in item.get("image_path", ""):
            item["image_path"] = item["image_path"].replace(path_token, "/", 1)
    return data


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = strip_prefix(data, args.prefix)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Saved cleaned metadata to {args.output}")


if __name__ == "__main__":
    main()
