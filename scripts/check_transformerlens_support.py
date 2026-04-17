"""Check whether any Qwen2.5 model is in TransformerLens registry."""

from __future__ import annotations

TARGET_SUBSTRING = "Qwen2.5"


def _get_official_model_names() -> list[str]:
    import transformer_lens.loading_from_pretrained as loading_from_pretrained

    names = loading_from_pretrained.OFFICIAL_MODEL_NAMES
    return list(names)


def main() -> None:
    names = _get_official_model_names()
    matches = [name for name in names if TARGET_SUBSTRING in name]
    supported = len(matches) > 0

    print(f"TransformerLens registry size: {len(names)} models")
    print(f"Target substring: {TARGET_SUBSTRING}")
    print(f"Supported: {'YES' if supported else 'NO'}")
    if supported:
        print("Matching entries:")
        for name in matches:
            print(f"- {name}")


if __name__ == "__main__":
    main()
