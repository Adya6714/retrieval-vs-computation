"""Check whether Qwen2.5-7B is listed in TransformerLens model registry."""

from __future__ import annotations

TARGET_MODEL = "Qwen/Qwen2.5-7B"


def _get_official_model_names() -> list[str]:
    import transformer_lens

    # Preferred path mentioned in request.
    if hasattr(transformer_lens, "utilities"):
        utilities = transformer_lens.utilities
        fn = getattr(utilities, "official_model_name_list", None)
        if callable(fn):
            return list(fn())

    # Common fallback in some versions.
    loading_mod = getattr(transformer_lens, "loading_from_pretrained", None)
    if loading_mod is not None:
        fn = getattr(loading_mod, "official_model_name_list", None)
        if callable(fn):
            return list(fn())

    # Last-resort fallback for older/newer layouts.
    try:
        from transformer_lens.loading_from_pretrained import official_model_name_list

        return list(official_model_name_list())
    except Exception as exc:  # pragma: no cover - script fallback path
        raise RuntimeError(
            "Could not find official model registry API in transformer_lens."
        ) from exc


def main() -> None:
    names = _get_official_model_names()
    supported = TARGET_MODEL in names

    print(f"TransformerLens registry size: {len(names)} models")
    print(f"Target model: {TARGET_MODEL}")
    print(f"Supported: {'YES' if supported else 'NO'}")


if __name__ == "__main__":
    main()
