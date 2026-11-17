"""
Dataset and data pipeline modules for XD-Violence CLIP features.
"""

def __getattr__(name: str):
    if name in ("XDViolenceClipDataset", "create_dataloaders"):
        from . import dataset
        return getattr(dataset, name)
    if name in ("extract_label_from_filename", "get_base_video_name", "generate_vadclip_csv", "scan_features_directory", "validate_features"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "XDViolenceClipDataset",
    "create_dataloaders",
    "extract_label_from_filename",
    "get_base_video_name",
    "generate_vadclip_csv",
    "scan_features_directory",
    "validate_features"
]
