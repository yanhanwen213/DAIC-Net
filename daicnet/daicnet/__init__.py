"""DAIC-Net model package."""

__all__ = [
    "DAICNet",
    "DAICLoss",
    "IlluminationInvariantHead",
    "FingerVeinROIExtractor",
]


def __getattr__(name: str):
    if name == "DAICNet":
        from .daic_model import DAICNet

        return DAICNet
    if name == "DAICLoss":
        from .daic_losses import DAICLoss

        return DAICLoss
    if name == "IlluminationInvariantHead":
        from .illumination_head import IlluminationInvariantHead

        return IlluminationInvariantHead
    if name == "FingerVeinROIExtractor":
        from .roi import FingerVeinROIExtractor

        return FingerVeinROIExtractor
    raise AttributeError(name)
