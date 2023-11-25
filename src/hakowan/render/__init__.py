from .render import render
from ..common import logger

import mitsuba as mi

for variant in ["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"]:
    if variant in mi.variants():
        logger.info("Using variant: %s" % variant)
        mi.set_variant(variant)
        break
assert mi.variant() is not None
