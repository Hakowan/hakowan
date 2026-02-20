from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

from .view import View


@dataclass
class Scene:
    views: list[View] = field(default_factory=list)

    def __len__(self):
        return self.views.__len__()

    def __getitem__(self, key):
        return self.views.__getitem__(key)

    def __setitem__(self, key, value: View):
        return self.views.__setitem__(key, value)

    def __delitem__(self, key):
        return self.views.__delitem__(key)

    def __iter__(self):
        return self.views.__iter__()

    def __contains__(self, item):
        return self.views.__contains__(item)

    def append(self, view: View) -> "Scene":
        self.views.append(view)
        return self

    def compute_global_transform(self):
        """Compute the global transformation matrix to fit all views in a unit sphere.

        The global transformation matrix is stored in each view.
        """
        if len(self.views) == 0:
            return

        bbox_min = None
        bbox_max = None
        for view in self.views:
            if view.bbox is None:
                continue
            bbox_min = view.bbox[0]
            bbox_max = view.bbox[1]
            break

        if bbox_min is None or bbox_max is None:
            # Data in all views are empty.
            return

        for view in self.views:
            if view.bbox is None:
                continue
            np.minimum(bbox_min, view.bbox[0], out=bbox_min)
            np.maximum(bbox_max, view.bbox[1], out=bbox_max)

        bbox_center = (bbox_min + bbox_max) / 2
        translation = np.eye(4)
        translation[0:3, 3] = -bbox_center

        # max_side = np.amax(bbox_max - bbox_min)
        diag = norm(bbox_max - bbox_min)

        # factor = max_side / diag
        scale = np.eye(4)
        scale[0:3, 0:3] *= 2 / diag

        global_transform = scale @ translation

        for view in self.views:
            view.global_transform = global_transform @ view.global_transform
