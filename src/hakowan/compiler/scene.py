from dataclasses import dataclass, field
import numpy as np
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

    def apply_layout(self, axis: int = 0, gap: float = 0.1, normalize: bool = False):
        """Lay out juxtaposition cells side by side along ``axis``.

        Views are grouped by their ``_layout_cell`` key (set during layer-tree
        flattening). Each cell is translated apart along ``axis`` so the cells do
        not overlap; optionally each cell is scaled to unit size first. The
        resulting transform is baked into each view's ``global_transform`` and
        the view bounding boxes are refreshed. This runs *before*
        :meth:`compute_global_transform`, which then fits the whole arrangement
        into the unit sphere.

        Cells are spaced by their **bounding-sphere diameter** (the bbox
        diagonal), not their extent along ``axis``. Because the interactive
        viewer rotates each cell about its own centre, a cell sweeps its bounding
        sphere; spacing by the diameter keeps those spheres disjoint so the cells
        never overlap under *any* rotation.

        Args:
            axis: Layout axis (0, 1, or 2).
            gap: Spacing between cells, as a fraction of the mean cell diameter.
            normalize: Whether to scale each cell to unit size before placing.
        """
        if len(self.views) == 0:
            return

        # Group views by cell key, preserving first-seen (left-to-right) order.
        cells: dict[tuple, list[View]] = {}
        for view in self.views:
            cells.setdefault(view._layout_cell, []).append(view)
        if len(cells) <= 1:
            # Nothing to juxtapose (e.g. a pure overlay or a single layer).
            return

        # Compute per-cell bounding box, scale factor, and bounding-sphere
        # diameter (the bbox diagonal). The cell rotates about its bbox centre,
        # so its geometry stays within the sphere of that diameter under any
        # rotation; spacing by the diameter keeps neighbouring cells disjoint.
        cell_specs = []
        for views in cells.values():
            bbox_min = None
            bbox_max = None
            for view in views:
                if view.bbox is None:
                    continue
                if bbox_min is None:
                    bbox_min = view.bbox[0].astype(np.float64).copy()
                    bbox_max = view.bbox[1].astype(np.float64).copy()
                else:
                    np.minimum(bbox_min, view.bbox[0], out=bbox_min)
                    np.maximum(bbox_max, view.bbox[1], out=bbox_max)

            if bbox_min is None:
                # Cell has no geometry; treat it as a zero-size point at origin.
                center = np.zeros(3)
                scale = 1.0
                diameter = 0.0
            else:
                center = (bbox_min + bbox_max) / 2
                diag = norm(bbox_max - bbox_min)
                scale = (1.0 / diag) if (normalize and diag > 0) else 1.0
                diameter = diag * scale
            cell_specs.append((views, center, scale, diameter))

        mean_diameter = np.mean([diameter for *_, diameter in cell_specs])
        gap_distance = gap * mean_diameter if mean_diameter > 0 else gap

        # Place cells one after another along ``axis``, centred on the off-axes.
        # Each centre sits half a diameter past the cursor, so consecutive
        # centres are separated by ``r_i + r_j + gap`` (sum of bounding-sphere
        # radii plus the gap) — i.e. the spheres are disjoint with a margin.
        cursor = 0.0
        for views, center, scale, diameter in cell_specs:
            target = np.zeros(3)
            target[axis] = cursor + diameter / 2

            # M = T(target) @ S(scale, about origin) @ T(-center)
            recenter = np.eye(4)
            recenter[0:3, 3] = -center
            scaling = np.eye(4)
            scaling[0:3, 0:3] *= scale
            placement = np.eye(4)
            placement[0:3, 3] = target
            cell_transform = placement @ scaling @ recenter

            for view in views:
                view.global_transform = cell_transform @ view.global_transform
                view.initialize_bbox()

            cursor += diameter + gap_distance

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
