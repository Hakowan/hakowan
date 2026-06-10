from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

from .view import View
from ..grammar.layer import LayoutOptions


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

    def apply_layout(self, node_options: dict[int, LayoutOptions]):
        """Lay out juxtaposition cells, honouring nested ``|`` / ``&`` as a grid.

        Views carry a ``_layout_cell`` key — a path of ``(layout_node_id, branch)``
        pairs recorded during layer-tree flattening. ``node_options`` maps each
        layout node id to its :class:`LayoutOptions`. The layout is applied
        **recursively**: every layout node positions its direct children along
        *its own* axis, and a child may itself be a sub-layout (packed first,
        bottom-up). Thus ``(a | b) & c`` places ``a``/``b`` in a row and stacks
        that row with ``c``; ``(a | b) & (c | d)`` yields a 2x2 grid.

        Cells are spaced by their **bounding-sphere diameter** (true max-vertex
        distance from the rotation centre), so neighbours never overlap under any
        per-cell rotation in the interactive viewer. Transforms are baked into
        each view's ``global_transform``; this runs *before*
        :meth:`compute_global_transform`, which fits the whole arrangement into
        the unit sphere.

        Args:
            node_options: Map from layout node id to its layout parameters.
        """
        if len(self.views) == 0:
            return
        if not any(len(v._layout_cell) > 0 for v in self.views):
            # No juxtaposition node in the tree (pure overlay / single layer).
            return
        self._layout_group(list(self.views), 0, node_options)

    def _layout_group(
        self, views: list[View], depth: int, node_options: dict[int, LayoutOptions]
    ) -> float:
        """Recursively lay out ``views`` (sharing a key prefix of length ``depth``).

        Centres the resulting group at the origin and returns its bounding-sphere
        radius, so the caller can pack it as one unit.
        """
        terminal = [v for v in views if len(v._layout_cell) == depth]
        descending = [v for v in views if len(v._layout_cell) > depth]

        if not descending:
            # Leaf cell: all views share one coordinate space (overlaid).
            center, radius = self._cell_sphere(terminal)
            self._translate_views(terminal, -center)
            return radius

        # A layout node governs the descending views at this depth.
        options = node_options[descending[0]._layout_cell[depth][0]]

        # Build ordered child groups (each centred at the origin, with a radius).
        branches: dict[int, list[View]] = {}
        for v in descending:
            branches.setdefault(v._layout_cell[depth][1], []).append(v)
        child_groups: list[tuple[list[View], float]] = []
        for branch in sorted(branches):
            cv = branches[branch]
            child_groups.append((cv, self._layout_group(cv, depth + 1, node_options)))
        # Loose overlay views terminating at this level (`+` straddling a layout
        # operator) become one extra cell.
        if terminal:
            center, radius = self._cell_sphere(terminal)
            self._translate_views(terminal, -center)
            child_groups.append((terminal, radius))

        # Optionally scale each child group to equal size before packing.
        if options.normalize:
            normalized = []
            for cv, r in child_groups:
                if r > 0:
                    self._scale_views(cv, 1.0 / r)
                    normalized.append((cv, 1.0))
                else:
                    normalized.append((cv, 0.0))
            child_groups = normalized

        # Pack child groups along the node's axis. Consecutive centres are
        # separated by r_i + r_j + gap, so bounding spheres stay disjoint.
        axis = options.axis
        mean_diameter = (
            float(np.mean([2.0 * r for _, r in child_groups])) if child_groups else 0.0
        )
        gap_distance = options.gap * mean_diameter if mean_diameter > 0 else options.gap
        cursor = 0.0
        for cv, r in child_groups:
            offset = np.zeros(3)
            offset[axis] = cursor + r
            self._translate_views(cv, offset)
            cursor += 2.0 * r + gap_distance
        total = cursor - gap_distance if child_groups else 0.0

        # Recentre the packed group at the origin along the layout axis.
        shift = np.zeros(3)
        shift[axis] = -total / 2.0
        all_views = [v for cv, _ in child_groups for v in cv]
        self._translate_views(all_views, shift)

        _, radius = self._cell_sphere(all_views)
        return radius

    @staticmethod
    def _cell_sphere(views: list[View]) -> tuple[npt.NDArray, float]:
        """Bounding-box centre and true geometry radius of a group of views."""
        bbox_min = None
        bbox_max = None
        for v in views:
            if v.bbox is None:
                continue
            if bbox_min is None:
                bbox_min = v.bbox[0].astype(np.float64).copy()
                bbox_max = v.bbox[1].astype(np.float64).copy()
            else:
                np.minimum(bbox_min, v.bbox[0], out=bbox_min)
                np.maximum(bbox_max, v.bbox[1], out=bbox_max)
        if bbox_min is None:
            return np.zeros(3), 0.0
        center = (bbox_min + bbox_max) / 2
        radius = max(
            (v.max_vertex_distance(center) for v in views if v.bbox is not None),
            default=0.0,
        )
        return center, radius

    @staticmethod
    def _translate_views(views: list[View], offset: npt.NDArray) -> None:
        if not np.any(offset):
            return
        transform = np.eye(4)
        transform[0:3, 3] = offset
        for v in views:
            v.global_transform = transform @ v.global_transform
            v.initialize_bbox()

    @staticmethod
    def _scale_views(views: list[View], factor: float) -> None:
        if factor == 1.0:
            return
        transform = np.eye(4)
        transform[0:3, 0:3] *= factor
        for v in views:
            v.global_transform = transform @ v.global_transform
            v.initialize_bbox()

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
