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

        Cells are spaced by their per-cell **swept spheres** (true max-vertex
        distance from each cell's own rotation centre) projected onto the layout
        axis, so neighbours never overlap under any per-cell rotation in the
        interactive viewer — yet a wide-but-short row stacks by its height, not
        its width. Transforms are baked into each view's ``global_transform``;
        this runs *before* :meth:`compute_global_transform`, which fits the
        whole arrangement into the unit sphere.

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
    ) -> list[tuple[npt.NDArray, float]]:
        """Recursively lay out ``views`` (sharing a key prefix of length ``depth``).

        Centres the resulting group at the origin and returns the per-cell
        ``(centre, radius)`` spheres of every leaf rotation cell it contains, in
        the group's own centred coordinates.

        The viewer rotates each *leaf* cell about its own centre (never a whole
        sub-grid as one rigid unit), so spacing only needs the per-cell spheres
        to stay disjoint — not the group's overall bounding sphere. Packing
        therefore uses each group's reach *along the layout axis* (the span of
        its cells' swept spheres projected onto that axis). A wide-but-short row
        then stacks by its height rather than its width, avoiding a cavernous
        gap between rows while still guaranteeing no two rotating cells overlap.
        """
        terminal = [v for v in views if len(v._layout_cell) == depth]
        descending = [v for v in views if len(v._layout_cell) > depth]

        if not descending:
            # Leaf cell: all views share one coordinate space (overlaid) and
            # rotate together as a single cell.
            center, radius = self._cell_sphere(terminal)
            self._translate_views(terminal, -center)
            return [(np.zeros(3), radius)]

        # A layout node governs the descending views at this depth.
        options = node_options[descending[0]._layout_cell[depth][0]]

        # Build ordered child groups (each centred at the origin), tracking the
        # views and the per-cell spheres each contains.
        branches: dict[int, list[View]] = {}
        for v in descending:
            branches.setdefault(v._layout_cell[depth][1], []).append(v)
        child_groups: list[tuple[list[View], list[tuple[npt.NDArray, float]]]] = []
        for branch in sorted(branches):
            cv = branches[branch]
            child_groups.append((cv, self._layout_group(cv, depth + 1, node_options)))
        # Loose overlay views terminating at this level (`+` straddling a layout
        # operator) become one extra leaf cell.
        if terminal:
            center, radius = self._cell_sphere(terminal)
            self._translate_views(terminal, -center)
            child_groups.append((terminal, [(np.zeros(3), radius)]))

        # Optionally scale each child group to equal size (its bounding-sphere
        # radius about the centre) before packing.
        if options.normalize:
            normalized = []
            for cv, cells in child_groups:
                r = max(
                    (float(np.linalg.norm(c)) + rad for c, rad in cells), default=0.0
                )
                if r > 0:
                    self._scale_views(cv, 1.0 / r)
                    cells = [(c / r, rad / r) for c, rad in cells]
                normalized.append((cv, cells))
            child_groups = normalized

        # Pack child groups along the node's axis using each group's axis reach
        # (the span its cells' spheres cover along ``axis``). Consecutive groups
        # are separated by reach_i + gap + reach_j, so no two rotating cells
        # overlap, yet the packing is as tight as that guarantee allows.
        axis = options.axis
        reaches = []  # (lo, hi) extent along ``axis`` for each child group
        for _, cells in child_groups:
            lo = min((c[axis] - rad for c, rad in cells), default=0.0)
            hi = max((c[axis] + rad for c, rad in cells), default=0.0)
            reaches.append((lo, hi))
        mean_diameter = (
            float(np.mean([hi - lo for lo, hi in reaches])) if reaches else 0.0
        )
        gap_distance = options.gap * mean_diameter if mean_diameter > 0 else options.gap

        cursor = 0.0
        all_views: list[View] = []
        all_cells: list[tuple[npt.NDArray, float]] = []
        for (cv, cells), (lo, hi) in zip(child_groups, reaches):
            offset = np.zeros(3)
            offset[axis] = cursor - lo
            self._translate_views(cv, offset)
            all_views.extend(cv)
            all_cells.extend((c + offset, rad) for c, rad in cells)
            cursor += (hi - lo) + gap_distance
        total = cursor - gap_distance if child_groups else 0.0

        # Recentre the packed group at the origin along the layout axis.
        shift = np.zeros(3)
        shift[axis] = -total / 2.0
        self._translate_views(all_views, shift)
        return [(c + shift, rad) for c, rad in all_cells]

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
                assert bbox_max is not None
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
            # Copy: ``bbox`` is a stacked array, so ``bbox[0]`` / ``bbox[1]`` are
            # row views into the view's stored ``_bbox``. The accumulation below
            # writes back via ``out=``, which would otherwise clobber this view's
            # bbox with the whole-scene union.
            bbox_min = view.bbox[0].copy()
            bbox_max = view.bbox[1].copy()
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
