import pytest
from hakowan import transform, scale
from hakowan.compiler.transform import principal_axes_affine_matrix
import copy
import numpy as np


class TestTransform:
    def test_filter(self):
        attr = scale.Attribute(name="index")
        t = transform.Filter(data=attr, condition=lambda x: True)
        assert t.data is attr
        assert t.condition(0)
        assert t._child is None

    def test_chaining_and_copy(self):
        attr0 = scale.Attribute(name="index")
        t0 = transform.Filter(data=attr0, condition=lambda x: True)
        attr1 = scale.Attribute(name="curvature")
        t1 = transform.Filter(data=attr1, condition=lambda x: True)
        t1 *= t0

        assert t1.data is attr1
        assert t1._child.data == attr0

        t2 = copy.deepcopy(t1)
        assert t2 is not t1
        assert t2.data is not t1.data
        assert t2._child is not t1._child
        assert t2._child.data is not t1._child.data

    def test_uv_mesh(self):
        t = transform.UVMesh(uv="@uv")
        assert t.uv == "@uv"
        assert t._child is None

    def test_affine(self):
        t = transform.Affine(matrix=np.eye(4))
        assert np.all(t.matrix == np.eye(4))
        assert t._child is None

    def test_principal_axes(self):
        t = transform.PrincipalAxes(frame=np.eye(3), orthonormalize_frame=False)
        assert np.allclose(t.frame, np.eye(3))
        assert t.orthonormalize_frame is False
        assert t._child is None

    def test_principal_axes_affine_matrix_maps_major_to_frame_column0(self):
        v = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
        frame = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        m = principal_axes_affine_matrix(v, frame, orthonormalize_frame=False)
        w = (m[:3, :3] @ v.T).T + m[:3, 3]
        assert w.std(0)[1] > max(w.std(0)[0], w.std(0)[2])
