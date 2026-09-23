import json
from pathlib import Path

import hakowan as hkw
import lagrange
import numpy as np
import pytest
from pydantic import ValidationError
from hakowan.spec import ExpressionError, ExpressionSpec, FigureSpec, compile_expression


def _mesh_with_fields():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    mesh.add_triangle(0, 1, 2)
    mesh.create_attribute(
        "value",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([0.25, 0.5, 0.75]),
    )
    mesh.create_attribute(
        "vector",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Vector,
        initial_values=np.eye(3),
    )
    return mesh


def test_categorical_spec_uses_qualitative_colormap_when_omitted():
    spec = FigureSpec.model_validate(
        {
            "root": {
                "kind": "layer",
                "spec": {
                    "channels": {
                        "material": {
                            "kind": "diffuse",
                            "reflectance": {
                                "kind": "scalar_field",
                                "data": {"name": "region"},
                                "categories": True,
                            },
                        }
                    }
                },
            }
        }
    ).to_dict()

    field = spec["root"]["spec"]["channels"]["material"]["reflectance"]
    assert field["colormap"] == "set1"


def test_schema_is_versioned_json_schema():
    schema = hkw.schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://hakowan.github.io/hakowan/schema/v1.json"
    assert schema["properties"]["version"]["default"] == "1.1"
    assert set(schema["properties"]["version"]["enum"]) == {"1.0", "1.1"}
    assert schema["additionalProperties"] is False
    assert len(schema["$defs"]) > 40
    json.dumps(schema)
    assert schema["description"]
    assert schema["examples"]
    for property_schema in schema["properties"].values():
        assert property_schema["description"]
    for definition in schema["$defs"].values():
        assert definition["description"]
        assert not definition["description"].startswith("Supporting schema")
        for property_schema in definition.get("properties", {}).values():
            assert property_schema["description"]
    for example in schema["examples"]:
        FigureSpec.model_validate(example)
    conventions = schema["x-hakowan-conventions"]
    assert "LayerPropertiesSpec.data" in conventions["geometry_source"]
    assert "diffuse material" in conventions["scalar_field"]
    assert "first value" in conventions["channel_composition"]
    assert (
        schema["$defs"]["LayerPropertiesSpec"]["properties"]["data"]["x-hakowan-role"]
        == "geometry_source"
    )
    assert (
        schema["$defs"]["ScalarFieldTextureSpec"]["properties"]["data"][
            "x-hakowan-role"
        ]
        == "attribute_reference"
    )


def test_published_schema_is_synchronized():
    published = Path(__file__).parents[1] / "docs" / "schema" / "v1.json"
    assert json.loads(published.read_text(encoding="utf-8")) == hkw.schema()


def test_layer_round_trip_preserves_canonical_composition_and_behavior():
    mesh = _mesh_with_fields()
    condition = compile_expression("value > 0.2")
    base = hkw.layer(mesh, name="input").transform(
        hkw.transform.Filter(data="value", condition=condition)
    )
    surface = base.material(
        "Diffuse",
        hkw.texture.ScalarField(
            hkw.attribute(
                "value",
                scale=hkw.scale.Normalize(
                    range_min=0.0,
                    range_max=1.0,
                    domain_min=0.0,
                    domain_max=1.0,
                ),
            ),
            colormap="magma",
        ),
    )
    points = base.mark("Point").channel(size=0.04).material("Diffuse", "white")
    vectors = (
        hkw.layer(mesh)
        .mark("Curve")
        .channel(
            vector_field=hkw.channel.VectorField(
                data="vector", end_type="arrow", normalize=True
            ),
            size=0.01,
        )
        .material("Diffuse", "orange")
    )
    layer = (surface + points) | vectors

    spec = hkw.to_spec(layer, data_ids={id(mesh): "mesh"})
    rebuilt = hkw.from_json(spec.to_json(), data_resolver={"mesh": mesh})
    rebuilt_spec = hkw.to_spec(rebuilt, data_ids={id(mesh): "mesh"})

    assert spec.to_json(canonical=True) == rebuilt_spec.to_json(canonical=True)
    assert hkw.validate(rebuilt, backend="webgl").valid
    assert len(hkw.compile(rebuilt)) == 3


def test_transform_arrays_follow_application_order():
    mesh = _mesh_with_fields()
    translate = np.eye(4)
    translate[:3, 3] = [1.0, 0.0, 0.0]
    scale = np.eye(4)
    scale[:3, :3] *= 2.0
    runtime_chain = hkw.transform.Affine(translate) * hkw.transform.Affine(scale)

    spec = hkw.to_spec(
        hkw.layer(mesh).transform(runtime_chain), data_ids={id(mesh): "mesh"}
    )
    matrices = [item.matrix for item in spec.root.spec.transforms]
    rebuilt = hkw.from_spec(spec, data_resolver={"mesh": mesh})

    assert matrices == [scale.tolist(), translate.tolist()]
    original_vertices = hkw.compile(hkw.layer(mesh).transform(runtime_chain))[
        0
    ].data_frame.mesh.vertices
    rebuilt_vertices = hkw.compile(rebuilt)[0].data_frame.mesh.vertices
    np.testing.assert_allclose(rebuilt_vertices, original_vertices)


def test_mesh_file_reference_round_trip(tmp_path):
    mesh = _mesh_with_fields()
    path = tmp_path / "mesh.ply"
    lagrange.io.save_mesh(path, mesh)

    spec = hkw.layer(path).to_spec()
    rebuilt = hkw.from_spec(spec)

    assert spec.root.spec.data.kind == "mesh_file"
    assert spec.root.spec.data.path == path.as_posix()
    assert rebuilt.to_spec().to_json(canonical=True) == spec.to_json(canonical=True)


def test_load_spec_and_layer_resolve_relative_resources(tmp_path):
    mesh = _mesh_with_fields()
    lagrange.io.save_mesh(tmp_path / "mesh.ply", mesh)
    (tmp_path / "texture.png").write_bytes(b"placeholder")
    spec = FigureSpec.model_validate(
        {
            "$schema": "https://hakowan.github.io/hakowan/schema/v1.json",
            "version": "1.0",
            "root": {
                "kind": "layer",
                "spec": {
                    "data": {"kind": "mesh_file", "path": "mesh.ply"},
                    "channels": {
                        "material": {
                            "kind": "diffuse",
                            "reflectance": {
                                "kind": "image",
                                "path": "texture.png",
                            },
                        }
                    },
                },
            },
        }
    )
    spec_path = tmp_path / "figure.json"
    spec.save(spec_path)

    loaded_spec = hkw.load_spec(spec_path)
    layer = hkw.load_layer(spec_path)
    material = layer._spec.channels[0]

    assert isinstance(loaded_spec, FigureSpec)
    assert layer._spec.data.source.as_posix() == "mesh.ply"
    assert material.reflectance.filename == tmp_path / "texture.png"
    assert layer.to_spec().to_json(canonical=True) == spec.to_json(canonical=True)


def test_external_data_requires_resolvers():
    mesh = _mesh_with_fields()
    layer = hkw.layer(mesh)

    with pytest.raises(hkw.SpecConversionError, match="requires data_ids"):
        hkw.to_spec(layer)

    spec = hkw.to_spec(layer, data_ids={id(mesh): "mesh"})
    with pytest.raises(hkw.SpecConversionError, match="requires data_resolver"):
        hkw.from_spec(spec)


def test_external_callable_round_trip():
    mesh = _mesh_with_fields()

    def square(value):
        return value**2

    layer = hkw.layer(mesh).material(
        "Diffuse",
        hkw.texture.ScalarField(hkw.attribute("value", scale=hkw.scale.Custom(square))),
    )
    with pytest.raises(hkw.SpecConversionError, match="requires function_ids"):
        hkw.to_spec(layer, data_ids={id(mesh): "mesh"})

    spec = hkw.to_spec(
        layer,
        data_ids={id(mesh): "mesh"},
        function_ids={square: "square"},
    )

    with pytest.raises(hkw.SpecConversionError, match="function_resolver"):
        hkw.from_spec(spec, data_resolver={"mesh": mesh})

    rebuilt = hkw.from_spec(
        spec,
        data_resolver={"mesh": mesh},
        function_resolver={"square": square},
    )
    rebuilt_spec = hkw.to_spec(
        rebuilt,
        data_ids={id(mesh): "mesh"},
        function_ids={square: "square"},
    )
    assert rebuilt_spec.to_json(canonical=True) == spec.to_json(canonical=True)


def test_expression_filter_executes_without_eval():
    payload = {
        "$schema": "https://hakowan.github.io/hakowan/schema/v1.json",
        "version": "1.0",
        "root": {
            "kind": "layer",
            "spec": {
                "data": {"kind": "external", "id": "mesh"},
                "mark": "point",
                "transforms": [
                    {
                        "kind": "filter",
                        "data": {"name": "value"},
                        "condition": {
                            "kind": "expression",
                            "source": "value >= 0.5 and isfinite(value)",
                        },
                    }
                ],
            },
        },
    }
    mesh = _mesh_with_fields()

    layer = hkw.from_spec(payload, data_resolver={"mesh": mesh})
    scene = hkw.compile(layer)

    assert scene[0].data_frame.mesh.num_vertices == 2


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('echo unsafe')",
        "value.__class__",
        "[x for x in value]",
        "open('/tmp/file')",
        "value ** 100",
        "([0] * 16) * 16",
        "(9 ** 16) ** 16",
        "value\x00",
    ],
)
def test_expression_rejects_unsafe_syntax(source):
    with pytest.raises(ExpressionError):
        compile_expression(source)

    with pytest.raises(ValidationError):
        ExpressionSpec(source=source)


def test_unknown_fields_and_kinds_are_rejected():
    payload = {
        "$schema": "https://hakowan.github.io/hakowan/schema/v1.json",
        "version": "1.0",
        "root": {
            "kind": "layer",
            "spec": {"unknown": True},
        },
    }
    with pytest.raises(ValidationError):
        FigureSpec.model_validate(payload)

    payload["root"]["kind"] = "mystery"
    payload["root"]["spec"] = {}
    with pytest.raises(ValidationError):
        FigureSpec.model_validate(payload)

    payload["root"]["kind"] = "layer"
    payload["version"] = "2.0"
    with pytest.raises(ValidationError):
        FigureSpec.model_validate(payload)

    payload["version"] = "1.0"
    payload["root"]["spec"] = {}
    payload["scene"] = {"camera": {"kind": "perspective"}}
    with pytest.raises(ValidationError, match="does not support scene"):
        FigureSpec.model_validate(payload)


def test_specification_nesting_is_bounded():
    root: dict = {"kind": "layer", "spec": {}}
    for _ in range(70):
        root = {"kind": "inherit", "spec": {}, "child": root}

    with pytest.raises(ValidationError, match="nesting exceeds"):
        FigureSpec.model_validate({"version": "1.0", "root": root})

    mesh = _mesh_with_fields()
    layer = hkw.layer(mesh)
    for index in range(70):
        layer = layer.name(f"layer-{index}")
    with pytest.raises(hkw.SpecConversionError, match="nesting exceeds"):
        hkw.to_spec(layer, data_ids={id(mesh): "mesh"})


def test_duplicate_channels_are_rejected_at_canonical_boundary():
    mesh = _mesh_with_fields()
    layer = hkw.layer(mesh)
    layer.channel(size=0.1, in_place=True)
    layer.channel(size=0.2, in_place=True)

    with pytest.raises(hkw.SpecConversionError, match="duplicate channels"):
        hkw.to_spec(layer, data_ids={id(mesh): "mesh"})


def test_canonical_json_is_deterministic_and_compact():
    mesh = _mesh_with_fields()
    spec = hkw.to_spec(hkw.layer(mesh), data_ids={id(mesh): "mesh"})

    first = spec.to_json(canonical=True)
    second = FigureSpec.from_json(first).to_json(canonical=True)

    assert first == second
    assert "\n" not in first
    assert "NaN" not in first


def _assert_runtime_round_trip(layer, mesh):
    spec = hkw.to_spec(layer, data_ids={id(mesh): "mesh"})
    rebuilt = hkw.from_spec(spec, data_resolver={"mesh": mesh})
    rebuilt_spec = hkw.to_spec(rebuilt, data_ids={id(mesh): "mesh"})
    assert rebuilt_spec.to_json(canonical=True) == spec.to_json(canonical=True)


@pytest.mark.parametrize(
    "material",
    [
        hkw.material.Diffuse("red"),
        hkw.material.Conductor(material="Au"),
        hkw.material.RoughConductor(material="Cu", alpha=0.2),
        hkw.material.Plastic(diffuse_reflectance="ivory"),
        hkw.material.RoughPlastic(diffuse_reflectance="blue", alpha=0.3),
        hkw.material.Principled(color="orange", roughness=0.4, metallic=0.2),
        hkw.material.ThinPrincipled(color="white", diff_trans=0.5),
        hkw.material.Dielectric(int_ior="water"),
        hkw.material.ThinDielectric(int_ior="bk7"),
        hkw.material.RoughDielectric(int_ior="water", alpha=0.15),
        hkw.material.Hair(
            root_color=[0.1, 0.05, 0.02],
            tip_color=[0.8, 0.6, 0.3],
            color_variation=0.2,
        ),
    ],
)
def test_all_material_variants_round_trip(material):
    mesh = _mesh_with_fields()
    _assert_runtime_round_trip(hkw.layer(mesh).channel(material=material), mesh)


@pytest.mark.parametrize(
    "scale",
    [
        hkw.scale.Uniform(factor=2.0),
        hkw.scale.Log(base=2.0),
        hkw.scale.Clip(domain=(0.0, 1.0)),
        hkw.scale.Normalize(
            range_min=0.0, range_max=1.0, domain_min=0.0, domain_max=2.0
        ),
        hkw.scale.Affine(matrix=np.eye(3)),
        hkw.scale.Norm(order=2),
        hkw.scale.Offset(offset=hkw.attribute("vector")),
    ],
)
def test_all_serializable_scale_variants_round_trip(scale):
    mesh = _mesh_with_fields()
    layer = hkw.layer(mesh).channel(position=hkw.attribute("vector", scale=scale))
    _assert_runtime_round_trip(layer, mesh)


@pytest.mark.parametrize(
    "transform",
    [
        hkw.transform.Clip(point=[0, 0, 0], normal=[1, 0, 0]),
        hkw.transform.UVMesh(uv="value"),
        hkw.transform.Affine(matrix=np.eye(4)),
        hkw.transform.PrincipalAxes(frame=np.eye(3)),
        hkw.transform.Normalize(),
        hkw.transform.Compute(x="x", component="component"),
        hkw.transform.Explode(pieces="value", magnitude=0.5),
        hkw.transform.Norm(data="vector", norm_attr_name="magnitude"),
        hkw.transform.Boundary(attributes=["value"]),
        hkw.transform.Streamline(vec_field="vector", n=4, cross_field=False),
        hkw.transform.Fur(vec_field="vector", n=4, children=2),
    ],
)
def test_all_non_callable_transform_variants_round_trip(transform):
    mesh = _mesh_with_fields()
    _assert_runtime_round_trip(hkw.layer(mesh).transform(transform), mesh)


def test_remaining_channel_and_texture_variants_round_trip():
    mesh = _mesh_with_fields()
    layer = (
        hkw.layer(mesh)
        .mark("Point")
        .channel(
            position="vector",
            normal="vector",
            size="value",
            shape=hkw.channel.Shape(base_shape="disk", orientation="vector"),
            covariance=hkw.channel.Covariance(data="vector", full=True),
            bump_map=hkw.channel.BumpMap(
                hkw.texture.Checkerboard(
                    uv="vector", texture1="black", texture2="white", size=16
                ),
                scale=0.5,
            ),
            normal_map=hkw.channel.NormalMap(
                hkw.texture.Image("normal.png", uv="vector", raw=True)
            ),
        )
    )
    _assert_runtime_round_trip(layer, mesh)
