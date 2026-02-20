import pytest
from hakowan import config
from hakowan.setup.integrator import AOV, Path


class TestConfigDefaults:
    def test_default(self):
        cfg = config()
        assert len(cfg.emitters) > 0
        assert cfg.emitters[0].filename.exists()

    def test_render_passes_empty_by_default(self):
        cfg = config()
        assert cfg.render_passes == set()
        assert not cfg.albedo
        assert not cfg.depth
        assert not cfg.normal
        assert not cfg.facet_id

    def test_integrator_is_path_by_default(self):
        cfg = config()
        assert isinstance(cfg.integrator, Path)


class TestRenderPassesInterface:
    """Tests for the unified render_passes set and bulk assignment."""

    def test_assign_set(self):
        cfg = config()
        cfg.render_passes = {"albedo", "depth"}
        assert "albedo" in cfg.render_passes
        assert "depth" in cfg.render_passes
        assert "normal" not in cfg.render_passes
        assert "facet_id" not in cfg.render_passes

    def test_assign_list(self):
        cfg = config()
        cfg.render_passes = ["normal", "depth"]
        assert cfg.render_passes == {"normal", "depth"}

    def test_assign_empty_clears(self):
        cfg = config()
        cfg.albedo = True
        cfg.depth = True
        cfg.render_passes = set()
        assert cfg.render_passes == set()

    def test_render_passes_is_set_type(self):
        cfg = config()
        cfg.render_passes = {"albedo", "albedo"}  # duplicates should collapse
        assert len(cfg.render_passes) == 1


class TestBooleanAliases:
    """Verify that the four convenience properties add/remove the right strings."""

    def test_albedo_setter_adds_key(self):
        cfg = config()
        cfg.albedo = True
        assert "albedo" in cfg.render_passes

    def test_albedo_getter(self):
        cfg = config()
        assert not cfg.albedo
        cfg.albedo = True
        assert cfg.albedo

    def test_albedo_remove(self):
        cfg = config()
        cfg.albedo = True
        cfg.albedo = False
        assert "albedo" not in cfg.render_passes

    def test_depth_setter_adds_key(self):
        cfg = config()
        cfg.depth = True
        assert "depth" in cfg.render_passes

    def test_normal_setter_adds_key(self):
        cfg = config()
        cfg.normal = True
        assert "normal" in cfg.render_passes

    def test_facet_id_setter_adds_key(self):
        cfg = config()
        cfg.facet_id = True
        assert "facet_id" in cfg.render_passes

    def test_setting_one_does_not_affect_others(self):
        cfg = config()
        cfg.albedo = True
        assert not cfg.depth
        assert not cfg.normal
        assert not cfg.facet_id

    def test_multiple_aliases_independent(self):
        cfg = config()
        cfg.albedo = True
        cfg.depth = True
        cfg.normal = True
        cfg.facet_id = True
        assert cfg.albedo
        assert cfg.depth
        assert cfg.normal
        assert cfg.facet_id
        assert cfg.render_passes == {"albedo", "depth", "normal", "facet_id"}


class TestAovSync:
    """Verify that __sync_aovs keeps Config.integrator consistent with render_passes."""

    def test_albedo_creates_aov_integrator(self):
        cfg = config()
        cfg.albedo = True
        assert isinstance(cfg.integrator, AOV)
        assert "albedo:albedo" in cfg.integrator.aovs

    def test_depth_creates_aov_integrator(self):
        cfg = config()
        cfg.depth = True
        assert isinstance(cfg.integrator, AOV)
        assert "depth:depth" in cfg.integrator.aovs

    def test_normal_creates_aov_integrator(self):
        cfg = config()
        cfg.normal = True
        assert isinstance(cfg.integrator, AOV)
        assert "sh_normal:sh_normal" in cfg.integrator.aovs

    def test_facet_id_does_not_create_aov(self):
        """facet_id is Blender-only and has no Mitsuba AOV counterpart."""
        cfg = config()
        cfg.facet_id = True
        assert isinstance(cfg.integrator, Path)

    def test_multiple_passes_all_in_aov(self):
        cfg = config()
        cfg.albedo = True
        cfg.depth = True
        cfg.normal = True
        assert isinstance(cfg.integrator, AOV)
        aovs = cfg.integrator.aovs
        assert "albedo:albedo" in aovs
        assert "depth:depth" in aovs
        assert "sh_normal:sh_normal" in aovs

    def test_remove_one_pass_updates_aov(self):
        cfg = config()
        cfg.albedo = True
        cfg.depth = True
        cfg.albedo = False
        assert isinstance(cfg.integrator, AOV)
        assert "depth:depth" in cfg.integrator.aovs
        assert "albedo:albedo" not in cfg.integrator.aovs

    def test_remove_all_passes_strips_aov_wrapper(self):
        cfg = config()
        cfg.albedo = True
        cfg.albedo = False
        assert isinstance(cfg.integrator, Path)

    def test_bulk_assign_syncs_aov(self):
        cfg = config()
        cfg.render_passes = {"albedo", "normal"}
        assert isinstance(cfg.integrator, AOV)
        assert "albedo:albedo" in cfg.integrator.aovs
        assert "sh_normal:sh_normal" in cfg.integrator.aovs
        assert "depth:depth" not in cfg.integrator.aovs

    def test_bulk_assign_empty_strips_aov(self):
        cfg = config()
        cfg.albedo = True
        cfg.render_passes = set()
        assert isinstance(cfg.integrator, Path)

    def test_aov_preserves_base_integrator(self):
        """The inner integrator of the AOV wrapper should be the original Path."""
        cfg = config()
        cfg.albedo = True
        assert isinstance(cfg.integrator, AOV)
        assert isinstance(cfg.integrator.integrator, Path)


class TestCoordinateSystemHelpers:
    """Sanity checks for z_up / z_down / y_up / y_down."""

    def test_z_up_sets_float_rotation(self):
        cfg = config()
        cfg.z_up()
        for emitter in cfg.emitters:
            assert isinstance(emitter.rotation, float)
            assert emitter.rotation == 180.0

    def test_z_down_sets_float_rotation(self):
        cfg = config()
        cfg.z_down()
        for emitter in cfg.emitters:
            assert isinstance(emitter.rotation, float)
            assert emitter.rotation == 180.0

    def test_y_up_sets_float_rotation(self):
        cfg = config()
        cfg.y_up()
        for emitter in cfg.emitters:
            assert isinstance(emitter.rotation, float)
            assert emitter.rotation == 180.0

    def test_y_down_sets_float_rotation(self):
        cfg = config()
        cfg.y_down()
        for emitter in cfg.emitters:
            assert isinstance(emitter.rotation, float)
            assert emitter.rotation == 180.0
