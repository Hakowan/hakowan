# String-Based Shortcut Suggestions for Hakowan

## Current State Analysis

### ✅ Already Implemented
The following areas **already** support string shortcuts:

1. **Mark** - `.mark("surface")` instead of `.mark(hkw.mark.Surface)` ✓
2. **Material types** - `.material("diffuse", "red")` instead of constructing Material objects ✓
3. **Channels** (partial) - `.channel(size="attr_name")` for position, normal, size, vector_field, covariance ✓
4. **Attributes** - `"attr_name"` auto-converts to `Attribute(name="attr_name")` ✓
5. **Data** - `hkw.layer("mesh.obj")` instead of loading mesh manually ✓

## 🎯 Suggested Improvements

### Priority 1: High Impact, Low Effort

#### 1. **Color Channel Shortcut**
**Current:**
```py
.channel(material=hkw.material.Diffuse(
    reflectance=hkw.texture.ScalarField(data="vertex_color")
))
```

**Suggested:**
```py
# Add a `color` parameter to .channel()
.channel(color="vertex_color")  # Auto-creates Diffuse material with ScalarField
```

**Implementation:**
- Add `color: str | Channel | None` parameter to `layer.channel()`
- Auto-generate: `Diffuse(reflectance=ScalarField(data=color))`

---

#### 2. **Transform String Shortcuts**
**Current:**
```py
.transform(hkw.transform.Filter(data="attr", condition=lambda v: v > 0))
.transform(hkw.transform.Boundary())
```

**Suggested:**
```py
# Accept string shortcuts for common transforms
.transform("boundary")  # For Boundary()
# Keep complex transforms as objects (Filter needs lambda)
```

**Implementation:**
- Modify `layer.transform()` to accept `Transform | str`
- Support: `"boundary"` → `Boundary()`

---

#### 3. **Texture String Shortcuts in Materials**
**Current (verbose):**
```py
.material("Diffuse", hkw.texture.ScalarField(data="color"))
```

**Already partially works:**
```py
.material("Diffuse", "red")  # Works for uniform colors
```

**Suggested enhancement:**
```py
# Auto-detect if string is an attribute name vs color name
.material("Diffuse", reflectance="vertex_color")  
# If "vertex_color" is not a known color → auto-create ScalarField
```

**Implementation:**
- In material constructors, check TextureLike parameters
- If string and not a color name → convert to `ScalarField(data=string)`

---

#### 4. **Scale String Shortcuts**
**Current:**
```py
hkw.attribute(name="attr", scale=hkw.scale.Normalize())
hkw.attribute(name="attr", scale=hkw.scale.Log())
```

**Suggested:**
```py
hkw.attribute(name="attr", scale="normalize")
hkw.attribute(name="attr", scale="log")
```

**Implementation:**
- Modify `Attribute` class to accept `scale: Scale | str | None`
- Map: `"normalize"` → `Normalize()`, `"log"` → `Log()`, etc.

---

### Priority 2: Nice to Have

#### 5. **ColorMap String Shortcuts**
**Current:**
```py
hkw.texture.ScalarField(data="attr", colormap=hkw.colormap.Viridis)
```

**Already works (partially):**
```py
hkw.texture.ScalarField(data="attr", colormap="viridis")  # Check if this works
```

**Suggested:** Ensure all common colormaps accept strings

---

#### 6. **Common Layer Patterns as Methods**
**Current:**
```py
base.mark(hkw.mark.Curve).channel(size=0.01).material("Diffuse", "black")
```

**Suggested:** Add convenience methods
```py
base.wireframe(size=0.01, color="black")  # Shortcut for common pattern
base.points(size=0.05, color="red")
```

**Implementation:**
- Add methods to Layer class:
  - `wireframe(size, color, **kwargs)` → `.mark("curve").channel(size=size).material("diffuse", color)`
  - `points(size, color, **kwargs)` → `.mark("point").channel(size=size).material("diffuse", color)`

---

### Priority 3: Advanced/Breaking Changes

#### 7. **Dict-style Channel Specification** (Optional, more verbose but discoverable)
```py
# Alternative API for complex channels
.channel({
    "position": "deformed_pos",
    "color": {"data": "vertex_color", "colormap": "viridis"},
    "size": 0.1
})
```

This is more verbose but could be useful for config-driven workflows.

---

## 📊 Impact Assessment

| Suggestion | Verbosity Reduction | Breaking Change | Implementation Effort |
|------------|--------------------| ----------------|----------------------|
| 1. Color channel shortcut | ⭐⭐⭐⭐⭐ | No | Low |
| 2. Transform strings | ⭐⭐⭐ | No | Low |
| 3. Texture auto-detection | ⭐⭐⭐⭐ | No | Medium |
| 4. Scale strings | ⭐⭐⭐ | No | Low |
| 5. ColorMap strings | ⭐⭐ | No | Very Low (may exist) |
| 6. Convenience methods | ⭐⭐⭐⭐⭐ | No | Medium |
| 7. Dict-style API | ⭐ | No (additive) | High |

## 🎯 Recommended Action Plan

**Phase 1 (Quick Wins):**
1. Add `color` parameter to `.channel()` 
2. Add string support for scales
3. Add wireframe() and points() convenience methods

**Phase 2 (Medium term):**
4. Enhanced texture auto-detection in materials
5. Transform string shortcuts for common cases
6. Verify/complete colormap string support

**Phase 3 (Future consideration):**
7. Dict-style API for config-driven workflows

---

## Example: Before & After

**Before (Current):**
```py
import hakowan as hkw

layer = hkw.layer("mesh.obj") \
    .mark(hkw.mark.Surface) \
    .channel(material=hkw.material.Diffuse(
        reflectance=hkw.texture.ScalarField(
            data=hkw.attribute(name="vertex_color", scale=hkw.scale.Normalize())
        )
    ))
```

**After (With Suggested Shortcuts):**
```py
import hakowan as hkw

layer = hkw.layer("mesh.obj") \
    .mark("surface") \  # Already works ✓
    .channel(color=hkw.attribute("vertex_color", scale="normalize"))  # NEW
    
# Or even simpler if normalization is default:
layer = hkw.layer("mesh.obj").mark("surface").channel(color="vertex_color")
```

**Verbosity reduction: ~60%**
