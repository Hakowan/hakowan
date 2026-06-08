import numpy as np
import numpy.typing as npt
from typing import Iterator, TypeAlias

ColorLike: TypeAlias = float | int | str | tuple | list
"""Color-like type alias is types that can be converted to a color."""


class Color:
    """A minimal representation of color."""

    @classmethod
    def from_hex(cls, hex_color: str) -> "Color":
        """Construct color from hex."""
        c = hex_color.lstrip("#")
        assert len(c) == 6
        return Color(
            int(c[0:2], 16) / 255.0,
            int(c[2:4], 16) / 255.0,
            int(c[4:6], 16) / 255.0,
        )

    def __init__(self, red: float = 0.0, green: float = 0.0, blue: float = 0.0) -> None:
        """Construct color from RGB."""
        self.color = np.array([red, green, blue])

    def __getitem__(self, i: int) -> float:
        """Get color channel."""
        return self.color[i]

    def __add__(self, other: "Color") -> "Color":
        """Add two colors."""
        c = self.color + other.color
        return Color(*c)

    def __mul__(self, scale: float) -> "Color":
        """Multiply color by a scalar."""
        c = self.color * scale
        return Color(*c)

    def __rmul__(self, scale: float) -> "Color":
        """Multiply color by a scalar."""
        return self.__mul__(scale)

    def __eq__(self, other: object) -> bool:
        """Check if two colors are equal."""
        if not isinstance(other, Color):
            return NotImplemented
        return bool(np.all(self.color == other.color))

    def __ne__(self, other: object) -> bool:
        """Check if two colors are not equal."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __iter__(self) -> Iterator[float]:
        """Iterate over color channels."""
        return self.color.__iter__()

    def __repr__(self) -> str:
        """String representation of color."""
        return f"Color ({self.red}, {self.green}, {self.blue})"

    @property
    def red(self) -> float:
        """Red channel."""
        return self.color[0]

    @property
    def green(self) -> float:
        """Green channel."""
        return self.color[1]

    @property
    def blue(self) -> float:
        """Blue channel."""
        return self.color[2]

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """Raw RGB data as a numpy array."""
        return self.color
