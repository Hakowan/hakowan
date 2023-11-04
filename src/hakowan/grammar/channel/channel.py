from dataclasses import dataclass
from typing import Optional, Union

from ..scale import Attribute, AttributeLike


@dataclass(kw_only=True, slots=True)
class Channel:
    pass


class ScalarChannel(Channel):
    """Scalar channel is the base class class for channels that maps from a scalar field.

    Attributes:
        data (Attribute | float): The attribute or value used to encode the scalar field.
    """

    def __init__(self, data: AttributeLike | float):
        """Constructor

        Args:
            data (AttributeLike | float): The attribute used to encode the vector.

        Returns:
            (ScalarChannel): The constructed scalar channel.
        """
        self.__set_data(data)

    @property
    def data(self) -> Attribute | float:
        return self._data

    @data.setter
    def data(self, data: AttributeLike | float):
        self.__set_data(data)

    def __set_data(self, data: AttributeLike | float):
        match data:
            case Attribute():
                self._data = data
            case str():
                self._data = Attribute(name=data)
            case float():
                self._data = data

    _data: Attribute | float
    __slots__ = ("_data",)


class VectorChannel(Channel):
    """Vector channel is the base class class for channels that maps from a vector field.

    Attributes:
        data (Attribute): The attribute used to encode the vector field.
    """

    def __init__(self, data: AttributeLike):
        """Constructor

        Args:
            data (AttributeLike): The attribute used to encode the vector.

        Returns:
            (VectorChannel): The constructed vector channel.
        """
        self.__set_data(data)

    @property
    def data(self) -> Attribute:
        return self._data

    @data.setter
    def data(self, data: AttributeLike):
        self.__set_data(data)

    def __set_data(self, data: AttributeLike):
        match data:
            case Attribute():
                self._data = data
            case str():
                self._data = Attribute(name=data)

    _data: Attribute
    __slots__ = ("_data",)


class Position(VectorChannel):
    """Position channel

    This class is used to specify the mapping from an attribute to the position channel.
    Note that, by default, the vertex coordinates of the data frame is used as the position
    channel. Thus, this class is mainly useful when we want to use non-vertex-coordinates as the
    position channel. For example, this method can be used for visualizing a deformed shape when
    the deformed position is stored as a vertex attribute in the data frame.
    """

    def __init__(self, data: AttributeLike):
        """Constructor

        Args:
            data (AttributeLike): The attribute used to encode the position field.

        Returns:
            (Position): The constructed position channel.
        """
        super().__init__(data)

    __slots__ = ()


class Normal(VectorChannel):
    """Normal channel

    This class is used to specify the mapping from an attribute to the normal channel.
    By default, Hakowan will automatically compute the normal field from the geometry if normal
    channel is not specified. This class is useful for ensure the visualization uses a pre-defined
    normal field.
    """

    def __init__(self, data: AttributeLike):
        """Constructor

        Args:
            data (AttributeLike): The attribute used to encode the normal field.

        Returns:
            (Normal): The constructed normal channel.
        """
        super().__init__(data)

    __slots__ = ()


class Size(ScalarChannel):
    """Size channel

    This class is used to specify the mapping from an attribute or value to the size channel. If a
    value is used, all elements will have the same size. Note that size is defined in the same unit
    as the input geometry.
    """

    def __init__(self, data: AttributeLike | float):
        """Constructor

        Args:
            data (AttributeLike | float): The attribute or value used to encode the normal field.
                If an attribute is used, the size field is spatially varying. If a float value is
                used, the size field is spatially uniform.

        Returns:
            (Size): The constructed size channel.
        """
        super().__init__(data)

    __slots__ = ()


class VectorField(VectorChannel):
    """Vector field channel

    This class is used to specify the mapping from an attribute to the vector field channel.

    A vector field can be define over the vertices or facets of the geometry. The vector field must
    have the same dimension as the geometry.
    """

    def __init__(self, data: AttributeLike):
        """Constructor

        Args:
            data (AttributeLike): The attribute used to encode the vector field.

        Returns:
            (VectorField): The constructed vector field channel.
        """
        super().__init__(data)

    __slots__ = ()
