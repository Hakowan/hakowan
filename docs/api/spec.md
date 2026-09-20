# Canonical specification

The canonical specification is the stable JSON boundary for Hakowan figures.
See the [schema guide](../guide/schema.md) for format semantics, resolver rules,
versioning, and complete examples.

## Public functions

::: hakowan.spec.codec.to_spec

::: hakowan.spec.codec.from_spec

::: hakowan.spec.codec.from_json

::: hakowan.spec.codec.load_spec

::: hakowan.spec.codec.load_layer

::: hakowan.spec.model.json_schema

## Root model

::: hakowan.spec.model.FigureSpec
    options:
      members:
        - to_dict
        - to_json
        - save
        - from_json
        - load

## Boundary errors

::: hakowan.spec.codec.SpecConversionError

::: hakowan.spec.expression.ExpressionError

## Expression compiler

::: hakowan.spec.expression.compile_expression
