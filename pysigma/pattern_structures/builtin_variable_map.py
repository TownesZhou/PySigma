"""
    Built-in VariableMapPrototype instances:
        - id_varmap: identity variable map.
"""
from .variable_map import VariableMap


# Identity map
@VariableMap(dynamic=False, strict=False)
def id_varmap(x):
    """
        A static, non-strict identity variable mapping.
    """
    return x
