"""
    Class methods for the Sigma main class

    Adding and registering new Sigma structures
"""
from ..structures import *
import warnings


def add(self, structure):
    """
        Add a Sigma structure, one of Type, Predicate, or Conditional

        "Fill in my data parameters INITIALIZATION."
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    # Check validity of argument
    if not isinstance(structure, (Type, Predicate, Conditional)):
        raise ValueError("structure must be one of Sigma's Type, Predicate, or Conditional")

    # Register new Type
    if isinstance(structure, Type):
        self._register_type(structure)

    # Register new Predicate
    if isinstance(structure, Predicate):
        self._register_predicate(structure)
        self._compile_predicate(structure)
        self._order_set = False  # Need new node ordering

    # Register new Conditional
    if isinstance(structure, Conditional):
        self._register_conditional(structure)
        self._compile_conditional(structure)
        self._order_set = False  # Need new node ordering


def add_type(self, *args, **kwargs):
    """
        Add a Sigma type to this Sigma program directly. Return the created Type instance
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    sigma_type = Type(*args, **kwargs)
    self.add(sigma_type)
    return sigma_type


def add_predicate(self, *args, **kwargs):
    """
        Add a Sigma predicate to this Sigma program directly. Return the created Predicate instance
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    predicate = Predicate(*args, **kwargs)
    self.add(predicate)
    return predicate


def add_conditional(self, *args, **kwargs):
    """
        Add a Sigma conditional to this Sigma program directly. Return the created Conditional Instance
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    conditional = Conditional(*args, **kwargs)
    self.add(conditional)
    return conditional


def _register_type(self, sigma_type):
    """
        Register a new type in this sigma program
    """
    from .. import Sigma
    assert isinstance(self, Sigma)
    assert isinstance(sigma_type, Type)

    # Check for duplication
    if sigma_type.name in self.name2type.keys():
        raise ValueError("The Sigma type '{}' has already been registered in the current model. Please check for "
                         "duplication or register with a different type name"
                         .format(extern_name(sigma_type.name, "type")))

    self.type_list.append(sigma_type)
    self.name2type[sigma_type.name] = sigma_type


def _register_predicate(self, predicate):
    """
        Register a new predicate in this sigma program. Check against existing type entries and fill up relevant
            lookup tables
    """
    from .. import Sigma
    assert isinstance(self, Sigma)
    assert isinstance(predicate, Predicate)

    # Check for duplication
    if predicate.name in self.name2predicate.keys():
        raise ValueError("The Sigma predicate '{}' has already been registered in the current model. Please check for "
                         "duplication or register with a different predicate name"
                         .format(extern_name(predicate.name, "predicate")))
    # Check if all types referenced by the predicate's arguments has been registered in the model
    #   If not, throw a user warning but proceed to have it registered
    for var_name, var_type in predicate.relvar_name2type.items():
        if var_type.name not in self.name2type.keys():
            warnings.warn("UNREGISTERED TYPE: The type '{}' referenced by the relational argument '{}' in predicate "
                          "'{}' is not registered in the current model. Sigma will try to have it registered."
                          .format(extern_name(var_type.name, "type"), var_name,
                                  extern_name(predicate.name, "predicate")))
            self.add(var_type)
    if predicate.ranvar_type.name not in self.name2type.keys():
        warnings.warn("UNREGISTERED TYPE: The type '{}' referenced by the random variable in predicate '{}' is not "
                      "registered in the current model. Sigma will try to have it registered."
                      .format(extern_name(predicate.ranvar_type.name, "type"),
                              extern_name(predicate.name, "predicate")))

    # Register predicate
    self.predicate_list.append(predicate)
    self.name2predicate[predicate.name] = predicate


def _register_conditional(self, conditional):
    """
        Register a new conditional in this sigma program. Check against existing type and predicate entries.
            Fill up relevant lookup tables.
    """
    from .. import Sigma
    assert isinstance(self, Sigma)
    assert isinstance(conditional, Conditional)

    # Check for duplication
    if conditional.name in self.name2conditional.keys():
        raise ValueError("The Sigma conditional '{}' has already been registered in the current model. Please check "
                         "for duplication or register with a different conditional name"
                         .format(extern_name(conditional.name, "conditional")))
    # Check if all predicates referenced by conditional's predicate patterns has been registered in the model
    #   If not, throw a user warning but proceed to have it registered
    for pat in conditional.conditions + conditional.condacts + conditional.actions:
        if pat[0].name not in self.name2conditional.keys():
            warnings.warn("UNREGISTERED PREDICATE: The predicate '{}' referenced by the predicate pattern '{}' in "
                          "conditional  '{}' is not registered in the current model. Sigma will try to have it "
                          "registered."
                          .format(extern_name(pat[0].name, "predicate"), pat,
                                  extern_name(conditional.name, "conditional")))

    # TODO: Register factor function. Also consider the case when function is shared with other conditionals

    # Register conditional
    self.conditional_list.append(conditional)
    self.name2conditional[conditional.name] = conditional
