"""
    Class methods for the Sigma main class

    Adding and registering new Sigma structures
"""
from ..structures import *
from ..graphical._structures import *


def add(self, structure):
    """
        Add a Sigma structure, one of Type, Predicate, or Conditional

        "Fill in my data parameters INITIALIZATION."
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    # Check validity of argument
    if type(structure) not in [Type, Predicate, Conditional]:
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
        Add a Sigma type to this Sigma program directly.
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    sigma_type = Type(*args, **kwargs)
    self.add(sigma_type)


def add_predicate(self, *args, **kwargs):
    """
        Add a Sigma predicate to this Sigma program directly.
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    predicate = Predicate(*args, **kwargs)
    self.add(predicate)


def add_conditional(self, *args, **kwargs):
    """
        Add a Sigma conditional to this Sigma program directly.
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    conditional = Conditional(*args, **kwargs)
    self.add(conditional)


def _register_type(self, sigma_type):
    """
        Register a new type in this sigma program
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

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
    # Check if the types in the predicate are already defined, and change str to Type
    for i, argument_type in enumerate(predicate.wm_var_types):
        if argument_type not in self.name2type.keys() and argument_type not in self.type_list:
            raise ValueError("Predicate {} try to include type {} that has not yet been defined in this program"
                             .format(predicate.name, argument_type))

    # Register predicate
    self.predicate_list.append(predicate)
    self.name2predicate[predicate.name] = predicate

    # Register in self.unique_preds if this predicate is unique
    if predicate.selection:
        self.unique_preds.append(predicate)


def _register_conditional(self, conditional):
    """
        Register a new conditional in this sigma program. Check against existing type and predicate entries.
            Fill up relevant lookup tables.
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    assert isinstance(conditional, Conditional)
    # Check if predicate patterns align with already registered predicates
    for pattern in conditional.conditions + conditional.condacts + conditional.actions:
        # Check if predicate exists
        if pattern.predicate_name not in self.name2predicate.keys():
            raise ValueError("The predicate pattern '{}' includes an unknown predicate '{}'".format(pattern,
                                                                                                    pattern.predicate_name))
        # Check if the number of predicate elements is no greater than the number of WM vars of that predicate
        pred = self.name2predicate[pattern.predicate_name]
        if len(pattern.elements) > len(pred.var_list):
            raise ValueError("The number of predicate elements declared in the predicate pattern '{}', "
                             "currently {}, exceeds the number of working memory variable in the predicate "
                             "'{}', currently {}".format(pattern, len(pattern.elements), pattern.predicate_name,
                                                         len(pred.wm_var_names)))
        # Check if the 'argument_name' in the 'elements' agree with that predicate's WM vars
        for element in pattern.elements:
            if element.argument_name not in pred.var_name2var.keys():
                raise ValueError("The 'argument_name' '{}' declared in the pattern element '{}' of the "
                                 "predicate pattern '{}' is not one of the working memory variables of the "
                                 "predicate '{}'".format(element.argument_name, element, pattern,
                                                         pattern.predicate_name))

    # Set up lookup tables
    # TODO: Also add a check to make sure that user does not declare a pattern variable associated with multiple
    #       working memory variables with different uniqueness.
    for pt_name, ptv_vals in conditional.pattern_pt_vals.items():
        for wmv, ptv_val in ptv_vals.items():
            ptv_name = ptv_val["name"]
            # By default, pt var's sum op is determined by its corresponding wm vars' uniqueness
            # TODO: discuss this and extend to support custom sum ops
            pred = self.name2predicate[conditional.name2pattern[pt_name].predicate_name]
            # get associated wm var's summarization op
            wmvar_sum_op = pred.var_name2var[wmv].sum_op
            if ptv_val["type"] is "const":
                conditional.global_pt_vals[ptv_name] = {
                    "type": "const",
                    "size": len(ptv_val["vals"]) if type(ptv_val["vals"]) is list else 1,
                    "sum_op": wmvar_sum_op
                }
            else:
                ptv_size = pred.var_name2var[wmv].size
                # If variable not already exists, insert entry
                if ptv_name not in conditional.global_pt_vals.keys():
                    conditional.global_pt_vals[ptv_name] = {
                        "type": "var",
                        "size": ptv_size,
                        "sum_op": wmvar_sum_op
                    }
                # Otherwise if already present, check whether uniqueness agree, and if so, take max size
                else:

                    assert conditional.global_pt_vals[ptv_name]["sum_op"] == wmvar_sum_op, \
                        "Found conflict of uniqueness among working memory variables that is associated with the " \
                        "pattern variable '{}'".format(ptv_name)
                    # Take max size
                    conditional.global_pt_vals[ptv_name]["size"] = \
                        max(conditional.global_pt_vals[ptv_name]["size"], ptv_size)

    # If function specified as a str, check whether there already exists a conditional with that name.
    # If so, check if there variable dimensions match
    if type(conditional.function) is str:
        if conditional.function not in self.name2conditional.keys():
            raise ValueError(
                "Unknown conditional '{}' specified in the function field in the conditional '{}'"
                .format(conditional.function, conditional.name))
        else:
            alien_cond = self.name2conditional[conditional.function]
            for func_var_name in alien_cond.function_var_names:
                this_attri = conditional.global_pt_vals[func_var_name]
                alien_attri = alien_cond.global_pt_vals[func_var_name]
                if this_attri != alien_attri:
                    raise ValueError("The attributes of the function variable '{}' does not match when attempting"
                                     "to share function between conditional '{}' and '{}'.\n The attributes "
                                     "determined in conditional '{}' is: '{}'.\n The attributes determined in "
                                     "conditional '{}' is: '{}'.\n"
                                     .format(func_var_name, conditional.name, alien_cond.name,
                                             conditional.name, this_attri, alien_cond.name, alien_attri))

    # Register conditional
    self.conditional_list.append(conditional)
    self.name2conditional[conditional.name] = conditional
