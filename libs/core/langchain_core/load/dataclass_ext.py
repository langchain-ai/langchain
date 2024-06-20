import sys
from dataclasses import MISSING, Field
from types import FunctionType
from typing import Iterable

_FIELDS = "__dataclass_fields__"
_POST_INIT_NAME = "__post_init__"


class _HAS_DEFAULT_FACTORY_CLASS:
    def __repr__(self):
        return "<factory>"


_HAS_DEFAULT_FACTORY = _HAS_DEFAULT_FACTORY_CLASS()


def _fields_in_init_order(fields: Iterable[Field]):
    # Returns the fields as __init__ will output them.  It returns 2 tuples:
    # the first for normal args, and the second for keyword args.

    return (
        tuple(f for f in fields if f.init and not f.kw_only),
        tuple(f for f in fields if f.init and f.kw_only),
    )


def _create_fn(name, args, body, *, globals=None, locals=None, return_type=MISSING):
    # Note that we may mutate locals. Callers beware!
    # The only callers are internal to this module, so no
    # worries about external callers.
    if locals is None:
        locals = {}
    return_annotation = ""
    if return_type is not MISSING:
        locals["_return_type"] = return_type
        return_annotation = "->_return_type"
    args = ",".join(args)
    body = "\n".join(f"  {b}" for b in body)

    # Compute the text of the entire function.
    txt = f" def {name}({args}){return_annotation}:\n{body}"

    local_vars = ", ".join(locals.keys())
    txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"
    ns = {}
    exec(txt, globals, ns)
    return ns["__create_fn__"](**locals)


def _field_assign(frozen, name, value, self_name):
    # If we're a frozen class, then assign to our fields in __init__
    # via object.__setattr__.  Otherwise, just use a simple
    # assignment.
    #
    # self_name is what "self" is called in this function: don't
    # hard-code "self", since that might be a field name.
    if frozen:
        return (
            f"__dataclass_builtins_object__.__setattr__({self_name},{name!r},{value})"  # noqa: E501
        )
    return f"{self_name}.{name}={value}"


def _field_init(f, frozen, globals, self_name, slots):
    # Return the text of the line in the body of __init__ that will
    # initialize this field.

    default_name = f"_dflt_{f.name}"
    if f.default_factory is not MISSING:
        if f.init:
            # This field has a default factory.  If a parameter is
            # given, use it.  If not, call the factory.
            globals[default_name] = f.default_factory
            value = (
                f"{default_name}() "
                f"if {f.name} is _HAS_DEFAULT_FACTORY "
                f"else {f.name}"
            )
        else:
            # This is a field that's not in the __init__ params, but
            # has a default factory function.  It needs to be
            # initialized here by calling the factory function,
            # because there's no other way to initialize it.

            # For a field initialized with a default=defaultvalue, the
            # class dict just has the default value
            # (cls.fieldname=defaultvalue).  But that won't work for a
            # default factory, the factory must be called in __init__
            # and we must assign that to self.fieldname.  We can't
            # fall back to the class dict's value, both because it's
            # not set, and because it might be different per-class
            # (which, after all, is why we have a factory function!).

            globals[default_name] = f.default_factory
            value = f"{default_name}()"
    else:
        # No default factory.
        if f.init:
            if f.default is MISSING:
                # There's no default, just do an assignment.
                value = f.name
            elif f.default is not MISSING:
                globals[default_name] = f.default
                value = f.name
        else:
            # If the class has slots, then initialize this field.
            if slots and f.default is not MISSING:
                globals[default_name] = f.default
                value = default_name
            else:
                # This field does not need initialization: reading from it will
                # just use the class attribute that contains the default.
                # Signify that to the caller by returning None.
                return None

    # Only test this now, so that we can create variables for the
    # default.  However, return None to signify that we're not going
    # to actually do the assignment statement for InitVars.
    if f._field_type.name == "_FIELD_INITVAR":
        return None

    # Now, actually generate the field assignment.
    return _field_assign(frozen, f.name, value, self_name)


def _init_param(f):
    # Return the __init__ parameter string for this field.  For
    # example, the equivalent of 'x:int=3' (except instead of 'int',
    # reference a variable set to int, and instead of '3', reference a
    # variable set to 3).
    if f.default is MISSING and f.default_factory is MISSING:
        # There's no default, and no default_factory, just output the
        # variable name and type.
        default = ""
    elif f.default is not MISSING:
        # There's a default, this will be the name that's used to look
        # it up.
        default = f"=_dflt_{f.name}"
    elif f.default_factory is not MISSING:
        # There's a factory function.  Set a marker.
        default = "=_HAS_DEFAULT_FACTORY"
    return f"{f.name}:_type_{f.name}{default}"


def _init_fn(
    fields, std_fields, kw_only_fields, frozen, has_post_init, self_name, globals, slots
):
    # fields contains both real fields and InitVar pseudo-fields.

    # Make sure we don't have fields without defaults following fields
    # with defaults.  This actually would be caught when exec-ing the
    # function source code, but catching it here gives a better error
    # message, and future-proofs us in case we build up the function
    # using ast.

    seen_default = False
    for f in std_fields:
        # Only consider the non-kw-only fields in the __init__ call.
        if f.init:
            if not (f.default is MISSING and f.default_factory is MISSING):
                seen_default = True
            elif seen_default:
                raise TypeError(
                    f"non-default argument {f.name!r} " "follows default argument"
                )

    locals = {f"_type_{f.name}": f.type for f in fields}
    locals.update(
        {
            "MISSING": MISSING,
            "_HAS_DEFAULT_FACTORY": _HAS_DEFAULT_FACTORY,
            "__dataclass_builtins_object__": object,
        }
    )

    body_lines = []
    for f in fields:
        line = _field_init(f, frozen, locals, self_name, slots)
        # line is None means that this field doesn't require
        # initialization (it's a pseudo-field).  Just skip it.
        if line:
            body_lines.append(line)

    # Does this class have a post-init function?
    if has_post_init:
        params_str = ",".join(
            f.name for f in fields if f._field_type.name == "_FIELD_INITVAR"
        )
        body_lines.append(f"{self_name}.{_POST_INIT_NAME}({params_str})")

    # If no body lines, use 'pass'.
    if not body_lines:
        body_lines = ["pass"]

    _init_params = [_init_param(f) for f in std_fields]
    if kw_only_fields:
        # Add the keyword-only args.  Because the * can only be added if
        # there's at least one keyword-only arg, there needs to be a test here
        # (instead of just concatenting the lists together).
        _init_params += ["*"]
        _init_params += [_init_param(f) for f in kw_only_fields]
    return _create_fn(
        "__init__",
        [self_name] + _init_params,
        body_lines,
        locals=locals,
        globals=globals,
        return_type=None,
    )


def _set_qualname(cls, value):
    # Ensure that the functions returned from _create_fn uses the proper
    # __qualname__ (the class they belong to).
    if isinstance(value, FunctionType):
        value.__qualname__ = f"{cls.__qualname__}.{value.__name__}"
    return value


def _set_new_attribute(cls, name, value):
    _set_qualname(cls, value)
    setattr(cls, name, value)
    return False


def set_init(cls) -> None:
    if cls.__module__ in sys.modules:
        globals = sys.modules[cls.__module__].__dict__
    else:
        # Theoretically this can happen if someone writes
        # a custom string to cls.__module__.  In which case
        # such dataclass won't be fully introspectable
        # (w.r.t. typing.get_type_hints) but will still function
        # correctly.
        globals = {}

    all_fields = getattr(cls, _FIELDS, None)
    all_init_fields = tuple(
        f for f in all_fields.values() if f._field_type.name != "_FIELD_CLASSVAR"
    )
    std_init_fields, kw_only_init_fields = _fields_in_init_order(all_init_fields)
    has_post_init = hasattr(cls, _POST_INIT_NAME)
    _set_new_attribute(
        cls,
        "__default_init__",
        _init_fn(
            all_init_fields,
            std_init_fields,
            kw_only_init_fields,
            False,
            has_post_init,
            # The name to use for the "self"
            # param in __init__.  Use "self"
            # if possible.
            "__dataclass_self__" if "self" in all_fields else "self",
            globals,
            False,
        ),
    )
