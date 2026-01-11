"""
Tests for Delta's type system.

Tests type definitions, type equality, substitution, and unification.
"""

import pytest
from delta.types.types import (
    Type, TypeVar, UnitType, BoolType, IntType, FloatType, StringType,
    TensorType, FunctionType, TupleType, StructType, GenericType,
    AnyType, NeverType,
    ShapeDim, ConcreteDim, SymbolicDim, DynamicDim,
    Substitution, apply_substitution, occurs_check,
    unify, UnificationError, fresh_type_var
)


class TestPrimitiveTypes:
    """Tests for primitive types."""
    
    def test_unit_type(self):
        """UnitType represents void."""
        unit = UnitType()
        assert str(unit) == "Unit"
    
    def test_bool_type(self):
        """BoolType represents booleans."""
        bool_t = BoolType()
        assert str(bool_t) == "Bool"
    
    def test_int_type(self):
        """IntType represents integers."""
        int_t = IntType()
        assert str(int_t) == "Int"
    
    def test_float_type(self):
        """FloatType represents floats."""
        float_t = FloatType()
        assert str(float_t) == "Float"
    
    def test_string_type(self):
        """StringType represents strings."""
        str_t = StringType()
        assert str(str_t) == "String"
    
    def test_primitive_equality(self):
        """Primitive types are equal to same types."""
        assert FloatType() == FloatType()
        assert IntType() == IntType()
        assert BoolType() == BoolType()
    
    def test_primitive_inequality(self):
        """Different primitive types are not equal."""
        assert FloatType() != IntType()
        assert BoolType() != FloatType()
        assert StringType() != IntType()
    
    def test_primitive_hashable(self):
        """Primitive types are hashable."""
        types = {FloatType(), IntType(), BoolType()}
        assert len(types) == 3


class TestTypeVar:
    """Tests for type variables."""
    
    def test_create_type_var(self):
        """Create a type variable."""
        tv = TypeVar(name="T")
        assert tv.name == "T"
        assert str(tv) == "T"
    
    def test_fresh_type_var(self):
        """Fresh type vars have unique names."""
        t1 = fresh_type_var("X")
        t2 = fresh_type_var("X")
        assert t1.name != t2.name
    
    def test_type_var_equality(self):
        """Type vars with same name are equal."""
        assert TypeVar("T") == TypeVar("T")
        assert TypeVar("T") != TypeVar("U")
    
    def test_type_var_free_vars(self):
        """Type var contains itself as free variable."""
        tv = TypeVar(name="T")
        assert tv in tv.free_vars()


class TestShapeDim:
    """Tests for tensor shape dimensions."""
    
    def test_concrete_dim(self):
        """ConcreteDim holds an integer."""
        dim = ConcreteDim(value=32)
        assert dim.value == 32
        assert str(dim) == "32"
    
    def test_symbolic_dim(self):
        """SymbolicDim holds a name."""
        dim = SymbolicDim(name="batch")
        assert dim.name == "batch"
        assert str(dim) == "batch"
    
    def test_dynamic_dim(self):
        """DynamicDim represents unknown dimension."""
        dim = DynamicDim()
        assert str(dim) == "?"


class TestTensorType:
    """Tests for tensor types."""
    
    def test_tensor_type_no_shape(self):
        """TensorType without shape."""
        tensor = TensorType(element_type=FloatType())
        assert tensor.element_type == FloatType()
        assert tensor.shape is None
        assert str(tensor) == "Tensor[Float]"
    
    def test_tensor_type_with_shape(self):
        """TensorType with concrete shape."""
        tensor = TensorType(
            element_type=FloatType(),
            shape=(ConcreteDim(32), ConcreteDim(10))
        )
        assert len(tensor.shape) == 2
        assert "32" in str(tensor)
        assert "10" in str(tensor)
    
    def test_tensor_type_equality(self):
        """TensorType equality."""
        t1 = TensorType(element_type=FloatType())
        t2 = TensorType(element_type=FloatType())
        t3 = TensorType(element_type=IntType())
        
        assert t1 == t2
        assert t1 != t3
    
    def test_tensor_no_free_vars(self):
        """Concrete tensor type has no free vars."""
        tensor = TensorType(element_type=FloatType())
        assert tensor.free_vars() == set()
    
    def test_tensor_with_type_var_element(self):
        """Tensor with type var element has free vars."""
        tv = TypeVar(name="T")
        tensor = TensorType(element_type=tv)
        assert tv in tensor.free_vars()


class TestFunctionType:
    """Tests for function types."""
    
    def test_function_type_no_params(self):
        """Function with no parameters."""
        fn = FunctionType(param_types=(), return_type=IntType())
        assert str(fn) == "fn() -> Int"
    
    def test_function_type_with_params(self):
        """Function with parameters."""
        fn = FunctionType(
            param_types=(FloatType(), FloatType()),
            return_type=FloatType()
        )
        assert "Float" in str(fn)
    
    def test_function_type_equality(self):
        """FunctionType equality."""
        f1 = FunctionType(param_types=(IntType(),), return_type=IntType())
        f2 = FunctionType(param_types=(IntType(),), return_type=IntType())
        f3 = FunctionType(param_types=(FloatType(),), return_type=IntType())
        
        assert f1 == f2
        assert f1 != f3
    
    def test_variadic_function(self):
        """Variadic function type."""
        fn = FunctionType(
            param_types=(FloatType(),),
            return_type=FloatType(),
            variadic=True,
            min_args=1
        )
        assert fn.variadic is True
        assert fn.min_args == 1


class TestTupleType:
    """Tests for tuple types."""
    
    def test_empty_tuple(self):
        """Empty tuple type."""
        tup = TupleType(element_types=())
        assert str(tup) == "()"
    
    def test_tuple_with_elements(self):
        """Tuple with elements."""
        tup = TupleType(element_types=(IntType(), FloatType(), BoolType()))
        assert len(tup.element_types) == 3
    
    def test_tuple_equality(self):
        """TupleType equality."""
        t1 = TupleType(element_types=(IntType(), FloatType()))
        t2 = TupleType(element_types=(IntType(), FloatType()))
        t3 = TupleType(element_types=(FloatType(), IntType()))
        
        assert t1 == t2
        assert t1 != t3


class TestStructType:
    """Tests for struct types."""
    
    def test_struct_type(self):
        """Create a struct type."""
        struct = StructType(
            name="Point",
            fields=(("x", FloatType()), ("y", FloatType()))
        )
        assert struct.name == "Point"
        assert str(struct) == "Point"
    
    def test_struct_get_field(self):
        """Get field type from struct."""
        struct = StructType(
            name="Point",
            fields=(("x", FloatType()), ("y", IntType()))
        )
        assert struct.get_field("x") == FloatType()
        assert struct.get_field("y") == IntType()
        assert struct.get_field("z") is None
    
    def test_struct_with_type_params(self):
        """Struct with type parameters."""
        struct = StructType(
            name="Box",
            fields=(("value", TypeVar("T")),),
            type_params=(IntType(),)
        )
        assert "Box[Int]" in str(struct)


class TestGenericType:
    """Tests for generic types."""
    
    def test_generic_type(self):
        """Create a generic type."""
        gen = GenericType(name="List", type_args=(IntType(),))
        assert str(gen) == "List[Int]"
    
    def test_generic_equality(self):
        """GenericType equality."""
        g1 = GenericType(name="List", type_args=(IntType(),))
        g2 = GenericType(name="List", type_args=(IntType(),))
        g3 = GenericType(name="List", type_args=(FloatType(),))
        
        assert g1 == g2
        assert g1 != g3


class TestSpecialTypes:
    """Tests for special types (Any, Never)."""
    
    def test_any_type(self):
        """AnyType represents any type."""
        any_t = AnyType()
        assert str(any_t) == "Any"
    
    def test_never_type(self):
        """NeverType represents the bottom type."""
        never = NeverType()
        assert str(never) == "Never"


class TestSubstitution:
    """Tests for type substitution."""
    
    def test_empty_substitution(self):
        """Empty substitution doesn't change types."""
        subst = Substitution()
        assert apply_substitution(FloatType(), subst) == FloatType()
    
    def test_substitute_type_var(self):
        """Substitution replaces type variables."""
        tv = TypeVar(name="T")
        subst = Substitution({tv: IntType()})
        
        assert apply_substitution(tv, subst) == IntType()
    
    def test_substitute_in_function(self):
        """Substitution applies to function types."""
        tv = TypeVar(name="T")
        fn = FunctionType(param_types=(tv,), return_type=tv)
        
        subst = Substitution({tv: FloatType()})
        result = apply_substitution(fn, subst)
        
        assert result.param_types[0] == FloatType()
        assert result.return_type == FloatType()
    
    def test_substitute_in_tensor(self):
        """Substitution applies to tensor element types."""
        tv = TypeVar(name="T")
        tensor = TensorType(element_type=tv)
        
        subst = Substitution({tv: IntType()})
        result = apply_substitution(tensor, subst)
        
        assert result.element_type == IntType()
    
    def test_compose_substitutions(self):
        """Compose two substitutions."""
        t1 = TypeVar(name="T1")
        t2 = TypeVar(name="T2")
        
        s1 = Substitution({t1: t2})
        s2 = Substitution({t2: IntType()})
        
        composed = s2.compose(s1)
        
        # After composition, T1 should map to Int (through T2)
        assert apply_substitution(t1, composed) == IntType()


class TestOccursCheck:
    """Tests for occurs check."""
    
    def test_occurs_in_self(self):
        """Type var occurs in itself."""
        tv = TypeVar(name="T")
        assert occurs_check(tv, tv) is True
    
    def test_not_occurs_in_primitive(self):
        """Type var doesn't occur in primitive."""
        tv = TypeVar(name="T")
        assert occurs_check(tv, FloatType()) is False
    
    def test_occurs_in_function(self):
        """Type var occurs in function parameter."""
        tv = TypeVar(name="T")
        fn = FunctionType(param_types=(tv,), return_type=IntType())
        assert occurs_check(tv, fn) is True
    
    def test_occurs_in_tensor(self):
        """Type var occurs in tensor element type."""
        tv = TypeVar(name="T")
        tensor = TensorType(element_type=tv)
        assert occurs_check(tv, tensor) is True


class TestUnification:
    """Tests for type unification."""
    
    def test_unify_same_types(self):
        """Same types unify with empty substitution."""
        subst = unify(FloatType(), FloatType())
        assert len(subst) == 0
    
    def test_unify_type_var_with_concrete(self):
        """Type var unifies with concrete type."""
        tv = TypeVar(name="T")
        subst = unify(tv, IntType())
        
        assert apply_substitution(tv, subst) == IntType()
    
    def test_unify_concrete_with_type_var(self):
        """Concrete type unifies with type var."""
        tv = TypeVar(name="T")
        subst = unify(FloatType(), tv)
        
        assert apply_substitution(tv, subst) == FloatType()
    
    def test_unify_type_vars(self):
        """Two type vars unify."""
        t1 = TypeVar(name="T1")
        t2 = TypeVar(name="T2")
        subst = unify(t1, t2)
        
        # One should map to the other
        assert len(subst) == 1
    
    def test_unify_tensors(self):
        """Tensor types unify."""
        t1 = TensorType(element_type=FloatType())
        t2 = TensorType(element_type=FloatType())
        subst = unify(t1, t2)
        
        assert len(subst) == 0
    
    def test_unify_tensor_with_type_var_element(self):
        """Tensor with type var element unifies."""
        tv = TypeVar(name="T")
        t1 = TensorType(element_type=tv)
        t2 = TensorType(element_type=FloatType())
        
        subst = unify(t1, t2)
        assert apply_substitution(tv, subst) == FloatType()
    
    def test_unify_with_any(self):
        """AnyType unifies with anything."""
        subst = unify(AnyType(), FloatType())
        assert len(subst) == 0
        
        subst = unify(IntType(), AnyType())
        assert len(subst) == 0
    
    def test_unify_functions(self):
        """Function types unify."""
        tv = TypeVar(name="T")
        f1 = FunctionType(param_types=(tv,), return_type=tv)
        f2 = FunctionType(param_types=(IntType(),), return_type=IntType())
        
        subst = unify(f1, f2)
        assert apply_substitution(tv, subst) == IntType()
    
    def test_unify_incompatible_primitives_fails(self):
        """Incompatible primitive types fail to unify."""
        with pytest.raises(UnificationError):
            unify(FloatType(), IntType())
    
    def test_unify_incompatible_function_arity_fails(self):
        """Functions with different arity fail to unify."""
        f1 = FunctionType(param_types=(IntType(),), return_type=IntType())
        f2 = FunctionType(param_types=(IntType(), IntType()), return_type=IntType())
        
        with pytest.raises(UnificationError):
            unify(f1, f2)
    
    def test_unify_infinite_type_fails(self):
        """Unification that creates infinite type fails."""
        tv = TypeVar(name="T")
        # T = List[T] would be infinite
        with pytest.raises(UnificationError):
            unify(tv, GenericType(name="List", type_args=(tv,)))


class TestTypeHashing:
    """Tests that types are properly hashable for use in sets/dicts."""
    
    def test_primitive_types_hashable(self):
        """Primitive types can be in sets."""
        types = {IntType(), FloatType(), BoolType(), StringType(), UnitType()}
        assert len(types) == 5
    
    def test_tensor_types_hashable(self):
        """Tensor types can be in sets."""
        t1 = TensorType(element_type=FloatType())
        t2 = TensorType(element_type=FloatType())
        t3 = TensorType(element_type=IntType())
        
        types = {t1, t2, t3}
        assert len(types) == 2  # t1 and t2 are equal
    
    def test_function_types_hashable(self):
        """Function types can be in sets."""
        f1 = FunctionType(param_types=(), return_type=IntType())
        f2 = FunctionType(param_types=(), return_type=IntType())
        
        types = {f1, f2}
        assert len(types) == 1
    
    def test_types_as_dict_keys(self):
        """Types can be used as dict keys."""
        type_to_name = {
            IntType(): "int",
            FloatType(): "float",
            BoolType(): "bool"
        }
        
        assert type_to_name[IntType()] == "int"
        assert type_to_name[FloatType()] == "float"
