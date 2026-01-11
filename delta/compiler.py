"""
Main Delta compiler orchestration.

Coordinates all compiler passes from source to executable:
1. Frontend: Parse, desugar
2. Type system: Inference, checking
3. Analysis: Effects, roles, mode
4. IR: Lower to SIR
5. Optimization: Relaxation, constraint compilation, specialization
6. Backend: FX lowering, code generation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch
from torch.fx import GraphModule

from delta.source import SourceFile
from delta.frontend.parser import Parser
from delta.frontend.desugar import desugar_module
from delta.frontend.ast import Module, ParamDecl, Call, Identifier, Literal
from delta.types.inference import TypeInference, TypeEnvironment
from delta.types.checker import TypeChecker
from delta.ir.sir import SIRModule
from delta.ir.sir_builder import SIRBuilder, build_sir
from delta.ir.sir_printer import format_sir
from delta.passes.name_resolution import resolve_names
from delta.passes.role_assignment import assign_roles
from delta.passes.effect_inference import infer_effects
from delta.passes.mode_check import check_mode
from delta.passes.relaxation import relax_sir, RelaxationConfig
from delta.passes.constraint_compile import compile_constraints
from delta.passes.mode_specialize import specialize_mode
from delta.passes.gradient_analysis import analyze_gradients
from delta.passes.dead_gradient_elim import eliminate_dead_gradients
from delta.backend.fx_lowering import lower_to_fx
from delta.backend.fx_optimize import optimize_fx
from delta.backend.codegen import generate_python, CodeGenConfig
from delta.runtime.executor import Executor, ExecutionContext
from delta.runtime.cache import CompilationCache, CacheKey, get_global_cache
from delta.errors import ErrorCollector, DeltaError


@dataclass
class CompileOptions:
    """Options for compilation."""
    mode: str = "train"  # train, infer, analyze
    optimize: bool = True
    use_cache: bool = True
    target: str = "fx"  # fx, python, sir
    debug: bool = False
    relaxation_temperature: float = 1.0
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class CompileResult:
    """Result of compilation."""
    success: bool
    errors: list[DeltaError] = field(default_factory=list)
    warnings: list[DeltaError] = field(default_factory=list)
    
    # Intermediate representations
    ast: Optional[Module] = None
    sir: Optional[SIRModule] = None
    
    # Compiled outputs
    graph_modules: Optional[Dict[str, GraphModule]] = None
    executor: Optional[Executor] = None
    python_code: Optional[str] = None
    
    # Metadata
    source_hash: Optional[str] = None
    compile_time_ms: float = 0.0
    param_initializers: Optional[Dict[str, Any]] = None  # Parameter initializer expressions


class Compiler:
    """
    The Delta compiler.
    
    Compiles Delta source code to executable PyTorch graphs.
    
    Usage:
        compiler = Compiler()
        result = compiler.compile_file("model.delta")
        if result.success:
            executor = result.executor
            output = executor("forward", x=input_tensor)
    """
    
    def __init__(self, options: Optional[CompileOptions] = None) -> None:
        self.options = options or CompileOptions()
        self.cache = get_global_cache() if self.options.use_cache else None
        self.errors = ErrorCollector()
    
    def compile(self, source: Union[str, SourceFile, Path]) -> CompileResult:
        """
        Compile Delta source code.
        
        Args:
            source: Source code as string, SourceFile, or path
        
        Returns:
            CompileResult with compiled artifacts
        """
        import time
        start_time = time.time()
        
        # Create SourceFile
        if isinstance(source, Path):
            source_file = SourceFile.from_path(source)
        elif isinstance(source, str):
            source_file = SourceFile.from_string(source)
        else:
            source_file = source
        
        result = CompileResult(success=False, source_hash=source_file.hash)
        
        try:
            # Frontend
            ast = self._parse(source_file)
            if self.errors.has_errors():
                result.errors = list(self.errors.errors)
                return result
            result.ast = ast
            
            # Type system
            type_env, type_errors = self._type_check(ast)
            if type_errors:
                result.errors.extend(type_errors)
                if any(e.severity.name == "ERROR" for e in type_errors):
                    return result
            
            # Lower to SIR
            sir = self._lower_to_sir(ast, type_env)
            result.sir = sir
            
            # Extract parameter initializers for runtime initialization
            result.param_initializers = extract_param_initializers(ast)
            
            # Optimization passes
            sir = self._optimize_sir(sir)
            
            # Backend
            if self.options.target == "fx":
                graph_modules = self._lower_to_fx(sir)
                result.graph_modules = graph_modules
                
                # Create executor
                parameters = {
                    name: torch.nn.Parameter(torch.randn(1))
                    for name in sir.params
                }
                
                device = torch.device(self.options.device)
                context = ExecutionContext(
                    mode=self.options.mode,
                    device=device
                )
                
                result.executor = Executor(graph_modules, parameters, context)
            
            elif self.options.target == "python":
                result.python_code = generate_python(sir)
            
            elif self.options.target == "sir":
                # Just return SIR, already set
                pass
            
            result.success = True
            
        except DeltaError as e:
            result.errors.append(e)
        except Exception as e:
            result.errors.append(DeltaError(
                message=f"Internal compiler error: {str(e)}",
            ))
            if self.options.debug:
                raise
        
        result.compile_time_ms = (time.time() - start_time) * 1000
        return result
    
    def compile_file(self, path: Union[str, Path]) -> CompileResult:
        """Compile a Delta source file."""
        return self.compile(Path(path))
    
    def compile_string(self, code: str, name: str = "<string>") -> CompileResult:
        """Compile Delta source code from a string."""
        source = SourceFile.from_string(code, name)
        return self.compile(source)
    
    def _parse(self, source: SourceFile) -> Module:
        """Parse source to AST."""
        parser = Parser(source)
        ast = parser.parse_module()
        
        # Collect parser errors
        for error in parser.errors.errors:
            self.errors.add(error)
        
        # Desugar
        ast = desugar_module(ast)
        
        return ast
    
    def _type_check(self, ast: Module) -> tuple[TypeEnvironment, list[DeltaError]]:
        """Run type inference and checking."""
        # Name resolution
        resolutions, name_errors = resolve_names(ast)
        errors: list[DeltaError] = list(name_errors.errors)
        
        # Role assignment (MUST come before type inference per spec Section 4)
        roles = assign_roles(ast)
        
        # Type inference
        inference = TypeInference()
        type_env = inference.infer_module(ast)
        errors.extend(inference.errors)
        
        # Type checking
        checker = TypeChecker(type_env, inference)
        check_errors = checker.check_module(ast)
        errors.extend(check_errors)
        
        # Effect inference
        effects = infer_effects(ast)
        
        # Mode checking
        mode_errors = check_mode(ast, effects, roles)
        errors.extend(mode_errors.errors)
        
        return type_env, errors
    
    def _lower_to_sir(self, ast: Module, type_env: TypeEnvironment) -> SIRModule:
        """Lower AST to Semantic IR."""
        inference = TypeInference()
        inference.infer_module(ast, type_env)
        return build_sir(ast, type_env, inference)
    
    def _optimize_sir(self, sir: SIRModule) -> SIRModule:
        """Run optimization passes on SIR."""
        if not self.options.optimize:
            # Still run these even if optimize=False because they are semantic
            # but maybe with different defaults.
            pass
            
        # Relaxation
        config = RelaxationConfig(
            default_temperature=self.options.relaxation_temperature
        )
        sir = relax_sir(sir, config)
        
        # Constraint compilation
        sir, total_penalty = compile_constraints(sir)
        
        # If we have a total penalty, ensure it's exposed in a 'forward' function
        # for training.
        if self.options.mode == "train":
            # Identify all observations mentioned in the penalty to build signature
            from delta.ir.sir import walk_sir, ObsRef, SIRFunction, SIRBlock, SIRProperty
            from delta.types.types import TensorType, FloatType
            
            obs_names = set()
            for node in walk_sir(total_penalty):
                if isinstance(node, ObsRef):
                    obs_names.add(node.name)
            
            # Synthesize a forward function that returns the loss
            sorted_obs = sorted(list(obs_names))
            params = [(name, SIRProperty(dtype=TensorType(FloatType()))) for name in sorted_obs]
            
            # We don't need nodes in the block if total_penalty is recursive
            from delta.types.effects import EffectSet
            forward_func = SIRFunction(
                name="forward",
                params=params,
                body=SIRBlock(nodes=[], result=total_penalty),
                return_type=FloatType(),
                effects=EffectSet.pure()
            )
            sir.functions["forward"] = forward_func
        
        # Mode specialization
        sir = specialize_mode(sir, self.options.mode)
        
        if not self.options.optimize:
            return sir
            
        # Gradient analysis and dead gradient elimination
        gradient_info, _ = analyze_gradients(sir)
        sir, _ = eliminate_dead_gradients(sir, gradient_info)
        
        return sir
    
    def _lower_to_fx(self, sir: SIRModule) -> Dict[str, GraphModule]:
        """Lower SIR to PyTorch FX."""
        graph_modules = lower_to_fx(sir)
        
        # Optimize FX graphs
        if self.options.optimize:
            for name, gm in graph_modules.items():
                optimized, _ = optimize_fx(gm)
                graph_modules[name] = optimized
        
        return graph_modules


def extract_param_initializers(ast: Optional[Module]) -> Dict[str, Any]:
    """
    Extract parameter initializer metadata from AST.
    
    Returns a dictionary mapping parameter names to their initializer expressions.
    This is used for parameter initialization in the runtime layer.
    """
    if not ast:
        return {}
    
    param_initializers = {}
    for item in ast.items:
        if isinstance(item, ParamDecl) and item.initializer:
            param_initializers[item.name] = item.initializer
    return param_initializers


def compile(source: Union[str, Path], **options) -> CompileResult:
    """
    Convenience function to compile Delta source.
    
    Args:
        source: Source code string or file path
        **options: Compilation options
    
    Returns:
        CompileResult
    """
    compile_options = CompileOptions(**options)
    compiler = Compiler(compile_options)
    
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        return compiler.compile_file(source)
    else:
        return compiler.compile_string(source)


def compile_and_run(
    source: Union[str, Path],
    inputs: Dict[str, torch.Tensor],
    **options
) -> torch.Tensor:
    """
    Compile and immediately execute Delta source.
    
    Args:
        source: Source code
        inputs: Input tensors
        **options: Compilation options
    
    Returns:
        Output tensor
    """
    result = compile(source, **options)
    
    if not result.success:
        raise RuntimeError(f"Compilation failed: {result.errors}")
    
    if result.executor is None:
        raise RuntimeError("No executor available")
    
    return result.executor.forward(**inputs)
