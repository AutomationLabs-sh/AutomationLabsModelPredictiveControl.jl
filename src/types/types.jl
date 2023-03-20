# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

# evaluate ModelPredictiveControl
"""
	AbstractReferences
An abstract type that should be subtyped for references extensions.
"""
abstract type AbstractReferences end

"""
    ReferencesStateInput
State and input references for model predictive control.

** Fields **
* `x`: state references.
* `u`: input references.
"""
struct ReferencesStateInput <: AbstractReferences
    x::Matrix
    u::Matrix
end
#ReferencesStateInputs --> dataframe

### Weighting parameters ###
"""
	AbstractWeights
An abstract type that should be subtyped for weights coefficients extensions.
"""
abstract type AbstractWeights end

"""
    WeightsCoefficient
State, input and input rate weighting coefficient for the cost function.

** Fields **
* `Q`: state weight coefficient.
* `R`: input weight coefficient.
* `S`: input rate weight coefficient.
"""
mutable struct WeightsCoefficient <: AbstractWeights
    Q::Matrix
    R::Matrix
    S::Matrix
end

### terminal ingredients ###
"""
	AbstractTerminal
An abstract type that should be subtyped for terminal extensions.
"""
abstract type AbstractTerminal end

"""
    TerminalIngredients1
Model predictive control without terminal ingredients.

** Fields **
* `set`: bool if terminal ingredients are needed (0 or 1).
* `P`: terminal matrix weight coefficient.
"""
#mutable struct TerminalIngredients1 <: AbstractTerminal
#    set::Bool
#    P::Matrix
#end

#const TerminalIngredient = TerminalIngredients1

"""
    TerminalIngredients2
Model predictive control with terminal ingredients.

** Fields **
* `set`: bool if teerminel ingretients are needed (0 or 1).
* `P`: terminal matrix weight coefficient.
* `Xf`: terminal constraints.
"""
#mutable struct TerminalIngredients2 <: AbstractTerminal
#    set::Bool
#    P::Matrix    
#    Xf#::LazySets #type of Hyperrectangle lazySets
#end

mutable struct TerminalIngredient <: AbstractTerminal
    Xf::String
    P::Matrix
end


"""
	AbstractController
An abstract type that should be subtyped for controllers extensions.
"""
abstract type AbstractController end

"""
    ModelPredictiveControlTuning
Model predictive control tuning implementation according to parameters and references.

** Fields **
* `modeler`: model predictive control implementation acocrding to JuMP.
* `reference`: model predictive control references.
* `horizon`: model predictive control horizon.
* `weights`: model predictive control weighting coefficients.
* `terminal_ingredient`: model predictive control terminal ingredients.
* `sample_time`: model predictive control sample time.
* `max_time`: model predictive control maximum time computation.
"""
struct ModelPredictiveControlTuning <: AbstractController
    modeler::Any#::JuMP
    reference::ReferencesStateInput
    horizon::Int
    weights::WeightsCoefficient
    terminal_ingredient::TerminalIngredient
    sample_time::Float64
    max_time::Int
end

"""
    ModelPredictiveControlResults
Model predictive control results after computation.

** Fields **
* `x`: state computed.
* `e_x`: state deviation computed.
* `u`: input computed.
* `e_u`: input deviation computed.
"""
mutable struct ModelPredictiveControlResults <: AbstractController
    x::Matrix{Float64}
    e_x::Matrix{Float64}
    u::Matrix{Float64}
    e_u::Matrix{Float64}
end

"""
    ModelPredictiveControlController
Model predictive control main struct parameters. The controller as all the necessary before optimization.

** Fields **
* `system`: mathematical system type of the dynamical system (f and constraints).
* `tuning`: model predictive control implementation according to reference.
* `initialization`: initialization of the model predictive control before computation.
* `inputs_command`: model predictive control input after computation, which are sent to dynamical system.
"""
mutable struct ModelPredictiveControlController <: AbstractController
    system::Any#::MathematicalSystems
    tuning::ModelPredictiveControlTuning
    initialization::Vector{Float64}
    computation_results::ModelPredictiveControlResults
end

"""
    AbstractSolvers
An abstract type that should be subtyped for solver extensions.
"""
abstract type AbstractSolvers end

"""
    osqp
Linear quadratic solver.
"""
struct osqp_solver_def <: AbstractSolvers end

"""
    highs
Linear quadratic solver.
"""
struct highs_solver_def <: AbstractSolvers end

"""
    ipopt
Linear and non-linear quadratic solver.
"""
struct ipopt_solver_def <: AbstractSolvers end

"""
    SCIP
Linear and integer and non-linear quadratic solver.
"""
struct scip_solver_def <: AbstractSolvers end

"""
    auto
Automatique selection of the solver according to the method.
"""
struct auto_solver_def <: AbstractSolvers end


"""
    AbstractImplementaiton
An abstract type that should be subtyped for activation function extensions, mainly for relu.
"""
abstract type AbstractImplementation end

"""
    MixedIntegerLinearProgramming
Milp tool for modeler implementation of Fnn and Resnet with relu activation function.
"""
struct MixedIntegerLinearProgramming <: AbstractImplementation end

"""
    NonLinearProgramming
Nl tool for modeler implementation of Fnn and Resnet abd DenseNet with any activation function.
"""
struct NonLinearProgramming <: AbstractImplementation end

"""
    LinearProgramming
Linear tool for modeler implementation of neural networks.
"""
struct LinearProgramming <: AbstractImplementation end

"""
    FuzzyProgramming
Linear fuzzy tool for modeler implementation of neural networks.
"""
struct FuzzyProgramming <: AbstractImplementation end

"""
    IMPLEMENTATION_PROGRAMMING_LIST = (
Constant tuple programming list.
"""
const IMPLEMENTATION_PROGRAMMING_LIST = (
    linear = LinearProgramming(),
    non_linear = NonLinearProgramming(),
    mixed_linear = MixedIntegerLinearProgramming(),
    fuzzy_linear = FuzzyProgramming(),
)

#=

"""
    TakagiSugeno
Fuzzy tool for modeler implementation of neural networks.
"""
struct   TakagiSugeno <: AbstractImplementation 
        nbr_models::Int
end

"""
	AbstractModel
An abstract type that should be subtyped for model extensions (linear àr non linear)
"""
abstract type AbstractModel end

"""
    LinearModel
Model predictive control tuning implementation according to parameters and references.

** Fields **
* `A`: state matrix.
* `B`: input matrix.
"""
struct LinearModel <: AbstractModel
    A::Matrix
	B::Matrix
end

"""
    NonLinearModel
Model predictive control tuning implementation according to parameters and references.

** Fields **
* `f`: the non linear model.
* `nbr_state`: the state number.
* `nbr_input`: the input number
"""
struct NonLinearModel <: AbstractModel
    f::Function
	nbr_state::Int
    nbr_input::Int
end

=#
