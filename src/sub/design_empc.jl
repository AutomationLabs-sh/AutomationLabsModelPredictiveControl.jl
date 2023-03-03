# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
    EconomicModelPredictiveControlTuning
Model predictive control tuning implementation according to parameters and references.

** Fields **
* `modeler`: model predictive control implementation acocrding to JuMP.
* `reference`: model predictive control references.
* `horizon`: model predictive control horizon.
* `weights`: model predictive control weighting coefficients.
* `terminal_ingredients`: model predictive control terminal ingredients.
* `sample_time`: model predictive control sample time.
* `max_time`: model predictive control maximum time computation.
"""
struct EconomicModelPredictiveControlTuning <: AbstractController
    modeler::Any#::JuMP
    horizon::Int
    weights::WeightsCoefficient
    sample_time::Float64
    max_time::Int
end

"""
    EconomicModelPredictiveControlResults
Model predictive control results after computation.

** Fields **
* `x`: state computed.
* `u`: input computed.
"""
mutable struct EconomicModelPredictiveControlResults <: AbstractController
    x::Matrix{Float64}
    u::Matrix{Float64}
end

"""
    EconomicModelPredictiveControlController
Model predictive control main struct parameters. The controller as all the necessary before optimization.

** Fields **
* `system`: mathematical system type of the dynamical system (f and constraints).
* `tuning`: model predictive control implementation according to reference.
* `initialization`: initialization of the model predictive control before computation.
* `inputs_command`: model predictive control input after computation, which are sent to dynamical system.
"""
mutable struct EconomicModelPredictiveControlController <: AbstractController
    system::Any#::MathematicalSystems
    tuning::EconomicModelPredictiveControlTuning
    initialization::Vector{Float64}
    computation_results::EconomicModelPredictiveControlResults
end


"""
    _economic_model_predictive_control_design
Function that tunes a economic model predictive control.

** Required fields **
* `system`: mathematical system type.
* `model_mlj`: AutomationLabsIdentification type model, ResNet, Fnn, ....
* `horizon`: horizon length of the model predictive control.
* `method`: implementation method (linear, non linear, mixed integer linear, fuzzy rules)
* `sample_time`: time sample of the model predictive control.

** Optional fields **
* `Q`: state weighting parameters.
* `R`: input weighting parameters.
* `S`: input rate weighting paramters.
* `terminal_ingredients = false`: terminal ingredients of economic model predictive control (not implemnted yet)
* `Computation`: modeler, optimizer and sample time.
"""
function _economic_model_predictive_control_design(
    system::Union{
        MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
        MathematicalSystems.ConstrainedLinearControlDiscreteSystem,
    },
    model_mlj::Union{
        AutomationLabsIdentification.linear,
        AutomationLabsIdentification.Fnn,
        AutomationLabsIdentification.Icnn,
        AutomationLabsIdentification.ResNet,
        AutomationLabsIdentification.DenseNet,
        AutomationLabsIdentification.Rbf,
        AutomationLabsIdentification.PolyNet,
        AutomationLabsIdentification.NeuralNetODE_type1,
        AutomationLabsIdentification.NeuralNetODE_type2,
        MLJMultivariateStatsInterface.MultitargetLinearRegressor,
    },
    horizon::Int,
    method::AbstractImplementation,
    sample_time::Int,
    linearization_point::ReferencesStateInput;
    Q::Float64 = 0.1,
    R::Float64 = 100.0,
    S::Float64 = 0.0,
    max_time::Float64 = 30.0,
    terminal_ingredients::Bool = false,
    solver::AbstractSolvers = auto_solver_def(),
) #rather than the name of the solver

    # create the weight of the controller
    weights = _create_weights_coefficients(system, Q, R, S)

    # modeller implementation of model predictive control
    model_empc = _economic_model_predictive_control_modeler_implementation(
        method,
        model_mlj,
        system,
        horizon,
        linearization_point,
        solver,
    )

    # implementetation of terminal ingredients
    # not yet implemented with empc

    # add the quadratic cost function to the modeler model
    model_empc = _create_economic_cost_function(model_empc, weights)

    ### struct design: MPC tuning ###
    tuning = EconomicModelPredictiveControlTuning(
        model_empc,
        horizon,
        weights,
        sample_time,
        max_time,
    )


    ### struct design: set the controller ###
    initialization, computation_results =
        _memory_allocation_initialization_results_empc(system, horizon)

    return empc_controller = EconomicModelPredictiveControlController(
        system,
        tuning,
        initialization,
        computation_results,
    )
end


"""
    _create_economic_cost_function
Function that create the quadratic cost of the model predictive control.

** Required fields **
* `model_empc`: the JuMP struct.
* `weights`: the weighing coefficient struct.
"""
function _create_economic_cost_function(model_empc::JuMP.Model, weights::WeightsCoefficient)

    #get variable from model JuMP model_mpc
    u = model_empc[:u]
    x = model_empc[:x]
    horizon = size(u, 2)

    #add the delta rate constraints if needed
    if weights.S[1, 1] != 0.0
        #add delta_u variable
        JuMP.@variables(model_empc, begin
            delta_u[1:size(u, 1), 1:horizon]
        end)

        #Deviation inputs delta rate constraint
        for i = 1:horizon-1
            JuMP.@constraint(model_empc, delta_u[:, i] .== u[:, i] .- u[:, i+1])
        end

    end

    #Add the quadratic cost function according to weight parameters
    if weights.R[1, 1] != 0.0 && weights.S[1, 1] != 0.0
        JuMP.@objective(
            model_empc,
            Min,
            x[:, end]' * weights.Q * x[:, end] +
            sum(
                x[:, i]' * weights.Q * x[:, i] + u[:, i]' * weights.R * u[:, i] for
                i = 1:horizon
            ) +
            sum(delta_u[:, i]' * weights.S * delta_u[:, i] for i = 1:horizon-1)
        )

    elseif weights.R[1, 1] != 0.0
        JuMP.@objective(
            model_empc,
            Min,
            x[:, end]' * weights.Q * x[:, end] + sum(
                x[:, i]' * weights.Q * x[:, i] + u[:, i]' * weights.R * u[:, i] for
                i = 1:horizon
            )
        )

    else
        JuMP.@objective(
            model_empc,
            Min,
            sum(x[:, i]' * weights.Q * x[:, i] for i = 1:horizon+1)
        )

    end

    return model_empc

end

# Memory allocation according to mathematical systems
function _memory_allocation_initialization_results_empc(
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon,
)

    initialization = Vector{Float64}(undef, system.statedim)
    computation_results = EconomicModelPredictiveControlResults(
        Array{Float64}(undef, system.statedim, horizon + 1),
        Array{Float64}(undef, system.inputdim, horizon),
    )

    return initialization, computation_results
end

function _memory_allocation_initialization_results_empc(
    system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem,
    horizon,
)

    initialization = Vector{Float64}(undef, size(system.A, 1))
    computation_results = EconomicModelPredictiveControlResults(
        Array{Float64}(undef, size(system.A, 1), horizon + 1),
        Array{Float64}(undef, size(system.B, 2), horizon),
    )

    return initialization, computation_results
end
