# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

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
Lienar tool for modeler implementation of neural networks.
"""
struct LinearProgramming <: AbstractImplementation end

"""
    TakagiSugeno
Fuzzy tool for modeler implementation of neural networks.
"""
struct   TakagiSugeno <: AbstractImplementation 
        nbr_models::Int
end

const IMPLEMENTATION_PROGRAMMING_LIST = (
    linear = LinearProgramming(),
    non_linear = NonLinearProgramming(),
    mixed_linear = MixedIntegerLinearProgramming(),
)

"""
    _model_predictive_control_design
Function that tunes a model predictive control.

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
* `terminal_ingredients = false`: terminal ingredients of model predictive control
* `Computation`: modeler, optimizer and sample time.
"""
function _model_predictive_control_design(system::Union{MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem, 
                                                    MathematicalSystems.ConstrainedLinearControlDiscreteSystem},
                                      model_mlj::Union{AutomationLabsIdentification.linear, 
                                                   AutomationLabsIdentification.Fnn, 
                                                   AutomationLabsIdentification.Icnn,
                                                   AutomationLabsIdentification.ResNet, 
                                                   AutomationLabsIdentification.DenseNet, 
                                                   AutomationLabsIdentification.Rbf, 
                                                   AutomationLabsIdentification.PolyNet, 
                                                   AutomationLabsIdentification.NeuralNetODE_type1, 
                                                   AutomationLabsIdentification.NeuralNetODE_type2, 
                                                   MLJMultivariateStatsInterface.MultitargetLinearRegressor},
                                      horizon::Int, 
                                      method::AbstractImplementation,
                                      sample_time::Int, 
                                      references::ReferencesStateInput;
                                      Q::Float64 = 100.0, 
                                      R::Float64 = 0.1,
                                      S::Float64 = 0.0,
                                      max_time::Float64 = 30.0,
                                      terminal_ingredients::Bool = false, 
                                      solver::AbstractSolvers = auto_solver_def()) #rather than the name of the solver

    # create the weight of the controller
    weights = _create_weights_coefficients( system, 
                                           Q, 
                                           R, 
                                           S)
    
    # modeller implementation of model predictive control
    model_mpc = _model_predictive_control_modeler_implementation( method,
                                                             model_mlj,
                                                             system,
                                                             horizon,
                                                             references, 
                                                             solver)

    # implementetation of terminal ingredients
    terminal, model_mpc = _create_terminal_ingredients( model_mpc, 
                                                       terminal_ingredients,
                                                       system, 
                                                       references, 
                                                       weights ) 

    # add the quadratic cost function to the modeler model
    model_mpc = _create_quadratic_cost_function( model_mpc, 
                                                weights, 
                                                terminal)

    ### struct design: MPC tuning ###
    tuning =  ModelPredictiveControlTuning( model_mpc,
                                            references,
                                            horizon,
                                            weights,
                                            terminal,
                                            sample_time,
                                            max_time)
    
    ### struct design: set the controller ###
    initialization, computation_results = _memory_allocation_initialization_results_mpc(system, horizon)


    return mpc_controller = ModelPredictiveControlController(   system,
                                                                tuning,
                                                                initialization,
                                                                computation_results)
end

###################################################
# Sub functions from _model_predictive_control_design #
###################################################
"""
    _create_weights_coefficients
Function that create the weighting struct for model predictive control.

** Required fields **
* `system`: the mathematical system from the dynamical system.
* `Q`: the states weighting parameter.
* `R`: the inputs weighting parameter.
* `S`: the rate inputs weighting parameter.
"""
function _create_weights_coefficients( system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem, 
                                      Q::Float64, 
                                      R::Float64, 
                                      S::Float64)

    #create the matrices
    QM = Q * LinearAlgebra.Matrix(LinearAlgebra.I, system.statedim, system.statedim)
    RM = R * LinearAlgebra.Matrix(LinearAlgebra.I, system.inputdim, system.inputdim)
    SM = S * LinearAlgebra.Matrix(LinearAlgebra.I, system.inputdim, system.inputdim)

    return WeightsCoefficient(QM, RM, SM)
end

function _create_weights_coefficients( system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem, 
                                        Q::Float64, 
                                        R::Float64, 
                                        S::Float64)

    #create the matrices
    QM = Q * LinearAlgebra.Matrix(LinearAlgebra.I, size(system.A, 1), size(system.A, 1))
    RM = R * LinearAlgebra.Matrix(LinearAlgebra.I, size(system.B, 2), size(system.B, 2))
    SM = S * LinearAlgebra.Matrix(LinearAlgebra.I, size(system.B, 2), size(system.B, 2))

    return WeightsCoefficient(QM, RM, SM)
end


"""
    _create_terminal_ingredients
Function that create the terminal ingredient for model predictive control. The terminal ingredients are the terminal weight and terminal constraints. 
The terminal constraints could be optional. 

** Required fields **
* `model_mpc`: the JuMP from the modeler design.
* `ingredients`: a boolean to set the terminal constraints.
* `system`: the dynamical system mathemacal systems.
* `references`: the states and inputs references.
* `weights`: the weighting coefficients.
"""
function _create_terminal_ingredients( model_mpc::JuMP.Model,
                                      ingredients::Bool,
                                      system::Union{MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem, 
                                                    MathematicalSystems.ConstrainedLinearControlDiscreteSystem},
                                      reference_in::ReferencesStateInput, 
                                      weights::WeightsCoefficient)

    #get the last references to compute the terminal ingredient
    state_reference = reference_in.x[:, end]
    input_reference = reference_in.u[:, end]

    # evaluate if the user choses the terminal constraints
    if ingredients == true

        #Terminal ingredients from H. Chen and F. Allgower, A Quasi-Infinite Horizon Nonlinear Model Predictive
        #Control Scheme with Guaranteed Stability, Automatica 1998

        #Linearization to get A and B at references
        A_sys, B_sys = system_linearization(system, state_reference, input_reference)

        #Calculate P matrix with ControlSystems and LQR
        P =  ControlSystems.are(Discrete, A_sys, B_sys, weights.Q, weights.R)

        #Calculate Xf
        A = A_sys - B_sys*K1

    #InvariantSets.jl
    # """
#     THEORY:
#   Invariance: Region in which an autonomous system
#   satisifies the constraints for all time.
#
#   Control Invariance: Region in which there exists a controller
#   so that the system satisfies the constraints for all time.
#
#   A set 𝒪 is positive invariant if and only if 𝒪 ⊆ pre(𝒪)!
#   """

    #Set new cost function with terminal cost P
    if weights.R != 0 && weights.S != 0
        JuMP.@objective(model_mpc, Min, sum(e_x[:,i]' * weights.Q * e_x[:,i] + e_u[:,i]' * weights.R * e_u[:,i] + d_u[:,i]' * weights.S * d_[:,i] for i in 1:Horizon) + e_x[:,i]' * P * e_x[:,i] )
    
    elseif weights.R != 0
        JuMP.@objective(model_mpc, Min, sum(e_x[:,i]' * weights.Q * e_x[:,i] + e_u[:,i]' * weights.R * e_u[:,i] for i in 1:Horizon) + e_x[:,i]' * P * e_x[:,i])
    
    else
        JuMP.@objective(model_mpc, Min, sum(e_x[:,i]' * weights.Q * e_x[:,i] for i in 1:Horizon) + e_x[:,i]' * P * e_x[:,i])

    end

    #Add terminal constraint
    x = model_mpc[:x]
    for i in 1 : nbr_states
        JuMP.@constraint(model_mpc, (x_ter[i,1]  <= x[i,end]  <= x_ter[i,2] ))
    end
    
        #const TerminalIngredients = TerminalIngredients2(true, P, Xf)
        TerminalIngredients = TerminalIngredients2(true, P, Xf)

        return TerminalIngredients, model_mpc

    else
        #no terminal constraints is choosen
            
        #Linearization to get A and B at references
        A_sys, B_sys = system_linearization(system, state_reference, input_reference)

        #Calculate P matrix with ControlSystems and LQR
        P =  ControlSystems.are(ControlSystems.Discrete, A_sys, B_sys, weights.Q, weights.R)
        
        #set the struct from the package model predictive control
        #const TerminalIngredients = TerminalIngredients1(false, P)
        TerminalIngredients = TerminalIngredients1(false, P)

        return TerminalIngredients, model_mpc

    end


end

"""
    _create_quadratic_cost_function
Function that create the quadratic cost of the model predictive control.

** Required fields **
* `model_mpc`: the JuMP struct.
* `weights`: the weighing coefficient struct.
* `terminal_ingredients`: the terminal ingredients struct.
* `horizon`: the horizon parameters
"""
function _create_quadratic_cost_function(model_mpc::JuMP.Model, 
                                        weights::WeightsCoefficient, 
                                        terminal_ingredients::AbstractTerminal)

    #get variable from model JuMP model_mpc
    u   = model_mpc[:u]
    e_x = model_mpc[:e_x]
    e_u = model_mpc[:e_u]
    horizon = size(u,2)

    #add the delta rate constraints if needed
    if weights.S[1,1] != 0.0
        #add delta_u variable
        JuMP.@variables(model_mpc, begin
            delta_u[1:size(u, 1), 1:horizon]
        end)

        #Deviation inputs delta rate constraint
        for i in 1 : horizon - 1
            JuMP.@constraint(model_mpc, delta_u[:,i] .==  u[:,i] .- u[:,i+1] )
        end 

    end

    #Add the quadratic cost function according to weight parameters
    if weights.R[1,1] != 0.0 && weights.S[1,1] != 0.0
        JuMP.@objective(model_mpc, Min, e_x[:,end]' * terminal_ingredients.P * e_x[:,end] +
                                         sum(e_x[:,i]' * weights.Q * e_x[:,i] + 
                                             e_u[:,i]' * weights.R * e_u[:,i] for i in 1:horizon) +
                                         sum(delta_u[:,i]' * weights.S * delta_u[:,i] for i in 1: horizon-1))
    
    elseif weights.R[1,1] != 0.0
        JuMP.@objective(model_mpc, Min, e_x[:,end]' * terminal_ingredients.P * e_x[:,end] +
                                        sum(e_x[:,i]' * weights.Q * e_x[:,i] +
                                            e_u[:,i]' * weights.R * e_u[:,i] for i in 1:horizon))
    
    else
        JuMP.@objective(model_mpc, Min, e_x[:,end]' * terminal_ingredients.P * e_x[:,end] +
                                        sum(e_x[:,i]' * weights.Q * e_x[:,i] for i in 1:horizon))

    end

    return model_mpc

end


"""
    system_linearization
Function that linearises a system from MathematicalSystems at state and input references. 
The function uses ForwardDiff package and the jacobian function.

** Required fields **
* `system`: the mathemacital system that as in it the julia non-linear function `f`.
* `state`: references state point.
* `input`: references input point.
"""
function system_linearization( system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
                               state::Vector{Float64}, 
                               input::Vector{Float64} )
   
    #Linearization to get A and B at references values (state and input)
    VectorXU = vcat(state, input);
    JacobianMatrix = ForwardDiff.jacobian(system.f, VectorXU);
    A_sys  = JacobianMatrix[1:system.statedim,1:system.statedim];
    B_sys  = JacobianMatrix[1:system.statedim, system.statedim+1:end];

    return A_sys, B_sys

end

"""
    system_linearization
Function that linearises a system from MathematicalSystems at state and input references. 
The function uses ForwardDiff package and the jacobian function.

** Required fields **
* `system`: the mathemacital system that as in it the julia non-linear function `f`.
* `state`: references state point.
* `input`: references input point.
"""
function system_linearization( system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem,
    state::Vector{Float64}, 
    input::Vector{Float64} )

return system.A, system.B

end

#get the activation function of the neural networks
#mandatory to evaluation activation function for MILP optimisation #to do
function get_activation_function(   method::Union{AutomationLabsIdentification.Fnn, AutomationLabsIdentification.Icnn, AutomationLabsIdentification.Rbf}, 
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem )

f = system.f
return getfield(f[2][1], :σ)

end

function get_activation_function(   method::Union{AutomationLabsIdentification.ResNet, AutomationLabsIdentification.DenseNet, AutomationLabsIdentification.PolyNet}, 
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem )

f = system.f
return getfield(f[2][1].layers.layers[1], :σ)

end


# Memory allocation according to mathematical systems
function _memory_allocation_initialization_results_mpc(system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem, horizon )

    initialization = Vector{Float64}(undef, system.statedim)
    computation_results = ModelPredictiveControlResults( Array{Float64}(undef, system.statedim, horizon +1),
                                                     Array{Float64}(undef, system.statedim, horizon +1),
                                                     Array{Float64}(undef, system.inputdim, horizon ),
                                                     Array{Float64}(undef, system.inputdim, horizon) )

    return  initialization, computation_results
end 

function _memory_allocation_initialization_results_mpc(system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem, horizon )

    initialization = Vector{Float64}(undef, size(system.A, 1))
    computation_results = ModelPredictiveControlResults( Array{Float64}(undef, size(system.A, 1), horizon +1),
                                                     Array{Float64}(undef, size(system.A, 1), horizon +1),
                                                     Array{Float64}(undef, size(system.B, 2), horizon ),
                                                     Array{Float64}(undef, size(system.B, 2), horizon) )

    return initialization, computation_results
end


#######################################################
# End Sub functions from _model_predictive_control_design #
#######################################################