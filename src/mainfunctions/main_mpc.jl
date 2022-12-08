# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

#NamedTuple default parameters definition
const _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL = (
    mpc_solver = "auto",
    mpc_terminal_ingredient = false,
    mpc_Q = 100.0,
    mpc_R = 0.1,
    mpc_S = 0.0,
    mpc_max_time = 30.0,
)

"""
    proceed_controller
A function for model predictive control and economic model predictive control.

The following variables are mendatories:
* `machine_mlj`: a machine.
* `mpc_controller_type`: a string for model predictive control or economic model predictive control. 
* `mpc_lower_state_constraints`: a constraint.
* `mpc_higher_state_constraints`: a constraint.
* `mpc_lower_input_constraints`: a constraint.
* `mpc_higher_input_constraints`: a constraint.
* `mpc_horizon`: a horizon for the predictive controller.
* `mpc_sample_time`: a sample time for the predictive controller.
* `mpc_state_reference`: state reference for mpc or linearization point for empc. 
* `mpc_input_reference`: input reference for mpc or linearization point for empc.

It is possible to define optional variables for the controller.
"""
function proceed_controller(
    machine_mlj, #Type?
    mpc_controller_type::String,
    mpc_programming_type::String,
    mpc_lower_state_constraints::Vector,
    mpc_higher_state_constraints::Vector,
    mpc_lower_input_constraints::Vector,
    mpc_higher_input_constraints::Vector,
    mpc_horizon::Int,
    mpc_sample_time::Int, 
    mpc_state_reference::Vector, 
    mpc_input_reference::Vector;
    kws_...
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Get the type of the model from the machine_mlj
    model_mlj_type = _get_mlj_model_type(machine_mlj)

    # System definition
    system = _controller_system_design(
        model_mlj_type,
        machine_mlj,
        mpc_lower_state_constraints,
        mpc_higher_state_constraints,
        mpc_lower_input_constraints,
        mpc_higher_input_constraints,
    )

    # Get default parameters or user parameters
    mpc_method_optimization = ImplementationProgrammingList[Symbol(mpc_programming_type)]

    mpc_solver_choosen = get(kws, :mpc_solver, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_solver])
    mpc_solver_type = _IMPLEMENTATION_SOLVER_LIST[Symbol(mpc_solver_choosen)]

    mpc_terminal_ingredient = get(kws, :mpc_terminal_ingredient, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_terminal_ingredient])
    mpc_Q = get(kws, :mpc_Q, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_Q])
    mpc_R = get(kws, :mpc_R, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_R])
    mpc_S = get(kws, :mpc_S, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_S])
    mpc_max_time = get(kws, :mpc_max_time, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_max_time])

    ### other parameters to be defined

    # Evaluate if mpc controller is quadratic or economic
    if mpc_controller_type == "ModelPredictiveControl"

        # Get linearisation poitn references 
        mpc_references = _design_reference_mpc(mpc_state_reference, mpc_input_reference, mpc_horizon)

        # Design the model predictive control controller
        controller = ModelPredictiveControlDesign(system, 
                                      model_mlj_type,
                                      mpc_horizon, 
                                      mpc_method_optimization,
                                      mpc_sample_time, 
                                      mpc_references;
                                      Q = mpc_Q, 
                                      R = mpc_R,
                                      S = mpc_S,
                                      max_time = mpc_max_time,
                                      terminal_ingredients = mpc_terminal_ingredient, 
                                      solver = mpc_solver_type
        ) 

     return controller
    end

    # Evaluate if mpc controller is quadratic or economic
    if mpc_controller_type == "EconomicModelPredictiveControl"

        # Set references 
        mpc_linearization_point = _design_reference_mpc(mpc_state_reference, mpc_input_reference, mpc_horizon)

        # Design the economic model predictive control controller
        controller = _economic_model_predictive_control_design(system,
                                      model_mlj_type,
                                      mpc_horizon, 
                                      mpc_method_optimization,
                                      mpc_sample_time,
                                      mpc_linearization_point,
                                      Q = mpc_Q, 
                                      R = mpc_R,
                                      S = mpc_S,
                                      max_time = mpc_max_time,
                                      terminal_ingredients = mpc_terminal_ingredient, 
                                      solver = mpc_solver_type
        ) 

        return controller
    end
end

"""
    _controller_system_design
A function for design the system (model and constrants) with MathematicalSystems for model predictive control and economic model predictive control.

"""
function _controller_system_design(
    model_mlj::Union{AutomationLabsIdentification.Fnn, 
                     AutomationLabsIdentification.Icnn,
                     AutomationLabsIdentification.ResNet, 
                     AutomationLabsIdentification.DenseNet, 
                     AutomationLabsIdentification.Rbf, 
                     AutomationLabsIdentification.PolyNet, 
                     AutomationLabsIdentification.NeuralNetODE_type1, 
                     AutomationLabsIdentification.NeuralNetODE_type2},
    machine_mlj,
    lower_state_constraints,
    higher_state_constraints,
    lower_input_constraints,
    higher_input_constraints,
    )

    # Get state and input number
    nbr_state = size(lower_state_constraints, 1)
    nbr_input = size(lower_input_constraints, 1)

    # Set constraints with Lazy Sets
    x_cons = LazySets.Hyperrectangle(low = lower_state_constraints, high = higher_state_constraints,)
    u_cons = LazySets.Hyperrectangle(low = lower_input_constraints, high = higher_input_constraints)
        
    # Extract best model from the machine
    f_model = MLJ.fitted_params(MLJ.fitted_params(machine_mlj).machine).best_fitted_params[1]

    # Set the system
    system = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(f_model, nbr_state, nbr_input, x_cons, u_cons)
    
    return system
end

"""
    _controller_system_design
A function for design the system (model and constrants) with MathematicalSystems for model predictive control and economic model predictive control.

"""
function _controller_system_design(
    model_mlj::MLJMultivariateStatsInterface.MultitargetLinearRegressor,
    machine_mlj,
    lower_state_constraints,
    higher_state_constraints,
    lower_input_constraints,
    higher_input_constraints,  
    )

    # Get state and input number
    nbr_state = size(lower_state_constraints, 1)
    nbr_input = size(lower_input_constraints, 1)
    
    # Set constraints with Lazy Sets
    x_cons = LazySets.Hyperrectangle(low = lower_state_constraints, high = higher_state_constraints,)
    u_cons = LazySets.Hyperrectangle(low = lower_input_constraints, high = higher_input_constraints)
            
    # Extract model from the machine
    AB_t = MLJ.fitted_params(machine_mlj).coefficients
    AB = copy(AB_t')
    A = AB[:, 1:4]
    B = AB[:, 5: end]

    # Set the system
    system = MathematicalSystems.ConstrainedLinearControlDiscreteSystem(A, B, x_cons, u_cons)

    return system
end

"""
    _design_reference_mpc
A function for references for model predictive control and linearization point for economic model predictive control.

The following variables are mendatories:
* `state_reference`: the state reference.
* `input_reference`: the input reference.
* `horizon`: the horizon for the mpc or empc. 
"""
function _design_reference_mpc(state_reference::Vector, input_reference::Vector, horizon::Int)


    x = state_reference .* ones(size(state_reference,1), horizon + 1)
    u = input_reference .* ones(size(input_reference, 1), horizon)

    references = ModelPredictiveControl.ReferencesStateInput(x, u)

    return references
end


#to do find a better way to get the AutomationLabsIdentification type
function _get_mlj_model_type(machine_mlj::MLJ.Machine{MLJMultivariateStatsInterface.MultitargetLinearRegressor, true})

    return  machine_mlj.model
end

function _get_mlj_model_type(machine_mlj)

    return MLJ.fitted_params(MLJ.fitted_params(machine_mlj).machine).best_model.builder
end