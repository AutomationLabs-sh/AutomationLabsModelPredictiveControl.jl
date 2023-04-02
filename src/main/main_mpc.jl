# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
    proceed_controller
A function for model predictive control and economic model predictive control.

The following variables are mendatories:
* `system`: a mathematical systems from AutomationLabsSystems.
* `mpc_controller_type`: a string for model predictive control or economic model predictive control. 
* `mpc_horizon`: a horizon for the predictive controller.
* `mpc_sample_time`: a sample time for the predictive controller.
* `mpc_state_reference`: state reference for mpc or linearization point for empc. 
* `mpc_input_reference`: input reference for mpc or linearization point for empc.
* `kws` optional argument.
"""
function proceed_controller(
    system,
    mpc_controller_type::String,
    mpc_horizon::Int,
    mpc_sample_time::Int,
    mpc_state_reference::Vector,
    mpc_input_reference::Vector;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Evaluate if mpc controller is quadratic or economic
    if mpc_controller_type == "model_predictive_control"

        # Get mpc references over horizon 
        mpc_references =
            _design_reference_mpc(mpc_state_reference, mpc_input_reference, mpc_horizon)

        # Design the model predictive control controller
        controller = _model_predictive_control_design(
            system,
            mpc_horizon,
            mpc_sample_time,
            mpc_references;
            kws,
        )

        return controller
    end
#=
    # Evaluate if mpc controller is quadratic or economic
    if mpc_controller_type == "economic_model_predictive_control"

        # Get default parameters or user parameters
        mpc_method_optimization =
            IMPLEMENTATION_PROGRAMMING_LIST[Symbol(mpc_programming_type)]

        # Set references 
        mpc_linearization_point =
            _design_reference_mpc(mpc_state_reference, mpc_input_reference, mpc_horizon)

        # Design the economic model predictive control controller
        controller = _economic_model_predictive_control_design(
            system,
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
            solver = mpc_solver_type,
        

        return controller
    end=#
end

#NamedTuple default parameters definition
const _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL = (
    mpc_solver = "auto",
    mpc_terminal_ingredient = "none",
    mpc_Q = 100.0,
    mpc_R = 0.1,
    mpc_S = 0.0,
    mpc_max_time = 30.0,
)

"""
    _design_reference_mpc
A function for references for model predictive control and linearization point for economic model predictive control.

The following variables are mendatories:
* `state_reference`: the state reference.
* `input_reference`: the input reference.
* `horizon`: the horizon for the mpc or empc. 
"""
function _design_reference_mpc(
    state_reference::Vector,
    input_reference::Vector,
    horizon::Int,
)

    x = state_reference .* ones(size(state_reference, 1), horizon + 1)
    u = input_reference .* ones(size(input_reference, 1), horizon)

    references = ReferencesStateInput(x, u)

    return references
end
