# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Linear MPC implementation method with neural net ode ###

"""
    _model_predictive_control_modeler_implementation
Modeler implementation of a Model Predictive Control.

** Required fields **
* `method`: method which is multipled dispatched type. Type are LinearProgramming, TakagiSugeno, NonLinearProgramming and MixedIntegerLinearProgramming.
* `model_mlj`: model of the dynamical system from AutomationLabsIdentification package.
* `system`: system from MethematicalSystems.
* `horizon` : model predictive control horizon parameter.
* `references` : model predictive control references.
* `solver`: model predictive control solver selection.
"""
function _model_predictive_control_modeler_implementation(
    method::LinearProgramming,
    model_mlj::AutomationLabsIdentification.NeuralNetODE_type1,
    system::MathematicalSystems.ConstrainedBlackBoxControlContinuousSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get A and B matrices from state space
    state_reference = reference.x[:, begin] #state at first reference to compute the jacobian
    input_reference = reference.u[:, begin] #input at first reference to compute the jacobian

    system_l = AutomationLabsSystems.proceed_system_linearization(system, state_reference, input_reference)

    if haskey(kws, :mpc_sample_time) == true  
        system_l_d = AutomationLabsSystems.proceed_system_discretization(system_l, sample_time)
    else
        @error "System discretization is requested but mpc sample time is not provided"
    end

    model_mpc = _model_predictive_control_modeler_implementation(
        method,
        system_l_d,
        horizon,
        reference,
        solver;
        kws
    )   

    return model_mpc
end