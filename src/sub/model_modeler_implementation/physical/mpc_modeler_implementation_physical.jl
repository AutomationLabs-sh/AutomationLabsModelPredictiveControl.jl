# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Linear MPC implementation method with user defined from discrete non linear function ###

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
    model_mlj::Function,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get A and B matrices from state space
    state_reference = reference.x[:, begin] #state at first reference to compute the jacobian
    input_reference = reference.u[:, begin] #input at first reference to compute the jacobian

    # Linearize the system at state and reference
    system_l = AutomationLabsSystems.proceed_system_linearization(
        system,
        state_reference,
        input_reference,
    )

    model_mpc = _model_predictive_control_modeler_implementation(
        method,
        system_l,
        horizon,
        reference,
        solver;
        kws,
    )

    return model_mpc
end


### Linear MPC implementation method with user defined from continuous non linear function ###

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
    model_mlj::Function,
    system::MathematicalSystems.ConstrainedBlackBoxControlContinuousSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get A and B matrices from state space
    state_reference = reference.x[:, begin] #state at first reference to compute the jacobian
    input_reference = reference.u[:, begin] #input at first reference to compute the jacobian

    system_l = AutomationLabsSystems.proceed_system_linearization(
        system,
        state_reference,
        input_reference,
    )

    if haskey(kws, :mpc_sample_time) == true
        system_l_d =
            AutomationLabsSystems.proceed_system_discretization(system_l, sample_time)
    else
        @error "System discretization is requested but mpc sample time is not provided"
    end

    model_mpc = _model_predictive_control_modeler_implementation(
        method,
        system_l_d,
        horizon,
        reference,
        solver;
        kws,
    )

    return model_mpc
end


### Non linear MPC implementation method with user defined from discrete non linear function ###

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
    method::NonLinearProgramming,
    model_mlj::Function,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get constraints of the dynamical system
    x_hyperrectangle = LazySets.vertices_list(system.X)
    u_hyperrectangle = LazySets.vertices_list(system.U)

    x_constraints = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])
    u_constraints = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

    #get A and B matrices from state space
    state_reference = reference.x[:, begin] #state at first reference to compute the jacobian
    input_reference = reference.u[:, begin] #input at first reference to compute the jacobian

    #the model is designed
    model_mpc = _JuMP_model_definition(method, solver)

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_mpc, begin
        x[1:system.statedim, 1:horizon+1]                  #state variable
        e_x[1:system.statedim, 1:horizon+1]                #state deviation variable
        x_reference[1:system.statedim, 1:horizon+1]        #state reference variable
        u[1:system.inputdim, 1:horizon]             #input variable
        e_u[1:system.inputdim, 1:horizon]           #input deviation variable
        u_reference[1:system.inputdim, 1:horizon]   #input reference variable
    end)

    # Register the non linear function for jump

    #get the activation function of the Fnn through NNlib
    f_activation = get_activation_function(model_mlj, system)
    JuMP.register(model_mpc, :f_activation, 1, f_activation, autodiff = true)


    #### TO DO #####

    # States constraints
    if haskey(kws, :mpc_state_constraint) == true
        for k = 1:1:horizon+1
            for i = 1:1:nbr_states
                JuMP.@constraint(model_mpc, x[i, k] <= x_constraints[i, 2])
                JuMP.@constraint(model_mpc, x_constraints[i, 1] <= x[i, k])
            end
        end
    end

    # Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:nbr_inputs_control
            JuMP.@constraint(model_mpc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_mpc, u_constraints[i, 1] <= u[i, k])
        end
    end

    # Deviation states and deviation inputs constraints
    for i = 1:1:horizon+1
        JuMP.@constraint(model_mpc, e_x[:, i] .== x[:, i] - x_reference[:, i])
    end

    for i = 1:1:horizon
        JuMP.@constraint(model_mpc, e_u[:, i] .== u[:, i] - u_reference[:, i])
    end

    # x reference is set
    for i = 1:horizon+1
        for j = 1:system.statedim
            JuMP.fix(x_reference[j, i], reference.x[j, i]; force = true)
        end
    end

    # u reference is set
    for i = 1:horizon
        for j = 1:system.inputdim
            JuMP.fix(u_reference[j, i], reference.u[j, i]; force = true)
        end
    end

    return model_mpc

end
