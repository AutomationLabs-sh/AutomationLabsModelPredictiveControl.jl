# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

"""
    _model_predictive_control_modeler_implementation
Modeler implementation of a Model Predictive Control.

** Required fields **
* `method`: method which is multipled dispatched type. Type are LinearProgramming, TakagiSugeno, NonLinearProgramming and MixedIntegerLinearProgramming.
* `system`: system from MethematicalSystems.
* `horizon` : model predictive control horizon parameter.
* `references` : model predictive control references.
* `solver`: model predictive control solver selection.
"""
function _model_predictive_control_modeler_implementation(
    method::LinearProgramming,
    system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...
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
    A  = system.A;
    B  = system.B;

    #the jump model is designed       
    model_mpc = _JuMP_model_definition(method, solver)

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_mpc, begin
        x[1:size(system.A, 1), 1:horizon+1]                  #state variable
        e_x[1:size(system.A, 1), 1:horizon+1]                #state deviation variable
        x_reference[1:size(system.A, 1), 1:horizon+1]        #state reference variable
        u[1:size(system.B, 2), 1:horizon]             #input variable
        e_u[1:size(system.B, 2), 1:horizon]           #input deviation variable
        u_reference[1:size(system.B, 2), 1:horizon]   #input reference variable
    end)

    #Linear model into JuMP    
    for k = 1:1:horizon
        JuMP.@constraint(model_mpc, e_x[:, k+1] .== A * e_x[:, k] + B * e_u[:, k]) #output layer
    end

    if haskey(kws, :mpc_state_constraint) == true 
        #States constraints
        for k = 1:1:horizon+1
            for i = 1:1:size(system.A, 1)
                JuMP.@constraint(model_mpc, x[i, k] <= x_constraints[i, 2])
                JuMP.@constraint(model_mpc, x_constraints[i, 1] <= x[i, k])
            end
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:size(system.B, 2)
            JuMP.@constraint(model_mpc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_mpc, u_constraints[i, 1] <= u[i, k])
        end
    end

    #Deviation states and deviation inputs constraints
    for i = 1:1:horizon+1
        JuMP.@constraint(model_mpc, e_x[:, i] .== x[:, i] - x_reference[:, i])
    end

    for i = 1:1:horizon
        JuMP.@constraint(model_mpc, e_u[:, i] .== u[:, i] - u_reference[:, i])
    end

    #x reference is set
    for i = 1:horizon+1
        for j = 1:size(system.A, 1)
            JuMP.fix(x_reference[j, i], reference.x[j, i]; force = true)
        end
    end
    #u reference is set
    for i = 1:horizon
        for j = 1:size(system.B, 2)
            JuMP.fix(u_reference[j, i], reference.u[j, i]; force = true)
        end
    end

    return model_mpc

end

#=
### TO DO delete not necessary ?###
"""
    _model_predictive_control_modeler_implementation
Modeler implementation of a Model Predictive Control.

** Required fields **
* `method`: method which is multipled dispatched type. Type are LinearProgramming, TakagiSugeno, NonLinearProgramming and MixedIntegerLinearProgramming.
* `system`: system from MethematicalSystems.
* `horizon` : model predictive control horizon parameter.
* `references` : model predictive control references.
* `solver`: model predictive control solver selection.
"""
function _model_predictive_control_modeler_implementation(
    method::LinearProgramming,
    system::MathematicalSystems.ConstrainedLinearControlContinuousSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get constraints of the dynamical system
    x_hyperrectangle = LazySets.vertices_list(system.X)
    u_hyperrectangle = LazySets.vertices_list(system.U)

    x_constraints = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])
    u_constraints = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

    # Discretise the continuous linear system 
    system_d = AutomationLabsSystems.proceed_discretization(system)

    #get A and B matrices from state space discrete system
    A  = system_d.A;
    B  = system_d.B;

    #the jump model is designed       
    model_mpc = _JuMP_model_definition(method, solver)

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_mpc, begin
        x[1:size(system.A, 1), 1:horizon+1]                  #state variable
        e_x[1:size(system.A, 1), 1:horizon+1]                #state deviation variable
        x_reference[1:size(system.A, 1), 1:horizon+1]        #state reference variable
        u[1:size(system.B, 2), 1:horizon]             #input variable
        e_u[1:size(system.B, 2), 1:horizon]           #input deviation variable
        u_reference[1:size(system.B, 2), 1:horizon]   #input reference variable
    end)

    #Linear model into JuMP    
    for k = 1:1:horizon
        JuMP.@constraint(model_mpc, e_x[:, k+1] .== A * e_x[:, k] + B * e_u[:, k]) #output layer
    end

    if haskey(kws, :mpc_state_constraint) == true 
        #States constraints
        for k = 1:1:horizon+1
            for i = 1:1:size(system.A, 1)
                JuMP.@constraint(model_mpc, x[i, k] <= x_constraints[i, 2])
                JuMP.@constraint(model_mpc, x_constraints[i, 1] <= x[i, k])
            end
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:size(system.B, 2)
            JuMP.@constraint(model_mpc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_mpc, u_constraints[i, 1] <= u[i, k])
        end
    end

    #Deviation states and deviation inputs constraints
    for i = 1:1:horizon+1
        JuMP.@constraint(model_mpc, e_x[:, i] .== x[:, i] - x_reference[:, i])
    end

    for i = 1:1:horizon
        JuMP.@constraint(model_mpc, e_u[:, i] .== u[:, i] - u_reference[:, i])
    end

    #x reference is set
    for i = 1:horizon+1
        for j = 1:size(system.A, 1)
            JuMP.fix(x_reference[j, i], reference.x[j, i]; force = true)
        end
    end
    #u reference is set
    for i = 1:horizon
        for j = 1:size(system.B, 2)
            JuMP.fix(u_reference[j, i], reference.u[j, i]; force = true)
        end
    end

    return model_mpc
end
=#


#=

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
    model_mlj::MLJMultivariateStatsInterface.MultitargetLinearRegressor,
    system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...
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
    A  = system.A;
    B  = system.B;

    #the jump model is designed       
    model_mpc = _JuMP_model_definition(method, solver)

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_mpc, begin
        x[1:size(system.A, 1), 1:horizon+1]                  #state variable
        e_x[1:size(system.A, 1), 1:horizon+1]                #state deviation variable
        x_reference[1:size(system.A, 1), 1:horizon+1]        #state reference variable
        u[1:size(system.B, 2), 1:horizon]             #input variable
        e_u[1:size(system.B, 2), 1:horizon]           #input deviation variable
        u_reference[1:size(system.B, 2), 1:horizon]   #input reference variable
    end)

    #Linear model into JuMP    
    for k = 1:1:horizon
        JuMP.@constraint(model_mpc, e_x[:, k+1] .== A * e_x[:, k] + B * e_u[:, k]) #output layer
    end

    if haskey(kws, :mpc_state_constraint) == true 
        #States constraints
        for k = 1:1:horizon+1
            for i = 1:1:size(system.A, 1)
                JuMP.@constraint(model_mpc, x[i, k] <= x_constraints[i, 2])
                JuMP.@constraint(model_mpc, x_constraints[i, 1] <= x[i, k])
            end
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:size(system.B, 2)
            JuMP.@constraint(model_mpc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_mpc, u_constraints[i, 1] <= u[i, k])
        end
    end

    #Deviation states and deviation inputs constraints
    for i = 1:1:horizon+1
        JuMP.@constraint(model_mpc, e_x[:, i] .== x[:, i] - x_reference[:, i])
    end

    for i = 1:1:horizon
        JuMP.@constraint(model_mpc, e_u[:, i] .== u[:, i] - u_reference[:, i])
    end

    #x reference is set
    for i = 1:horizon+1
        for j = 1:size(system.A, 1)
            JuMP.fix(x_reference[j, i], reference.x[j, i]; force = true)
        end
    end
    #u reference is set
    for i = 1:horizon
        for j = 1:size(system.B, 2)
            JuMP.fix(u_reference[j, i], reference.u[j, i]; force = true)
        end
    end

    return model_mpc

end

=#