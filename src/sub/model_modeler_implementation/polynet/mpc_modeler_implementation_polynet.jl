# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Linear MPC implementation method ###

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
    model_mlj::AutomationLabsIdentification.PolyNet,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
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
       
    # Linearize the system at state and reference
    system_l = AutomationLabsSystems.proceed_system_linearization(system, state_reference, input_reference)
   
    model_mpc = _model_predictive_control_modeler_implementation(
        method,
        system_l,
        horizon,
        reference,
        solver;
        kws
    )
   
    return model_mpc
end

### Non linear MPC implementation method with PolyNet ###

function _model_predictive_control_modeler_implementation(
    method::NonLinearProgramming,
    model_mlj::AutomationLabsIdentification.PolyNet,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get constraints
    x_hyperrectangle = LazySets.vertices_list(system.X)
    u_hyperrectangle = LazySets.vertices_list(system.U)

    x_constraints = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])
    u_constraints = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

    #the model is designed
    model_mpc = _JuMP_model_definition(method, solver)

    #get information from the state, input and neural network and extract neural weughts 
    weights = Flux.params(system.f)
    nbr_inputs = size(weights[1], 2)
    nbr_neurons = size(weights[1], 1)
    nbr_outputs = size(weights[length(weights)], 1)
    nbr_hidden = trunc(Int, (length(weights) - 2) / 2)
    nbr_inputs_control = nbr_inputs - nbr_outputs
    nbr_states = nbr_outputs

    #get neural weights
    W_layer = [] #memory allocation need 3D array multi dimention
    B_layer = [] #memory allocation, need 3D array multi dimention

    push!(W_layer, weights[1])

    for i = 2:2:length(weights)-1
        push!(W_layer, weights[i])
        push!(B_layer, weights[i+1])
    end

    push!(W_layer, weights[length(weights)])

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_mpc, begin
        x[1:nbr_states, 1:horizon+1]                 #state variable
        e_x[1:nbr_states, 1:horizon+1]               #state deviation variable
        x_reference[1:nbr_states, 1:horizon+1]       #state reference variable
        u[1:nbr_inputs_control, 1:horizon]            #input variable
        e_u[1:nbr_inputs_control, 1:horizon]          #input deviation variable
        u_reference[1:nbr_inputs_control, 1:horizon]  #input reference variable
        y[1:nbr_neurons, 1:nbr_hidden+1, 1:horizon]  #output hidden neuron network
        branch_poly[1:nbr_neurons, 1:nbr_hidden, 1:horizon]  #output hidden neuron network
    end)

    f_activation = get_activation_function(model_mlj, system)
    JuMP.register(model_mpc, :f_activation, 1, f_activation, autodiff = true)

    #Modeling the PolyNet into JuMP    
    for k = 1:1:horizon
        for i = 1:1:nbr_neurons
            JuMP.@constraint(
                model_mpc,
                y[i, 1, k] == W_layer[1][i, :]' * [x[:, k]; u[:, k]]
            ) #input layer
        end

        for j = 2:1:nbr_hidden+1
            for i = 1:1:nbr_neurons
                JuMP.@NLconstraint( model_mpc, branch_poly[i, j-1, k] == f_activation([W_layer[j][i, :]' * y[:, j-1, k] + B_layer[j-1][i]][1]) )
                JuMP.@NLconstraint( model_mpc, y[i, j, k] == y[i, j-1, k] 
                + branch_poly[i, j-1, k]
                + f_activation([W_layer[j][i, :]' * branch_poly[:, j-1, k] + B_layer[j-1][i]][1])
                ) #hidden layer [1]for scalar expression
            end
        end

        JuMP.@constraint(model_mpc, x[:, k+1] .== W_layer[:][end] * y[:, end, k]) #output layer
    end

    if haskey(kws, :mpc_state_constraint) == true 
        #States constraints
        for k = 1:1:horizon+1
            for i = 1:1:nbr_states
                JuMP.@constraint(model_mpc, x[i, k] <= x_constraints[i, 2])
                JuMP.@constraint(model_mpc, x_constraints[i, 1] <= x[i, k])
            end
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:nbr_inputs_control
            JuMP.@constraint(model_mpc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_mpc, u_constraints[i, 1] <= u[i, k])
        end
    end

    #Deviation states and inputs constraints
    for i = 1:1:horizon+1
        JuMP.@constraint(model_mpc, e_x[:, i] .== x[:, i] - x_reference[:, i])
    end

    for i = 1:1:horizon
        JuMP.@constraint(model_mpc, e_u[:, i] .== u[:, i] - u_reference[:, i])
    end

    #x reference is set
    for i = 1:horizon+1
        for j = 1:system.statedim
            JuMP.fix(x_reference[j, i], reference.x[j, i]; force = true)
        end
    end

    #u reference is set
    for i = 1:horizon
        for j = 1:system.inputdim
            JuMP.fix(u_reference[j, i], reference.u[j, i]; force = true)
        end
    end

    return model_mpc

end

### Mixed integer linear MPC implementation method with ResNet ###

function _model_predictive_control_modeler_implementation(
    method::MixedIntegerLinearProgramming,
    model_mlj::AutomationLabsIdentification.PolyNet,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers;
    kws_...
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get constraints
    x_hyperrectangle = LazySets.vertices_list(system.X)
    u_hyperrectangle = LazySets.vertices_list(system.U)

    x_constraints = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])
    u_constraints = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

    #the model is designed
    model_mpc = _JuMP_model_definition(method, solver)

    #get information from the state, input and neural network and extract neural weughts 
    weights = Flux.params(system.f)
    nbr_inputs = size(weights[1], 2)
    nbr_neurons = size(weights[1], 1)
    nbr_outputs = size(weights[length(weights)], 1)
    nbr_hidden = trunc(Int, (length(weights) - 2) / 2)
    nbr_inputs_control = nbr_inputs - nbr_outputs
    nbr_states = nbr_outputs

    #get neural weights
    W_layer = [] #memory allocation need 3D array multi dimention
    B_layer = [] #memory allocation, need 3D array multi dimention

    push!(W_layer, weights[1])

    for i = 2:2:length(weights)-1
        push!(W_layer, weights[i])
        push!(B_layer, weights[i+1])
    end

    push!(W_layer, weights[length(weights)])

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_mpc, begin
        x[1:nbr_states, 1:horizon+1]                 #state variable
        e_x[1:nbr_states, 1:horizon+1]               #state deviation variable
        x_reference[1:nbr_states, 1:horizon+1]       #state reference variable
        u[1:nbr_inputs_control, 1:horizon]           #input variable
        e_u[1:nbr_inputs_control, 1:horizon]         #input deviation variable
        u_reference[1:nbr_inputs_control, 1:horizon] #input reference variable
        y_be[1:nbr_neurons, 1:nbr_hidden, 1:horizon]   #output hidden neuron network before resnet addition
        y_af[1:nbr_neurons, 1:nbr_hidden+1, 1:horizon]   #output hidden neuron network after resnet addition
        Bin_1[1:nbr_neurons, 1:nbr_hidden, 1:horizon], Bin
        Bin_2[1:nbr_neurons, 1:nbr_hidden, 1:horizon], Bin
    end)

    #BigM variable
    BigM = 1000

    #Modeling the Fnn into JuMP with relu
    for k = 1:1:horizon
        for i = 1:1:nbr_neurons
            JuMP.@constraint(
                model_mpc,
                y_af[i, 1, k] == W_layer[1][i, :]' * [x[:, k]; u[:, k]]
            ) #input layer
        end

        for j = 1:1:nbr_hidden
            for i = 1:1:nbr_neurons
                JuMP.@constraint(model_mpc, y_be[i, j, k] >= 0)

                JuMP.@constraint(model_mpc, y_be[i, j, k] >= [W_layer[j+1][i, :]' * y_af[:, j, k] + B_layer[j][i]][1] 
                                    + [W_layer[j+1][i, :]' * (W_layer[j+1][i, :]' * y_af[:, j, k] + B_layer[j][i])][1][1] + B_layer[j][i]
                )
                JuMP.@constraint(model_mpc, y_be[i, j, k] .<= [W_layer[j+1][i, :]' * y_af[:, j, k] + B_layer[j][i]][1] 
                    + [W_layer[j+1][i, :]' * (W_layer[j+1][i, :]' * y_af[:, j, k] + B_layer[j][i])][1][1] + B_layer[j][i] .+ BigM * (1 .- Bin_1[i, j, k])
                )
                JuMP.@constraint(
                    model_mpc,
                    y_be[i, j, k] .<= 0 .+ BigM * (1 .- Bin_2[i, j, k])
                )
                JuMP.@constraint(model_mpc, Bin_1[i, j, k] + Bin_2[i, j, k] == 1)
            end
        end

        for j = 2:1:nbr_hidden+1
            for i = 1:1:nbr_neurons
                JuMP.@constraint(
                    model_mpc,
                    y_af[i, j, k] == y_be[i, j-1, k] + y_af[i, j-1, k]
                )
            end
        end


        JuMP.@constraint(model_mpc, x[:, k+1] .== W_layer[:][end] * y_af[:, end, k]) #output layer
    end

    if haskey(kws, :mpc_state_constraint) == true 
        #States constraints
        for k = 1:1:horizon+1
            for i = 1:1:nbr_states
                JuMP.@constraint(model_mpc, x[i, k] <= x_constraints[i, 2])
                JuMP.@constraint(model_mpc, x_constraints[i, 1] <= x[i, k])
            end
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:nbr_inputs_control
            JuMP.@constraint(model_mpc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_mpc, u_constraints[i, 1] <= u[i, k])
        end
    end

    #Deviation states and inputs constraints
    for i = 1:1:horizon+1
        JuMP.@constraint(model_mpc, e_x[:, i] .== x[:, i] - x_reference[:, i])
    end

    for i = 1:1:horizon
        JuMP.@constraint(model_mpc, e_u[:, i] .== u[:, i] - u_reference[:, i])
    end

    #x reference is set
    for i = 1:horizon+1
        for j = 1:system.statedim
            JuMP.fix(x_reference[j, i], reference.x[j, i]; force = true)
        end
    end

    #u reference is set
    for i = 1:horizon
        for j = 1:system.inputdim
            JuMP.fix(u_reference[j, i], reference.u[j, i]; force = true)
        end
    end

    return model_mpc
end