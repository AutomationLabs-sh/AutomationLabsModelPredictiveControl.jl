
# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### First case non linear MathematicalSystems from AutomationLabsIdentification ### 
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
mpc_input_constraint,
mpc_state_reference,
mpc_input_reference;
kws_...
)

# Get argument kws
dict_kws = Dict{Symbol,Any}(kws_)
kws = get(dict_kws, :kws, kws_)

# Evaluate if the state constraint is selected
if haskey(kws, :mpc_state_constraint) == true
    #there are state constraints 
    mpc_state_constraint = kws[:mpc_state_constraint]

    # Get state and input number
    nbr_state = size(mpc_state_constraint[:, 1], 1)
    nbr_input = size(mpc_input_constraint[:, 1], 1)

    # Set constraints with Lazy Sets
    lower_state_constraints = mpc_state_constraint[:, 1]
    higher_state_constraints = mpc_state_constraint[:, 2]
    lower_input_constraints = mpc_input_constraint[:, 1]
    higher_input_constraints = mpc_input_constraint[:, 2]

    x_cons = LazySets.Hyperrectangle(low = lower_state_constraints, high = higher_state_constraints,)
    u_cons = LazySets.Hyperrectangle(low = lower_input_constraints, high = higher_input_constraints)
    
    # Extract best model from the machine
    f_model = MLJ.fitted_params(MLJ.fitted_params(machine_mlj).machine).best_fitted_params[1]

    # Set the system
    return system = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(f_model, nbr_state, nbr_input, x_cons, u_cons)
end

# Evaluate if the state constraint is selected
if haskey(kws, :mpc_state_constraint) == false
    #there are no state constraints 

    # Get state and input number
    nbr_state = size(mpc_state_reference[:, 1], 1)
    nbr_input = size(mpc_input_reference[:, 1], 1)

    # Set constraints with Lazy Sets
    lower_state_constraints = zeros(nbr_state, 1)[:, 1]
    higher_state_constraints = zeros(nbr_state, 1)[:, 1]
    lower_input_constraints = mpc_input_constraint[:, 1]
    higher_input_constraints = mpc_input_constraint[:, 2]

    x_cons = LazySets.Hyperrectangle(low = lower_state_constraints, high = higher_state_constraints,)
    u_cons = LazySets.Hyperrectangle(low = lower_input_constraints, high = higher_input_constraints)
     
    # Extract best model from the machine
    f_model = MLJ.fitted_params(MLJ.fitted_params(machine_mlj).machine).best_fitted_params[1]

    # Set the system
    return system = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(f_model, nbr_state, nbr_input, x_cons, u_cons)
end

end

### Second case linear MathematicalSystems from AutomationLabsIdentification ### 
"""
    _controller_system_design
A function for design the system (model and constrants) with MathematicalSystems for model predictive control and economic model predictive control.

"""
function _controller_system_design(
    model_mlj::MLJMultivariateStatsInterface.MultitargetLinearRegressor,
    machine_mlj,
    mpc_input_constraint,
    mpc_state_reference,
    mpc_input_reference;
    kws_...)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Evaluate if the state constraint is selected
    if haskey(kws, :mpc_state_constraint) == true

        #there are state constraints 
        mpc_state_constraint = kws[:mpc_state_constraint]

        # Set constraints with Lazy Sets
        lower_state_constraints = mpc_state_constraint[:, 1]
        higher_state_constraints = mpc_state_constraint[:, 2]
        lower_input_constraints = mpc_input_constraint[:, 1]
        higher_input_constraints = mpc_input_constraint[:, 2]

        # Get state and input number
        nbr_state = size(lower_state_constraints, 1)
        nbr_input = size(lower_input_constraints, 1)
    
        # Set constraints with Lazy Sets
        x_cons = LazySets.Hyperrectangle(low = lower_state_constraints, high = higher_state_constraints,)
        u_cons = LazySets.Hyperrectangle(low = lower_input_constraints, high = higher_input_constraints)
            
        # Extract model from the machine
        AB_t = MLJ.fitted_params(machine_mlj).coefficients
        AB = copy(AB_t')
        A = AB[:, 1:nbr_state]
        B = AB[:, nbr_state+1: end]

        # Set the system
        system = MathematicalSystems.ConstrainedLinearControlDiscreteSystem(A, B, x_cons, u_cons)
    end

    # Evaluate if the state constraint is selected
    if haskey(kws, :mpc_state_constraint) == false

        # Get state and input number
        nbr_state = size(mpc_state_reference[:, 1], 1)
        nbr_input = size(mpc_input_reference[:, 1], 1)

        # Set constraints with Lazy Sets
        lower_state_constraints = zeros(nbr_state, 1)[:, 1]
        higher_state_constraints = zeros(nbr_state, 1)[:, 1]
        lower_input_constraints = mpc_input_constraint[:, 1]
        higher_input_constraints = mpc_input_constraint[:, 2]
 
        # Set constraints with Lazy Sets
        x_cons = LazySets.Hyperrectangle(low = lower_state_constraints, high = higher_state_constraints,)
        u_cons = LazySets.Hyperrectangle(low = lower_input_constraints, high = higher_input_constraints)
            
        # Extract model from the machine
        AB_t = MLJ.fitted_params(machine_mlj).coefficients
        AB = copy(AB_t')
        A = AB[:, 1:nbr_state]
        B = AB[:, nbr_state+1: end]

        # Set the system
        system = MathematicalSystems.ConstrainedLinearControlDiscreteSystem(A, B, x_cons, u_cons)
    end

    return system
end


### Third case linear MathematicalSystems from simple linear model ### 

#=
"""
    _controller_system_design
A function for design the system (model and constrants) with MathematicalSystems for model predictive control and economic model predictive control.

"""
function _controller_system_design(
            model_mlj::LinearModel,
            lower_state_constraints,
            higher_state_constraints,
            lower_input_constraints,
            higher_input_constraints;  
            kws_...
    )

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Evaluate if the state constraint is selected
    if haskey(kws, :mpc_state_constraint) == true
        #there are state constraints 
        mpc_state_constraint = kws[:mpc_state_constraint]

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
=#

### Fourth case non linear MathematicalSystems from simple non linear model ### 