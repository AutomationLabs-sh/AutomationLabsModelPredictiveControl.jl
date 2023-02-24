# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Linear system continuous and discrete ###
"""
    _model_predictive_control_design
Function that tunes a model predictive control.

** Required fields **
* `system`: mathematical system type.
* `horizon`: horizon length of the model predictive control.
* `method`: implementation method (linear, non linear, mixed integer linear, fuzzy rules)
* `sample_time`: time sample of the model predictive control.
* `kws` optional fields
"""

function _model_predictive_control_design(
    system::MathematicalSystems.ConstrainedLinearControlContinuousSystem,
    horizon::Int,
    sample_time::Int,
    references::ReferencesStateInput;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # Discretization
    system_d = AutomationLabsSystems.proceed_system_discretization(system)

    mpc_controller = _model_predictive_control_design(
        system_d,
        horizon,
        sample_time,
        references;
        kws,
    )

    return mpc_controller
end

"""
    _model_predictive_control_design
Function that tunes a model predictive control.

** Required fields **
* `system`: mathematical system type.
* `horizon`: horizon length of the model predictive control.
* `method`: implementation method (linear, non linear, mixed integer linear, fuzzy rules)
* `sample_time`: time sample of the model predictive control.
* `kws` optional fields
"""
function _model_predictive_control_design(
    system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem,
    horizon::Int,
    sample_time::Int,
    references::ReferencesStateInput;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # User defined parameters or users define parameters
    mpc_programming_type = get(kws, :mpc_programming_type, "linear")
    mpc_method_optimization = IMPLEMENTATION_PROGRAMMING_LIST[Symbol(mpc_programming_type)]
    mpc_solver_choosen = get(kws, :mpc_solver, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_solver])
    mpc_solver = _IMPLEMENTATION_SOLVER_LIST[Symbol(mpc_solver_choosen)]
    mpc_terminal_ingredient = get(
        kws,
        :mpc_terminal_ingredient,
        _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_terminal_ingredient],
    )
    mpc_max_time =
        get(kws, :mpc_max_time, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_max_time])

    # create the weight of the controller
    weights = _create_weights_coefficients(system; kws)

    # modeller implementation of model predictive control
    model_mpc = _model_predictive_control_modeler_implementation(
        mpc_method_optimization,
        system,
        horizon,
        references,
        mpc_solver;
        kws,
    )

    # implementetation of terminal ingredients
    model_mpc, P_cost, terminal = _create_terminal_ingredient(
        model_mpc,
        mpc_terminal_ingredient,
        system,
        references,
        weights;
        kws,
    )

    # add the quadratic cost function to the modeler model
    model_mpc = _create_quadratic_cost_function(model_mpc, weights, P_cost)

    ### struct design: MPC tuning ###
    tuning = ModelPredictiveControlTuning(
        model_mpc,
        references,
        horizon,
        weights,
        terminal,
        sample_time,
        mpc_max_time,
    )

    ### struct design: set the controller ###
    initialization, computation_results =
        _memory_allocation_initialization_results_mpc(system, horizon)


    return mpc_controller = ModelPredictiveControlController(
        system,
        tuning,
        initialization,
        computation_results,
    )
end

### Blackbox system from identification or user selection ###
"""
    _model_predictive_control_design
Function that tunes a model predictive control.

** Required fields **
* `system`: mathematical system type.
* `horizon`: horizon length of the model predictive control.
* `method`: implementation method (linear, non linear, mixed integer linear, fuzzy rules)
* `sample_time`: time sample of the model predictive control.
* `kws` optional fields
"""
function _model_predictive_control_design(
    system::Union{
        MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
        MathematicalSystems.ConstrainedBlackBoxControlContinuousSystem,
    },
    horizon::Int,
    sample_time::Int,
    references::ReferencesStateInput;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    # User defined parameters or users define parameters
    mpc_programming_type = get(kws, :mpc_programming_type, "non_linear")
    mpc_method_optimization = IMPLEMENTATION_PROGRAMMING_LIST[Symbol(mpc_programming_type)]
    mpc_solver_choosen = get(kws, :mpc_solver, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_solver])
    mpc_solver = _IMPLEMENTATION_SOLVER_LIST[Symbol(mpc_solver_choosen)]
    mpc_terminal_ingredient = get(
        kws,
        :mpc_terminal_ingredient,
        _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_terminal_ingredient],
    )
    mpc_max_time = get(kws, :mpc_max_time, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_max_time])

    # Create the weight of the controller
    weights = _create_weights_coefficients(system; kws)

    # Evaluate the model from the systems if it a Fnn, DenseNet, or user defined model
    model_type = AutomationLabsSystems.proceed_system_model_evaluation(system)

    # Modeller implementation of model predictive control
    model_mpc = _model_predictive_control_modeler_implementation(
        mpc_method_optimization,
        model_type,
        system,
        horizon,
        references,
        mpc_solver;
        kws,
    )

    # Implementetation of terminal ingredients
    model_mpc, P_cost, terminal = _create_terminal_ingredient(
        model_mpc,
        mpc_terminal_ingredient,
        system,
        references,
        weights;
        kws,
    )

    # Add the quadratic cost function to the modeler model
    model_mpc = _create_quadratic_cost_function(model_mpc, weights, P_cost)

    ### Struct design: MPC tuning ###
    tuning = ModelPredictiveControlTuning(
        model_mpc,
        references,
        horizon,
        weights,
        terminal,
        sample_time,
        mpc_max_time,
    )

    ### Struct design: set the controller ###
    initialization, computation_results =
        _memory_allocation_initialization_results_mpc(system, horizon)

    return mpc_controller = ModelPredictiveControlController(
        system,
        tuning,
        initialization,
        computation_results,
    )
end

"""
    _create_weights_coefficients
Function that create the weighting struct for model predictive control.

** Required fields **
* `system`: the mathematical system from the dynamical system.
* `kws`: optional parameters.
"""
function _create_weights_coefficients(
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    Q = get(kws, :mpc_Q, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_Q])
    R = get(kws, :mpc_R, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_R])
    S = get(kws, :mpc_S, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_S])

    #create the matrices
    QM = Q * LinearAlgebra.Matrix(LinearAlgebra.I, system.statedim, system.statedim)
    RM = R * LinearAlgebra.Matrix(LinearAlgebra.I, system.inputdim, system.inputdim)
    SM = S * LinearAlgebra.Matrix(LinearAlgebra.I, system.inputdim, system.inputdim)

    return WeightsCoefficient(QM, RM, SM)
end

"""
    _create_weights_coefficients
Function that create the weighting struct for model predictive control.

** Required fields **
* `system`: the mathematical system from the dynamical system.
* `kws`: optional parameters.
"""
function _create_weights_coefficients(
    system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    Q = get(kws, :mpc_Q, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_Q])
    R = get(kws, :mpc_R, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_R])
    S = get(kws, :mpc_S, _DEFAULT_PARAMETERS_MODEL_PREDICTIVE_CONTROL[:mpc_S])

    #create the matrices
    QM = Q * LinearAlgebra.Matrix(LinearAlgebra.I, size(system.A, 1), size(system.A, 1))
    RM = R * LinearAlgebra.Matrix(LinearAlgebra.I, size(system.B, 2), size(system.B, 2))
    SM = S * LinearAlgebra.Matrix(LinearAlgebra.I, size(system.B, 2), size(system.B, 2))

    return WeightsCoefficient(QM, RM, SM)
end

"""
    _create_terminal_ingredient
Function that create the terminal ingredient for model predictive control. The terminal ingredients are the terminal weight and terminal constraints. 
The terminal constraints could be optional. 

** Required fields **
* `model_mpc`: the JuMP from the modeler design.
* `terminal_ingredient`: a string to set the terminal constraints.
* `system`: the dynamical system mathemacal systems.
* `references`: the states and inputs references.
* `weights`: the weighting coefficients.
* `kws` optional parameters.
"""
function _create_terminal_ingredient(
    model_mpc::JuMP.Model,
    terminal_ingredient::String,
    system,
    reference_in::ReferencesStateInput,
    weights::WeightsCoefficient;
    kws_...,
)

    # Get argument kws
    dict_kws = Dict{Symbol,Any}(kws_)
    kws = get(dict_kws, :kws, kws_)

    #get the last references to compute the terminal ingredient
    state_reference = reference_in.x[:, end]
    input_reference = reference_in.u[:, end]

    #get variable from model JuMP model_mpc
    e_x = model_mpc[:e_x]

    # Compute the terminal cost P with algebraix ricatti equation (ARE)
    system_l = AutomationLabsSystems.proceed_system_linearization(system, state_reference, input_reference)
    #A_sys, B_sys = system_linearization(system, state_reference, input_reference)
    A_sys = system_l.A 
    B_sys = system_l.B
    P_cost = ControlSystems.are(ControlSystems.Discrete, A_sys, B_sys, weights.Q, weights.R)

    # Add the terminal constraints and evaluate if there are state constraint
    if terminal_ingredient == "equality" #&& haskey(kws, :mpc_state_constraint) == true
        JuMP.@constraint(model_mpc, e_x[:, end] .== 0)

    elseif terminal_ingredient == "contractive" #&&
           #haskey(kws, :mpc_state_constraint) == true
        P_contract = LinearAlgebra.Matrix(LinearAlgebra.I, size(e_x, 1), size(e_x, 1))
        JuMP.@constraint(
            model_mpc,
            e_x[:, end]' * P_contract * e_x[:, end] <=
            0.9 * (e_x[:, begin]' * P_contract * e_x[:, begin])
        )

    elseif terminal_ingredient == "neighborhood" #&&
           #haskey(kws, :mpc_state_constraint) == true

        # Get state constraint
        x_hyperrectangle = LazySets.vertices_list(system.X)
        mpc_state_constraint = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])

        # Get input constraint
        u_hyperrectangle = LazySets.vertices_list(system.U)
        mpc_input_constraint = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

        # Set constraints to zero with Lazy Sets
        lower_state_constraints = mpc_state_constraint[:, 1] .- state_reference
        higher_state_constraints = mpc_state_constraint[:, 2] .- state_reference
        lower_input_constraints = mpc_input_constraint[:, 1] .- input_reference
        higher_input_constraints = mpc_input_constraint[:, 2] .- input_reference

        X = LazySets.Hyperrectangle(
            low = lower_state_constraints,
            high = higher_state_constraints,
        )
        U = LazySets.Hyperrectangle(
            low = lower_input_constraints,
            high = higher_input_constraints,
        )

        # LQR gain
        L = ControlSystems.lqr(ControlSystems.Discrete, A_sys, B_sys, weights.Q, weights.R)

        # Compute the terminal set
        terminal_set = InvariantSets.terminal_set(A_sys, B_sys, X, U, L)

        #Evaluate if LazySets min max is possible to implet?

        #Get Hx and bx
        Hx, bx = InvariantSets.tosimplehrep(terminal_set)

        # Add the terminal constraint to JuMP model
        for i = 1:1:size(Hx, 1)
            JuMP.@constraint(model_mpc, e_x[:, end]' * Hx[i, :] <= bx[i])
        end

    elseif terminal_ingredient == "none" #&& haskey(kws, :mpc_state_constraint) == true
        #no terminal constraint to add

    else
        #no terminal constraint to add
    end

    return model_mpc, P_cost, TerminalIngredient(terminal_ingredient, P_cost)
end

"""
    _create_quadratic_cost_function
Function that create the quadratic cost of the model predictive control.

** Required fields **
* `model_mpc`: the JuMP struct.
* `weights`: the weighing coefficient struct.
* `P_cost`: the terminal ingredient cost.
"""
function _create_quadratic_cost_function(
    model_mpc::JuMP.Model,
    weights::WeightsCoefficient,
    P_cost,
)

    #get variable from model JuMP model_mpc
    u = model_mpc[:u]
    e_x = model_mpc[:e_x]
    e_u = model_mpc[:e_u]
    horizon = size(u, 2)

    #add the delta rate constraints if needed
    if weights.S[1, 1] != 0.0
        #add delta_u variable
        JuMP.@variables(model_mpc, begin
            delta_u[1:size(u, 1), 1:horizon]
        end)

        #Deviation inputs delta rate constraint
        for i = 1:horizon-1
            JuMP.@constraint(model_mpc, delta_u[:, i] .== u[:, i] .- u[:, i+1])
        end

    end

    #Add the quadratic cost function according to weight parameters
    if weights.R[1, 1] != 0.0 && weights.S[1, 1] != 0.0
        JuMP.@objective(
            model_mpc,
            Min,
            e_x[:, end]' * P_cost * e_x[:, end] +
            sum(
                e_x[:, i]' * weights.Q * e_x[:, i] + e_u[:, i]' * weights.R * e_u[:, i] for
                i = 1:horizon
            ) +
            sum(delta_u[:, i]' * weights.S * delta_u[:, i] for i = 1:horizon-1)
        )

    elseif weights.R[1, 1] != 0.0
        JuMP.@objective(
            model_mpc,
            Min,
            e_x[:, end]' * P_cost * e_x[:, end] + sum(
                e_x[:, i]' * weights.Q * e_x[:, i] + e_u[:, i]' * weights.R * e_u[:, i] for
                i = 1:horizon
            )
        )

    else
        JuMP.@objective(
            model_mpc,
            Min,
            e_x[:, end]' * P_cost * e_x[:, end] +
            sum(e_x[:, i]' * weights.Q * e_x[:, i] for i = 1:horizon)
        )

    end

    return model_mpc
end

#get the activation function of the neural networks
#mandatory to evaluation activation function for MILP optimisation #to do
function get_activation_function(
    method::Union{
        AutomationLabsIdentification.Fnn,
        AutomationLabsIdentification.Icnn,
        AutomationLabsIdentification.Rbf,
    },
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
)

    f = system.f
    return getfield(f[2][1], :σ)
end

function get_activation_function(
    method::Union{
        AutomationLabsIdentification.ResNet,
        AutomationLabsIdentification.DenseNet,
        AutomationLabsIdentification.PolyNet,
    },
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
)

    f = system.f
    return getfield(f[2][1].layers.layers[1], :σ)
end

# Memory allocation according to mathematical systems
function _memory_allocation_initialization_results_mpc(
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon,
)

    initialization = Vector{Float64}(undef, system.statedim)
    computation_results = ModelPredictiveControlResults(
        Array{Float64}(undef, system.statedim, horizon + 1),
        Array{Float64}(undef, system.statedim, horizon + 1),
        Array{Float64}(undef, system.inputdim, horizon),
        Array{Float64}(undef, system.inputdim, horizon),
    )

    return initialization, computation_results
end

function _memory_allocation_initialization_results_mpc(
    system::MathematicalSystems.ConstrainedLinearControlDiscreteSystem,
    horizon,
)

    initialization = Vector{Float64}(undef, size(system.A, 1))
    computation_results = ModelPredictiveControlResults(
        Array{Float64}(undef, size(system.A, 1), horizon + 1),
        Array{Float64}(undef, size(system.A, 1), horizon + 1),
        Array{Float64}(undef, size(system.B, 2), horizon),
        Array{Float64}(undef, size(system.B, 2), horizon),
    )

    return initialization, computation_results
end
