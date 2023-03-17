# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module AutomationLabsModelPredictiveControl

import ControlSystems
import Flux
import ForwardDiff
import HiGHS
import AutomationLabsIdentification
import AutomationLabsSystems
#import InvariantSets
import Ipopt
import JuMP
import LazySets
import MathematicalSystems
import MLJ
import OSQP
import Polyhedra
import SCIP
import LinearAlgebra
import MLJMultivariateStatsInterface

# export 
export proceed_controller

# export computation and update functions
export update_and_compute!
export update_initialization!
export update!
export calculate!

# include sub files
include("types/types.jl")
include("sub/design_mpc.jl")
include("sub/design_empc.jl")
include("sub/solver_selection.jl")

# modeler implementation of models
# Fnn
include("sub/model_modeler_implementation/fnn/mpc_modeler_implementation_fnn.jl")
include("sub/model_modeler_implementation/fnn/empc_modeler_implementation_fnn.jl")

#icnn
include("sub/model_modeler_implementation/icnn/mpc_modeler_implementation_icnn.jl")
include("sub/model_modeler_implementation/icnn/empc_modeler_implementation_icnn.jl")

# resnet 
include("sub/model_modeler_implementation/resnet/mpc_modeler_implementation_resnet.jl")
include("sub/model_modeler_implementation/resnet/empc_modeler_implementation_resnet.jl")

# densenet
include("sub/model_modeler_implementation/densenet/mpc_modeler_implementation_densenet.jl")
include("sub/model_modeler_implementation/densenet/empc_modeler_implementation_densenet.jl")

# rbf
include("sub/model_modeler_implementation/rbf/mpc_modeler_implementation_rbf.jl")
include("sub/model_modeler_implementation/rbf/empc_modeler_implementation_rbf.jl")

# polynet
include("sub/model_modeler_implementation/polynet/mpc_modeler_implementation_polynet.jl")
include("sub/model_modeler_implementation/polynet/empc_modeler_implementation_polynet.jl")

# neuralode
include(
    "sub/model_modeler_implementation/neuralode/mpc_modeler_implementation_neuralode.jl")
#include(
#    "sub/model_modeler_implementation/neuralnetode_type1/empc_modeler_implementation_neuralnetode_type1.jl",
#)

# Rknn1
include(
    "sub/model_modeler_implementation/rknn1/mpc_modeler_implementation_rknn1.jl")
#include(
#    "sub/model_modeler_implementation/neuralnetode_type1/empc_modeler_implementation_neuralnetode_type1.jl",
#)

# Rknn2
include("sub/model_modeler_implementation/rknn2/mpc_modeler_implementation_rknn2.jl")
#include(
#    "sub/model_modeler_implementation/neuralnetode_type1/empc_modeler_implementation_neuralnetode_type1.jl",
#)

# Rknn4
include("sub/model_modeler_implementation/rknn4/mpc_modeler_implementation_rknn4.jl")
#include(
#    "sub/model_modeler_implementation/neuralnetode_type1/empc_modeler_implementation_neuralnetode_type1.jl",
#)

# linear model 
include("sub/model_modeler_implementation/linear/mpc_modeler_implementation_linear.jl")
include("sub/model_modeler_implementation/linear/empc_modeler_implementation_linear.jl")

# include main files
include("main/main_mpc.jl")
include("main/computation_mpc.jl")
include("main/computation_empc.jl")

end
