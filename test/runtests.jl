# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

print("Testing modeler implementation...")
took_seconds = @elapsed include("./modeler_implementation_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing design model predictive control...")
took_seconds = @elapsed include("./design_mpc_implementation_test.jl");
println("done (took ", took_seconds, " seconds)")

print("Testing computation terminal ingredient model predictive control...")
took_seconds = @elapsed include("./terminal_ingredient_test.jl");
println("done (took ", took_seconds, " seconds)")


# Extra tests not suitable with github ci
#print("Testing computation model predictive control...")
#took_seconds = @elapsed include("./computation_mpc_test.jl");
#println("done (took ", took_seconds, " seconds)")
