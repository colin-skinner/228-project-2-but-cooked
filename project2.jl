
include("policies.jl")

train = false
print = true

# Executes policy at a state and depth
# function simulate(𝒫::MDP, s, π, d)
#     τ = []

#     for i = 1:d
#         a = π(s)
#         s′, r = 𝒫.TR(s,a)
#         push!(τ, (s,a,r))
#         s = s′
#     end

#     return τ
# end

bad_policy(state::Int) = rand([2,3])


if print
    save_policy("small.policy", bad_policy, 100)
    save_policy("medium.policy", bad_policy, 50000)
    save_policy("large.policy", bad_policy, 302020)
end

# Small - 100 states
# Medium - 50000 states
# Large - 302020 states
