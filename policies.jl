using Printf
using Serialization

# EpsilonGreedyExploration?
# PolicyIteratio



function save_policy(filename::String, policy_func::Function, num_states::Int)
    open(filename, "w") do f
        for si in 1:num_states
            write(f, @sprintf("%d\n", policy_func(si)))
        end
    end
end

function save_action_value_function(filename::String, model)
    serialize(filename, model)
end

function load_action_value_function(filename::String)
    return deserialize(filename)
end

##################################################
#   Q Learning
##################################################

mutable struct QLearning
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    α # learning rate
end

lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end

function get_lines(filename::String)

    lines = Vector{Vector{Int}}()
    open(filename, "r") do input
        header = readline(input) # Header is ignored

        # Parse lines
        for line in eachline(input)
            sample = parse.(Int, split(line, ','))
            push!(lines, sample)
        end
    end
    return lines
end

##################################################
#   MaximumLikelihoodMDP
##################################################

mutable struct MaximumLikelihoodMDP
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    N # transition count N(s,a,s′)
    ρ # reward sum ρ(s, a)
    γ # discount
    U # value function
    planner
end

function lookahead(model::MaximumLikelihoodMDP, U::AbstractVector{<:Real}, s::Int, a::Int)
    key = (s,a)
    nextdict = get(model.N, key, nothing)
    if nextdict === nothing || isempty(nextdict)
        return 0.0
    end
    # Sum counts and compute expected reward
    n = 0.0
    for c in values(nextdict)
        n += c
    end
    if n == 0.0
        return 0.0
    end
    r = model.ρ[key] / n
    # expected value
    ev = 0.0
    for (sp, cnt) in nextdict
        ev += (cnt / n) * U[sp]
    end
    return r + model.γ * ev
end

function backup(model::MaximumLikelihoodMDP, U::AbstractVector{<:Real}, s::Int)
    best = -Inf
    for a in model.𝒜
        val = lookahead(model, U, s, a)
        if val > best
            best = val
        end
    end
    # if no actions or all -Inf (shouldn't happen) return 0.0
    return isfinite(best) ? best : 0.0
end

function update!(model::MaximumLikelihoodMDP, s::Int, a::Int, r::Number, s′::Int)
    key = (s,a)
    # lazy-create per-(s,a) dict
    nextdict = get(model.N, key, nothing)
    if nextdict === nothing
        nextdict = Dict{Int,Float64}()
        model.N[key] = nextdict
    end
    nextdict[s′] = get(nextdict, s′, 0.0) + 1.0
    model.ρ[key] = get(model.ρ, key, 0.0) + float(r)
    return model
end

struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end
function greedy(𝒫::MaximumLikelihoodMDP, U, s)
    u, a = findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.𝒫, π.U, s).a


struct ValueIteration
    k_max::Int
    tol::Float64
end

function ValueIteration(k_max::Int; tol=1e-6)
    return ValueIteration(k_max, tol)
end

function solve(M::ValueIteration, 𝒫::MaximumLikelihoodMDP)
    n = length(𝒫.𝒮)
    U = zeros(Float64, n)         # current values (indexed 1..n)
    U_new = similar(U)

    for k in 1:M.k_max
        # compute in-place, avoid allocations inside loop
        maxdiff = 0.0
        for s in 𝒫.𝒮
            u_s = backup(𝒫, U, s)
            U_new[s] = u_s
            diff = abs(u_s - U[s])
            if diff > maxdiff
                maxdiff = diff
            end
        end
        # swap / copy
        copy!(U, U_new)
        # early stopping
        if maxdiff < M.tol
            # println("Value iteration converged at iter $k (maxdiff=$maxdiff)")
            break
        end
    end

    𝒫.U = U  # store final values back in model
    return ValueFunctionPolicy(𝒫, U)
end

function make_policy_function(policy::ValueFunctionPolicy)
    return s -> greedy(policy.𝒫, policy.U, s).a
end


# 341 for MaximumLikelihoodMDP
# 318 for FullUpdate