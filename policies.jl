using Printf
using Serialization

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

function lookahead(model::MaximumLikelihoodMDP, U::Vector{Float64}, s::Int, a::Int)
    key = (s,a)
    nextdict = get(model.N, key, Dict())
    if isempty(nextdict)
        return 0.0
    end
    total = sum(values(nextdict))
    r = model.ρ[key] / total
    ev = sum((cnt/total) * U[sp] for (sp,cnt) in nextdict)
    return r + model.γ * ev
end

function backup(model::MaximumLikelihoodMDP, U::Vector{Float64}, s::Int)
    vals = [lookahead(model, U, s, a) for a in model.𝒜]
    return isempty(vals) ? 0.0 : maximum(vals)
end

function update!(model::MaximumLikelihoodMDP, s::Int, a::Int, r::Number, s′::Int)
    key = (s,a)
    nextdict = get(model.N, key, Dict{Int,Float64}())
    nextdict[s′] = get(nextdict, s′, 0.0) + 1.0
    model.N[key] = nextdict
    model.ρ[key] = get(model.ρ, key, 0.0) + float(r)
end

struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end
function greedy(𝒫::MaximumLikelihoodMDP, U::Vector{Float64}, s::Int)
    u, a = findmax(a -> lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.𝒫, π.U, s).a

function softmax_probs(Q::Vector{<:Real}, τ::Float64)
    Q_shift = Q .- maximum(Q)         # prevent overflow
    exps = exp.(Q_shift ./ τ)
    return exps ./ sum(exps)
end

# Sample an index from a probability vector p (sums to 1)
function sample_index(p::Vector{<:Real})
    r = rand()          # random number in [0,1)
    cumsum_val = 0.0
    for (i, prob) in enumerate(p)
        cumsum_val += prob
        if r < cumsum_val
            return i     # return 1-based index
        end
    end
    return length(p)    # fallback in case of rounding error
end

# Softmax action selection
function softmax_action(Q::Vector{<:Real}, τ::Float64)
    p = softmax_probs(Q, τ)
    return sample_index(p)
end

# Wrapper for MaximumLikelihoodMDP: returns action given state
function softmax_policy(model::MaximumLikelihoodMDP, U::Vector{Float64}, s::Int, τ::Float64)
    # compute lookahead values for all actions
    Q = [lookahead(model, U, s, a) for a in model.𝒜]
    return softmax_action(Q, τ)
end

# Make a callable policy function
function make_softmax_policy(policy::ValueFunctionPolicy, τ::Float64)
    return s -> softmax_policy(policy.𝒫, policy.U, s, τ)
end
struct ValueIteration
    k_max::Int
    tol::Float64
end

function ValueIteration(k_max::Int; tol=1e-6)
    return ValueIteration(k_max, tol)
end

function solve(M::ValueIteration, 𝒫::MaximumLikelihoodMDP)
    n = length(𝒫.𝒮)
    U = zeros(Float64, n)
    U_new = similar(U)

    for k in 1:M.k_max
        maxdiff = 0.0
        for s in 𝒫.𝒮
            u_s = backup(𝒫, U, s)
            U_new[s] = u_s
            maxdiff = max(maxdiff, abs(U[s]-u_s))
        end
        copy!(U, U_new)
        if maxdiff < M.tol
            break
        end
    end

    𝒫.U = U
    return ValueFunctionPolicy(𝒫, U)
end

make_policy_function(policy::ValueFunctionPolicy) = s -> greedy(policy.𝒫, policy.U, s).a

# ε-greedy policy wrapper for MLE MDP
function make_epsilon_greedy_policy(policy::ValueFunctionPolicy, ε::Float64, default_action::Int=1)
    return s -> begin
        # handle unseen or out-of-bounds states
        if s < 1 || s > length(policy.𝒫.𝒮)
            return default_action
        end
        if rand() < ε
            # explore randomly
            return rand(policy.𝒫.𝒜)
        else
            # greedy action
            return greedy(policy.𝒫, policy.U, s).a
        end
    end
end

# Revised train_max_likelihood
function train_max_likelihood(
        name, csv_name, cache_name, save_name,
        rows, cols, rate, discount = 0.95,
        iters = 1000, k_max = 300, ε = 0.05)

    # Load or initialize MLE MDP
    if isfile(cache_name)
        max_likelihood = load_action_value_function(cache_name)
        println("Loaded cached MDP")
    else
        planner = ValueIteration(k_max)
        max_likelihood = MaximumLikelihoodMDP(
            1:rows, 1:cols, Dict(), Dict(), discount, zeros(rows), planner
        )
    end

    # Load dataset
    lines = get_lines(csv_name)

    println("Training $name")
    for i in 1:iters
        for line in lines
            update!(max_likelihood, line[1], line[2], line[3], line[4])
        end
        if i % 50 == 0
            println("Iteration $i / $iters")
        end
    end

    # Value iteration
    policy = solve(max_likelihood.planner, max_likelihood)
    println("Value iteration done")

    # ε-greedy policy function
    ε_policy = make_epsilon_greedy_policy(policy, ε, default_action=1)

    # Save policy and MDP
    save_policy(save_name, ε_policy, rows)
    save_action_value_function(cache_name, max_likelihood)

    println("Saved $name policy and MDP cache")
end


# 341 for MaximumLikelihoodMDP
# 318 for FullUpdate