include("policies.jl")

save_outputs = true

function train_max_likelihood(name, csv_name, cache_name, save_name, rows, cols, rate, discount = 0.95, iters = 1000)

    max_likelihood = MaximumLikelihoodMDP(1:rows, 1:cols, 
        Dict{Tuple{Int,Int}, Dict{Int,Int}}(),  # Can't fit in a 3D array
        Dict{Tuple{Int,Int}, Float64}(),        # rewards also in a dictionary
        discount, 
        zeros(Float64, rows)
    )

    lines = get_lines(csv_name)

    print("Training "); println(name)
    for i in 1:iters
        for line in lines
            s, a, r, s′ = line
            update!(max_likelihood, s, a, r, s′)
        end
    end
    println("Done")

    softmax = SoftmaxExploration(.05, 0.99)
    println("Solved")

    softmax_policy(state::Int) = softmax(max_likelihood, state)

    save_policy(save_name, softmax_policy, rows)
    save_action_value_function(cache_name, max_likelihood)

    print("Saved "); println(name)
end

##################################################
#   Small CSV - 100 states
##################################################

# train_max_likelihood("small", "small.csv", "small_cache", "policies/small.policy", 100, 4, 1, 1000)
# train_max_likelihood("medium", "medium.csv", "medium_cache", "policies/medium.policy", 50000, 7, 1, 1000)
train_max_likelihood("large", "large.csv", "large_cache", "policies/large.policy", 302020, 9, 0.95, 1000)