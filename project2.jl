
include("policies.jl")

save_outputs = true

function train_q_learning(name, csv_name, cache_name, save_name, rows, cols, rate, discount = 0.95, iters = 1000)

    if isfile(cache_name)
        q_learning::QLearning = load_action_value_function(cache_name)
    else
        q_learning = QLearning(1:rows, 1:cols, discount, zeros(rows, cols), rate)
    end

    lines = get_lines(csv_name)

    print("Training "); println(name)
    for i in 1:iters
        for line in lines
            update!(q_learning, line[1], line[2], line[3], line[4])
        end
        println(i)
    end
    println("Done")

    action_map = [argmax(q_learning.Q[row, :]) for row in 1:rows]
    policy(state::Int) = action_map[state]

    save_policy(save_name, policy, rows)
    save_action_value_function(cache_name, q_learning)

    print("Saved "); println(name)
end

function train_max_likelihood(name, csv_name, cache_name, save_name, rows, cols, rate, discount = 0.95, iters = 1000)

    planner = ValueIteration(300)
    if isfile(cache_name)
        max_likelihood::MaximumLikelihoodMDP = load_action_value_function(cache_name)
    else
        max_likelihood = MaximumLikelihoodMDP(
            1:rows, 1:cols, Dict(), Dict(), discount, zeros(rows), planner)
    end

    lines = get_lines(csv_name)

    print("Training "); println(name)
    for i in 1:iters
        for line in lines
            update!(max_likelihood, line[1], line[2], line[3], line[4])
        end
        println(i)
    end

    println("Done")
    policy = solve(max_likelihood.planner, max_likelihood)

    println("Solved")

    # stochastic_policy = epsilon_greedy_policy(policy, 0.05)
    τ = 1.0  # small temperature → mostly greedy; increase to explore more
    soft_policy = make_softmax_policy(policy, τ)
    


    save_policy(save_name, soft_policy, rows)
    save_action_value_function(cache_name, max_likelihood)

    print("Saved "); println(name)
end

##################################################
#   Small CSV - 100 states
##################################################

learning_rate = 1 # dumb hyperparameter

# train_q_learning("small", "small.csv", "small_cache", "small.policy", 100, 4, learning_rate, 0.95)
# train_q_learning("medium", "medium.csv", "medium_cache", "medium.policy", 50000, 7, learning_rate, 1, 1000)
train_max_likelihood("medium", "medium.csv", "medium_cache", "medium.policy", 50000, 7, learning_rate, 1, 1000)
# train_max_likelihood("large", "large.csv", "large_cache", "large.policy", 302020, 9, learning_rate, 0.95, 1000)

# train_q_learning("large", "large.csv", "large_cache", "large.policy", 302020, 9, learning_rate, 0.95, 1000)