
include("policies.jl")

save_outputs = true
# save_outputs = false

train_small = true
train_medium = true
train_large = true

function train(name, csv_name, cache_name, save_name, rows, cols, rate, discount = 0.95, iters = 1000)

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
##################################################
#   Small CSV - 100 states
##################################################

learning_rate = 0.2 # dumb hyperparameter

# train("small", "small.csv", "small_cache", "small.policy", 100, 4, learning_rate, 0.95)
train("medium", "medium.csv", "medium_cache", "medium.policy", 50000, 7, learning_rate, 1, 10000)
# train("large", "large.csv", "large_cache", "large.policy", 302020, 9, learning_rate, 0.95)