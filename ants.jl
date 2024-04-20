using LinearAlgebra
using Random
using Distributions

const Observation = Int8

function l2_2d(x1, y1, x2, y2)
    return sqrt(((x1 - x2) ^ 2) + ((y1 - y2) ^ 2))
end

function norm_prob(A::Matrix{Float64})
    # return np.dot(x, np.diag(1 / np.sum(x, 0)))
    sA = sum(A, dims=1)
    return A * Diagonal(vec(1 ./ sA))
end

function softmax(x::Union{Matrix{Float64}, Vector{Float64}})
    x = x .- maximum(x)
    x = exp.(x)
    x = x ./ sum(x)
    return x
end


mutable struct MDP
    A::Matrix{Float64}
    B::Array{Float64, 3}
    C::Matrix{Float64}
    p0::Float64
    num_states::Int
    num_obs::Int
    num_actions::Int
    lnA::Matrix{Float64}
    sQ::Matrix{Float64}
    uQ::Matrix{Float64}
    prev_action::Union{Int, Nothing}
    rng::Random.MersenneTwister

    function MDP(A::Matrix{Float64}, B, C; seed=nothing)
        p0 = exp(-16)
        num_states = size(A, 2)
        num_obs = size(A, 1)
        num_actions = size(B, 1)

        A = A .+ p0
        A = norm_prob(A)
        lnA = log.(A)

        B = B .+ p0
        for a in 1:num_actions
            B[a, :, :] = norm_prob(B[a, :, :])
        end

        C = C .+ p0
        C = norm_prob(C)


        sQ = zeros(num_states, 1)
        uQ = zeros(num_actions, 1)
        prev_action = nothing

        if isnothing(seed)
            rng = Random.MersenneTwister()
        else
            rng = Random.MersenneTwister(seed)
        end

        return new(A, B, C, p0, num_states, num_obs, num_actions, lnA, sQ, uQ, prev_action, rng)
    end
end

function set_A(mdp::MDP, A)
    mdp.A = A .+ mdp.p0
    mdp.A = norm_prob(mdp.A)
    mdp.lnA = log.(mdp.A)
end


function reset(mdp::MDP, obs::Observation)
    obs_idx = obs + 1
    likelihood = mdp.lnA[obs_idx, :]
    likelihood = copy(likelihood)
    # likelihood = likelihood[:, np.newaxis]
    mdp.sQ = reshape(softmax(likelihood), (length(likelihood), 1))
    mdp.prev_action = random_action(mdp)
end

function random_action(mdp::MDP)
    return rand(mdp.rng, 1:mdp.num_actions)
end

function step(mdp::MDP, obs::Observation, opposite_actions)
    # state inference
    obs_idx = obs + 1
    likelihood = mdp.lnA[obs_idx, :]
    likelihood = reshape(likelihood, (size(likelihood, 1), 1))
    prior = mdp.B[mdp.prev_action, :, :] * mdp.sQ
    prior = log.(prior)
    mdp.sQ = softmax(prior)

    # error("unimplemented")
    # action inference
    SCALE = 10
    neg_efe = zeros(mdp.num_actions, 1)
    for a in 1:mdp.num_actions
        fs = mdp.B[a, :, :] * mdp.sQ
        fo = mdp.A * fs
        fo = norm_prob(fo .+ mdp.p0)
        utility = sum(fo .* log.(fo ./ mdp.C), dims=1)
        utility = utility[1]
        neg_efe[a] -= utility / SCALE
    end

    # priors
    neg_efe[5] -= 20.0
    neg_efe[opposite_actions[mdp.prev_action]] -= 20.0

    # action selection
    mdp.uQ = softmax(neg_efe)


    dist = Multinomial(1, mdp.uQ[:, 1])
    sample = rand(mdp.rng, dist)
    action = findfirst(sample .== 1)

    mdp.prev_action = action
    return action
end




mutable struct Ant
    mdp::MDP
    x_pos::Int
    y_pos::Int
    traj::Vector{Tuple{Int, Int}}
    distance::Vector{Float64}
    backward_step::Int
    is_returning::Bool

    function Ant(mdp::MDP, init_x::Int, init_y::Int)
        new(mdp, init_x, init_y, [(init_x, init_y)], [], 0, false)
    end
end

function update_forward(ant::Ant, x_pos::Int, y_pos::Int)
    ant.x_pos = x_pos
    ant.y_pos = y_pos
    init_x = ant.traj[1][1]
    init_y = ant.traj[1][2]
    push!(ant.traj, (x_pos, y_pos))
    push!(ant.distance, l2_2d(x_pos, y_pos, init_x, init_y))
end

function update_backward(ant::Ant, x_pos::Int, y_pos::Int)
    ant.x_pos = x_pos
    ant.y_pos = y_pos
    init_x = ant.traj[1][1]
    init_y = ant.traj[1][2]
    push!(ant.distance, l2_2d(x_pos, y_pos, init_x, init_y))
end




struct AntTMazeEnvConfig
    grid_dims::Tuple{Int, Int}
    num_init_ants::Int
    num_max_ants::Int
    add_ant_every::Int
    new_ant_x_init::Int
    new_ant_y_init::Int
    wall_left_x::Int
    wall_right_x::Int
    wall_t_bot_y::Int
    num_pheromone_levels::Int
    num_states::Int
    num_actions::Int
    food_return_to_nest_prob::Float64
    decay_prob::Float64
    action_map::Array{Tuple{Int, Int}, 1}
    opposite_actions::Array{Int, 1}
    food_location::Tuple{Int, Int}
    food_size::Tuple{Int, Int}
end


mutable struct AntTMazeEnv
    cells::Array{Int8, 2}
    config::AntTMazeEnvConfig
    function AntTMazeEnv(config)
        cells = zeros(config.grid_dims...)
        new(cells, config)
    end
end

function env_decay(env::AntTMazeEnv)
    for x in 1:size(env.cells, 1)
        for y in 1:size(env.cells, 2)
            curr_obs = env.cells[x, y]
            if (curr_obs > 0) && (rand() < env.config.decay_prob)
                env.cells[x, y] = curr_obs - 1
            end
        end
    end
end

function get_ant_A(env::AntTMazeEnv, ant)
    A = zeros((env.config.num_pheromone_levels, env.config.num_states))
    for s in 1:env.config.num_states
        delta = env.config.action_map[s]
        neighbor_ph_level = env.cells[ant.x_pos + delta[1], ant.y_pos + delta[2]]
        A[:, s] = [i == neighbor_ph_level ? 1 : 0 for i in 1:env.config.num_pheromone_levels]
    end
    return A
end

function get_ant_obs(env, ant)::Observation
    return env.cells[ant.x_pos, ant.y_pos]
end

function check_food(env::AntTMazeEnv, x_pos::Int, y_pos::Int)
    is_food = false
    if (x_pos > (env.config.food_location[1] - env.config.food_size[1])) &&
       (x_pos < (env.config.food_location[1] + env.config.food_size[1]))
        if (y_pos > (env.config.food_location[2] - env.config.food_size[2])) &&
           (y_pos < (env.config.food_location[2] + env.config.food_size[2]))
            is_food = true
        end
    end
    return is_food
end

function check_walls(env::AntTMazeEnv, orig_x::Int, orig_y::Int, x_pos::Int, y_pos::Int)
    is_valid = true
    if orig_y > env.config.wall_t_bot_y
        if orig_x >= env.config.wall_left_x && x_pos <= env.config.wall_left_x
            is_valid = false
        end
        if orig_x <= env.config.wall_right_x && x_pos >= env.config.wall_right_x
            is_valid = false
        end
    end
    if orig_y <= env.config.wall_t_bot_y
        if y_pos > env.config.wall_t_bot_y && ((x_pos < env.config.wall_left_x) || (x_pos > env.config.wall_right_x))
            valid = false
        end
    end
    return is_valid
end

function step_forward(env::AntTMazeEnv, ant::Ant, action::Int)
    delta = env.config.action_map[action]
    x_pos = clamp(ant.x_pos + delta[1], 2, env.config.grid_dims[1] - 1)
    y_pos = clamp(ant.y_pos + delta[2], 2, env.config.grid_dims[2] - 1)

    if check_food(env, x_pos, y_pos) && rand() < env.config.food_return_to_nest_prob
        ant.is_returning = true
        ant.backward_step = 0
    end

    if check_walls(env, ant.x_pos, ant.y_pos, x_pos, y_pos)
        update_forward(ant, x_pos, y_pos)
    end
end

function step_backward(env::AntTMazeEnv, ant::Ant)
    path_len = length(ant.traj)
    next_step = path_len - (ant.backward_step + 1)
    pos = ant.traj[next_step]
    ant.x_pos = pos[1]
    ant.y_pos = pos[2]
    update_backward(ant, pos[1], pos[2])

    curr_obs = env.cells[pos[1], pos[2]]
    env.cells[pos[1], pos[2]] = min(curr_obs + 1, env.config.num_pheromone_levels - 1)

    ant.backward_step += 1
    if ant.backward_step >= path_len - 1
        ant.is_returning = false
        traj = ant.traj
        ant.traj = [traj[1]]
        return true, traj
    else
        return false, nothing
    end
end




function create_ant(config, env, C)
    A = zeros((config.num_pheromone_levels, config.num_states))
    # B = [Diagonal(ones(config.num_states)) for i in 1:config.num_states]
    B = zeros((config.num_actions, config.num_states, config.num_states))
    for a in 1:config.num_actions
        B[a, a, :] .= 1.0
    end
    mdp = MDP(A, B, C)
    ant = Ant(mdp, config.new_ant_x_init, config.new_ant_y_init)

    # configure new ant
    obs = get_ant_obs(env, ant)
    A = get_ant_A(env, ant)
    set_A(ant.mdp, A)
    reset(ant.mdp, obs)
    return ant
end


function total_pairwise_distance(ants::Vector{Ant})
    t_dis = 0

    for ant in ants
        for ant_2 in ants
            t_dis += l2_2d(ant.x_pos, ant.y_pos, ant_2.x_pos, ant_2.y_pos)
        end
    end
    return t_dis / length(ants)
end


function run()
    GRID_DIMS = (40,40)
    ADD_ANT_EVERY = 50
    NEW_ANT_X_INIT = 20
    NEW_ANT_Y_INIT = 30
    WALL_LEFT_X = 15
    WALL_RIGHT_X = 25
    WALL_T_BOT_Y = 10
    NUM_PHEROMONE_LEVELS = 10
    NUM_STATES = 9
    NUM_ACTIONS = 9
    FOOD_RETURN_TO_NEST_PROB = 0.1
    DECAY_PROB = 0.01
    ACTION_MAP = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    OPPOSITE_ACTIONS = reverse(1:length(ACTION_MAP))
    FOOD_LOCATION = (40, 5)
    FOOD_SIZE = (10, 10)

    NUM_STEPS = 2000
    NUM_INIT_ANTS = 40
    NUM_MAX_ANTS = 70
    SWITCH_FOOD = false

    C::Matrix{Float64} = zeros(NUM_PHEROMONE_LEVELS, 1)
    for o in 1:NUM_PHEROMONE_LEVELS
        C[o] = o - 1
    end

    config = AntTMazeEnvConfig(
        GRID_DIMS,
        NUM_INIT_ANTS,
        NUM_MAX_ANTS,
        ADD_ANT_EVERY,
        NEW_ANT_X_INIT,
        NEW_ANT_Y_INIT,
        WALL_LEFT_X,
        WALL_RIGHT_X,
        WALL_T_BOT_Y,
        NUM_PHEROMONE_LEVELS,
        NUM_STATES,
        NUM_ACTIONS,
        FOOD_RETURN_TO_NEST_PROB,
        DECAY_PROB,
        ACTION_MAP,
        OPPOSITE_ACTIONS,
        FOOD_LOCATION,
        FOOD_SIZE
    )

    env = AntTMazeEnv(config)
    ants::Vector{Ant} = []

    # create initial ants
    for i in 1:NUM_INIT_ANTS
        ant = create_ant(config, env, C)
        push!(ants, ant)
    end

    println("Starting with $NUM_INIT_ANTS ants")

    distance = 0

    print_frac = 0.05
    print_every = Int(NUM_STEPS * print_frac)

    num_completed_trips = 0
    paths = []
    ant_locations = []

    for t in 1:NUM_STEPS
        distance += total_pairwise_distance(ants)

        if t % print_every == 0
            println("Distance at step $t: $distance")
        end

        # periodically add ant (coming out of the nest?)
        if t % ADD_ANT_EVERY == 0 && length(ants) < NUM_MAX_ANTS
            ant = create_ant(config, env, C)
            push!(ants, ant)
            if length(ants) == NUM_MAX_ANTS
                println("Now at $NUM_MAX_ANTS ants (max)")
            end
        end

        if SWITCH_FOOD && t % (NUM_STEPS / 2) == 0
            config.food_location[1] = config.grid_dims[1] - config.food_location[1]
        end

        for ant in ants
            if !ant.is_returning
                obs = get_ant_obs(env, ant)
                A = get_ant_A(env, ant)
                set_A(ant.mdp, A)
                action = step(ant.mdp, obs, config.opposite_actions)
                step_forward(env, ant, action)
            else
                is_complete, traj = step_backward(env, ant)
                num_completed_trips += Int(is_complete)
                if is_complete
                    push!(paths, traj)
                end
            end
        end

        env_decay(env)

        push!(ant_locations, [(ant.x_pos, ant.y_pos) for ant in ants])
    end

    return num_completed_trips, paths, distance

end


num_completed_trips, paths, distance =  run()
println("Number of completed trips: $num_completed_trips")
println("Total distance: $distance")
