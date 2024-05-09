from env import HighRiseFireEnv
from scalarisation import *
from execution import *
import pickle

def Q_function_calculator(env, state, V, discount_factor, model_used=None):
    """

    Calculates the value of applying each action to a given state. Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the value obtained for each action
    """
    state_translated = env.translate(state)
    Q_state = np.zeros((env.action_space.n, env.objective_space.n))

    for action in range(env.action_space.n):
        if model_used is not None:
            pass
        else:
            env.reset(force_new_state=state_translated)
            next_state, rewards, _, _ = env.step(action)

        encrypted_next_state = env.encrypt(next_state)

        for objective in range(env.objective_space.n):
            Q_state[action, objective] += rewards[objective] + discount_factor * V[encrypted_next_state, objective]

    return Q_state






def generate_model(env):

    n_objectives = env.objective_space.n
    n_actions = env.action_space.n
    n_states = env.n_states
    model = np.zeros([n_states, n_actions, n_objectives])

    np.save("model.npy", model)


def value_iteration(env, weights, lex=None, theta=1.0, discount_factor=0.7, model_used=None):
    """
    Value Iteration Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 4.4,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic envirnoment, and also
     adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param theta: convergence parameter, the smaller it is the more precise the algorithm
    :param discount_factor: discount factor of the (MO)MPD, can be set at discretion
    :return:
    """

    n_objectives = env.objective_space.n
    n_actions = env.action_space.n
    n_states = env.n_states
    V = np.zeros([n_states, n_objectives])
    Q = np.zeros([n_states, n_actions, n_objectives])

    print(n_states)

    if lex is not None:
        len_l = len(lex)
        weights = [10**(len_l-lex[i]) for i in range(len_l)]

    max_iterations = 100
    for iteration in range(max_iterations):
        # Threshold delta
        delta = 0

        max_delta = 0
        # Sweep for every state
        for i in range(n_states):

            if not env.is_done(env.translate(i)):
                # calculate the value of each action for the state
                Q[i] = Q_function_calculator(env, i, V, discount_factor, model_used)
                # compute the best action for the state
                if lex is None:
                    best_action = np.argmax(scalarised_Qs(Q[i], weights))
                else:
                    best_action = -999
                best_action_value = scalarisation_function(Q[i, best_action], weights)
                # Recalculate delta
                delta += np.abs(best_action_value - scalarisation_function(V[i], weights))
                if delta > max_delta:
                    max_delta = delta
                # Update the state value function
                V[i] = Q[i, best_action]

        # Check if we can finish the algorithm

        if delta < theta:
            print('Iteration ' + str(iteration) + '. Delta = ' + str(round(delta, 3)) + " < Theta = " + str(theta))
            print("Learning Process finished!")
            break
        else:
            print('Iteration ' + str(iteration) + '. Delta = ' + str(round(delta, 3)) + " > Theta = " + str(theta))


    # Output a deterministic optimal policy
    env = HighRiseFireEnv()

    policy = deterministic_optimal_policy_calculator(Q, weights)

    return policy, V, Q

def learn_and_do():
    env = HighRiseFireEnv()
    policy, v, q = value_iteration(env, weights=weights, discount_factor=0.7, model_used=None)
    np.save("policy.npy", policy)
    np.save("v_function.npy", v)
    np.save("q_function.npy", q)

if __name__ == "__main__":


    weights = [1.0, 4.0]

    #generate_model(env)

    train = True
    test = True

    if train:
        learn_and_do()

    policy = np.load("policy.npy")
    v = np.load("v_function.npy")
    q = np.load("q_function.npy")
    print("-------------------")

    print("V(s0): ", v[323])

    #print("---")
    #print("Q(s0): ", q[323])
    #print("---")
    #print("policy(s0):", policy[323])

    if test:
        print("-------------------")
        print("We Proceed to show the learnt policy.")
        example_execution(policy, q, discount_factor=0.7)
        print("-------------------")


