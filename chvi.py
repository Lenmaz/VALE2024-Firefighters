import convexhull
from env import HighRiseFireEnv
from scalarisation import *
import pickle




def Q_function_calculator(env, state, V, discount_factor, model_used=None, epsilon=-1.0):
    """

    Calculates the (partial convex hull)-value of applying each action to a given state.
    Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the new convex obtained after checking for each action (this is the operation hull of unions)
    """

    hulls = list()

    state_translated = env.translate(state)


    for action in range(env.action_space.n):
        hull_sa_all = []

        if model_used is not None:
           pass

        else:
            env.reset(force_new_state=state_translated)
            next_state, rewards, _, _ = env.step(action)

        encrypted_next_state = env.encrypt(next_state)

        hull_sa = convexhull.translate_hull(rewards, discount_factor, V[encrypted_next_state])
        hull_sa_all = convexhull.sum_hulls(hull_sa, hull_sa_all, epsilon=epsilon)

        for point in hull_sa_all:
            hulls.append(point)

    hulls = np.unique(np.array(hulls), axis=0)

    new_hull = convexhull.get_hull(hulls, epsilon=epsilon)

    return new_hull

def get_full_q_function(env, v, discount_factor=0.7):

    n_states = env.n_states
    n_actions = env.action_space.n

    Q = list()
    for i in range(n_states):
        Q.append(list())
        state_translated = env.translate(i)

        for action in range(n_actions):

            env.reset(force_new_state=state_translated)

            next_state, rewards, _, _ = env.step(action)
            encrypted_next_state = env.encrypt(next_state)

            hull_sa = convexhull.translate_hull(rewards, discount_factor, v[encrypted_next_state])
            Q[i].append(hull_sa)

    return Q

def partial_convex_hull_value_iteration(env, discount_factor=0.7, max_iterations=30, model_used=None, from_scratch=True):
    """
    Partial Convex Hull Value Iteration algorithm adapted from "Convex Hull Value Iteration" from
    Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Calculates the partial convex hull for each state of the MOMDP

    :param env: the environment encoding the MOMDP
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: value function storing the partial convex hull for each state
    """

    n_states = env.n_states

    V = list()
    for i in range(n_states):
        V.append(list())

    if not from_scratch:
        try:
            with open(r"v_function.pickle", "rb") as input_file:
                V = pickle.load(input_file)
        except:
            print("Warning: Pickle file storing values does not exist!")
    iteration = 0
    print("Number of states:", env.n_states)
    origi_eps = 0.75
    eps = 1.0

    while iteration < max_iterations:
        iteration += 1
        if origi_eps >= 0.0:
            eps *= origi_eps
        else:
            eps = -1.0
        # Sweep for every state

        for i in range(n_states):
            if i % 100 == 0:
                print(i)

            if not env.is_done(env.translate(i)):
                V[i] = Q_function_calculator(env, i, V, discount_factor, model_used, eps)

        print("Iterations: ", iteration, "/", max_iterations)
        print(V[323])
        print()

    Q = get_full_q_function(env, V, discount_factor)

    return V, Q


def scalarise_q_function(q_hull, objectives, weights):

    scalarised_q = np.zeros((len(q_hull), len(q_hull[0]), objectives))

    for state in range(len(q_hull)):
        for action in range(len(q_hull[0])):
            best_value = randomized_argmax(scalarised_Qs(q_hull[state][action], weights))
            for obj in range(objectives):
                scalarised_q[state, action, obj] = q_hull[state][action][best_value][obj]
    return scalarised_q

def learn_and_do():
    env = HighRiseFireEnv()
    v, q = partial_convex_hull_value_iteration(env, model_used=None)
    with open(r"v_hull.pickle", "wb") as output_file:
        pickle.dump(v, output_file)

    with open(r"q_hull.pickle", "wb") as output_file:
        pickle.dump(q, output_file)


if __name__ == "__main__":

    learn = False

    if learn:
        learn_and_do()

    # Returns partial convex hull of initial state
    with open(r"v_hull.pickle", "rb") as input_file:
        v_func = pickle.load(input_file)

    with open(r"q_hull.pickle", "rb") as input_file:
        q_func = pickle.load(input_file)

    print("--")


    print(v_func[323])
    #print("---")
    #print(q_func[323][0])

    w = [1.0, 0.23]
    scalarised_q = scalarise_q_function(q_func, objectives=2, weights=w)

    print("----")
    print(scalarised_q[323])

    #pi = deterministic_optimal_policy_calculator(scalarised_q, w)

    #print(pi[323])

    """
 [16.6455308  24.6765605 ] EXISTS [1, 6]
 [17.32083606 24.55302905] EXISTS [1, 5]
 [17.49730956 24.51773435] EXISTS [1, 4]
 [19.9478708  23.2604205 ] EXISTS [1, 0.78]
 [22.4774597  16.246933  ] DOES NOT EXIST??
 [25.46350956 16.18229435] EXISTS [1, 0.77]
 [25.57880558 16.00935032] EXISTS [1, 0.5]
 [27.13879359 12.5505344 ] EXISTS [1, 0.23]
 [27.32697459  9.5830244 ] DOES NOT EXIST??
 [27.92697459  8.9830244 ] EXISTS [1, 0.22]




    """