import numpy as np
import os
from env import HighRiseFireEnv
from execution import protocol_alignment_prediction



def generate_protocol(only_allowed_actions=True, allowed_actions_model=None):

    pi = np.zeros(env.n_states)

    for s in range(env.n_states):
        translated_state = env.translate(s)

        if only_allowed_actions:

            if allowed_actions_model is not None:
                allowed_actions = allowed_actions_model[s]
            else:
                allowed_actions = env.allowed_actions(translated_state, return_mask=False)

            if len(allowed_actions) > 0:
                pi[s] = allowed_actions[np.random.randint(len(allowed_actions))]
            else:
                pi[s] = 0
        else:
            pi[s] = np.random.randint(env.action_space.n)

    return pi


old_method = False

try:
    os.remove("synth_data.csv")
except:
    print("Could not remove it!")
    pass

env = HighRiseFireEnv()

amount_of_steps = 30
amount_of_iterations = 1
amount_of_protocols = 1000
gamma = 0.7

if not old_method:
    execution_list = list()
else:
    execution = 0

allowed_actions_model = list()

for s in range(env.n_states):

    translated_state = env.translate(s)
    allowed_actions_model.append(env.allowed_actions(translated_state, return_mask=False))

for i in range(amount_of_protocols):

    print("Evaluating protocol " + str(i+1))

    protocol = generate_protocol(allowed_actions_model=allowed_actions_model)
    np.save("Protocols/protocol_" + str(i) + ".npy", protocol)

    Value_1, Value_2 = protocol_alignment_prediction(protocol, discount_factor=gamma, max_steps=amount_of_steps, max_iterations=amount_of_iterations)

    if old_method:
        execution_list = [Value_1] + [Value_2]
        execution_list = np.asarray(execution_list)
    else:
        execution_list.append([Value_1, Value_2])

    if old_method:
        with open("synth_data.csv", "ab") as f:
            #f.write(b"\n")
            print(i)
            np.savetxt(f, [execution_list], delimiter=",")

if not old_method:
    np.save("dataset.npy", execution_list)

if old_method:
    data = [np.loadtxt('synth_data.csv', delimiter=',', skiprows=0)]
    # print the array
    print("Commence???")
    data = data[0]

    print("Now let's investigate a particular patient")

    for i in range(10):
        print(len(data[i]))
        R_1 = data[i][0]
        R_2 = data[i][1]
        print("Value Professionalism : ", R_1)
        print("Value Proximity : ", R_2)