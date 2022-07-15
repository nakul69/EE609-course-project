from random import sample
import pickle
#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torch
import tf_util
from tensorflow import keras
import gym
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def sample_action(observation,expert_policy):
    observation = observation.reshape(1, -1)
    logits = expert_policy(observation)
    action = tf.random.categorical(logits, 1)
    return action.eval()[0,0]

def main():
    #===========================================================================
    # generate expert data
    #===========================================================================
    # parameters for the algorithm

    test_states = torch.load('test_states.pt')
    test_expert_actions = torch.load('test_expert_actions.pt')
    total_trajs_collected = torch.load('total_trajs.pt')

    test_states = test_states.cpu().numpy()
    test_expert_actions = test_expert_actions.cpu().numpy()

    envname = 'CartPole-v1'
    render = 0
    num_rollouts = 10
    dagger_iters=10
    # policy_fn contains expert policy(loaded using keras)
    policy_fn = keras.models.load_model("PPOexpert")
    loss_list = []
    crash_list = []
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(envname)
        max_steps = env.spec.max_episode_steps
    
        returns = []
        observations = []
        actions = []
        #First we collect some data using the expert policy provided to us
        for i in tqdm(range(num_rollouts)):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sample_action(obs,policy_fn)
                # action using expert policy policy_fn
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                #if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
    
        print('Expert returns', returns)
        print('Expert mean return', np.mean(returns))
        print('Expert std of return', np.std(returns))
        
        # pass observations, actions to imitation learning
        obs_data = np.array(observations)
        act_data = np.array(actions)
        act_data=act_data.reshape(-1,1)
        
    save_expert_mean = np.mean(returns)
    save_expert_std = np.std(returns)

    print(obs_data.shape)
    print(act_data.shape)
    
    #===========================================================================
    # set up the network structure for the imitation learning policy function
    #===========================================================================
    # dim for input/output
    #This may be different for different environmennts. But in both our applications action dimension is same(1)
    obs_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]
    
    # architecture of the neural network policy function. We use two hidden layers having 64 and 32 units repectively
    x = tf.placeholder(tf.float32, shape=[None, obs_dim])
    yhot = tf.placeholder(tf.float32, shape=[None, act_dim])

    print(x)
    print(yhot)
    
    h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
    yhat = tf.layers.dense(inputs=h3, units=act_dim, activation=None)
    
    loss_l2 = tf.reduce_mean(tf.square(yhot - yhat))                #loss function defined as the square loss
    train_step = tf.train.AdamOptimizer().minimize(loss_l2)

    #===========================================================================
    # run DAgger alg
    #===========================================================================
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # record return and std for plotting
        save_mean = []
        save_std = []
        save_train_size = []
        episode_wise_return = []
        # loop for dagger alg
        loss = 0
        for i in range(100):
            for j in range(max_steps):
                obs = test_states[i,j,:]
                action = yhat.eval(feed_dict={x:obs[None,:]})
                if action>=0.5:
                    action=1.0
                else:
                    action=0.0
                expert_action = test_expert_actions[i,j]
                loss +=(action-expert_action)**2
                #print(action,expert_action)

        loss_list.append(loss/100)
        for i_dagger in tqdm(range(dagger_iters)):
            print('DAgger iteration ', i_dagger)
            # train a policy by fitting the MLP
            batch_size = 25
            max_steps = env.spec.max_episode_steps

            for step in range(100):
                batch_i = np.random.randint(0, obs_data.shape[0], size=batch_size)
                train_step.run(feed_dict={x: obs_data[batch_i, ], yhot: act_data[batch_i, ]})
                # if (step % 10 == 0):
                #     print('opmization step ', step)
                #     print('obj value is ', loss_l2.eval(feed_dict={x:obs_data, yhot:act_data}))
                loss = 0
                for i in range(100):
                    for j in range(max_steps):
                        obs = test_states[i,j,:]
                        action = yhat.eval(feed_dict={x:obs[None,:]})
                        if action>=0.5:
                            action=1.0
                        else:
                            action=0.0
                        expert_action = test_expert_actions[i,j]
                        loss +=(action-expert_action)**2
                        #print(action,expert_action)

                loss_list.append(loss/100)
                np.save("loss_list",loss_list)

                total_crashes = 0
        
                for i in range(10):
                    noCrashes=0
                    state = env.reset()
                    done = False
                    running_reward=0
                    crashes = 0
                    t=0
                    while t<500:
                        action = yhat.eval(feed_dict={x:state[None,:]})
                        if action>=0.5:
                            action=1
                        else:
                            action=0

                        state,_,done,_ = env.step(action)

                        if done==True:
                            crashes += 1
                            state = env.reset()
                            done = False

                        t+=1

                    total_crashes += crashes

                crash_list.append(total_crashes/10)
                np.save("crash_list",crash_list)
            print('Optimization Finished!')
            # use trained MLP to perform

            max_steps = env.spec.max_episode_steps

            #I have collected each test trajectory such that its length is 500(max steps for CartPolev1) ex
            


    
            returns = []
            observations = []
            actions = []
            for i in range(num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = yhat.eval(feed_dict={x:obs[None, :]})
                    #we will have to limit the action to either take value 0 or 1 for the CartPole environment
                    if action>=0.5:
                        action=1
                    else:
                        action=0
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1   
                    if render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
                episode_wise_return.append(totalr)
           
    
            # expert labeling
            #On the new observations that are seen we query the expert for the actions that it would have taken
            #We then aggregate it with the existing dataset
            act_new = []
            for i_label in range(len(observations)):
                action_new = sample_action(observations[i_label][None, :],policy_fn)
                act_new.append(action_new)
            # record training size
            
            train_size = obs_data.shape[0]
            # data aggregation
            obs_data = np.concatenate((obs_data, np.array(observations)), axis=0)
            act_data = np.concatenate((act_data, np.array(act_new).reshape(-1,1)), axis=0)
            # record mean return & std
            save_mean = np.append(save_mean, np.mean(returns))
            save_std = np.append(save_std, np.std(returns))
            save_train_size = np.append(save_train_size, train_size)
            
    #final dagger results
    dagger_results = {'means': save_mean, 'stds': save_std, 'train_size': save_train_size,
                      'expert_mean':save_expert_mean, 'expert_std':save_expert_std}
    print('DAgger iterations finished!')
    print(dagger_results)

    file_name = "DaggerCartPole.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(save_mean, open_file)
    open_file.close()

    file_name2 = "DaggerCartPoleEPR.pkl"
    open_file = open(file_name2, "wb")
    pickle.dump(episode_wise_return, open_file)
    open_file.close()

    plt.figure()
    plt.title("Loss")
    plt.plot(loss_list)
    plt.show()

    plt.figure()
    plt.title("Crashes")
    plt.plot(crash_list)
    plt.show()
    
    

if __name__ == '__main__':
    main()