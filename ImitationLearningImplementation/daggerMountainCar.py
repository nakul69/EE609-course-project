from random import sample
#import tensorflow as tf
import numpy as np
import tf_util
from tensorflow import keras
import gym
import pickle
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
    envname = 'MountainCar-v0'
    render = 0
    num_rollouts = 10
    dagger_iters=10
    # policy_fn contains expert policy(loaded using keras)
    policy_fn = keras.models.load_model("PPOexpertMountainCar")
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(envname)
        max_steps = env.spec.max_episode_steps
    
        returns = []
        observations = []
        actions = []
        #First we collect some data using the expert policy provided to us
        for i in range(num_rollouts):
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
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
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
    
    #===========================================================================
    # set up the network structure for the imitation learning policy function
    #===========================================================================
    # dim for input/output
    obs_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]
    
     # architecture of the neural network policy function. We use two hidden layers having 64 and 32 units repectively
    x = tf.placeholder(tf.float32, shape=[None, obs_dim])
    yhot = tf.placeholder(tf.float32, shape=[None, act_dim])
    
    h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
    yhat = tf.layers.dense(inputs=h3, units=act_dim, activation=None)
    
    loss_l2 = tf.reduce_mean(tf.square(yhot - yhat))                    #loss function defined as the square loss
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
        episode_wise_return=[]
        # loop for dagger alg
        for i_dagger in range(dagger_iters):
            print('DAgger iteration ', i_dagger)
            # train a policy by fitting the MLP
            batch_size = 25
            for step in range(1000):
                batch_i = np.random.randint(0, obs_data.shape[0], size=batch_size)
                train_step.run(feed_dict={x: obs_data[batch_i, ], yhot: act_data[batch_i, ]})
            #     if (step % 10 == 0):
            #         print('opmization step ', step)
            #         print('obj value is ', loss_l2.eval(feed_dict={x:obs_data, yhot:act_data}))
            # print('Optimization Finished!')
            # use trained MLP to perform
            max_steps = env.spec.max_episode_steps
    
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
                    action = np.round(action)
                    action = int(action[0][0])
                    #we will have to limit the action to either take value 0, 1 or 2 for the MountainCar environment
                    if action>2:
                        action=2;
                    if action<0:
                        action=0;
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
                episode_wise_return.append(totalr)
            # print('mean return', np.mean(returns))
            # print('std of return', np.std(returns))
    
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
            
    dagger_results = {'means': save_mean, 'stds': save_std, 'train_size': save_train_size,
                      'expert_mean':save_expert_mean, 'expert_std':save_expert_std}
    print('DAgger iterations finished!')
    print(dagger_results)
    file_name = "DaggerMountainCar.pkl"
    file_name2 = "DaggerMountainCarEPR.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(save_mean, open_file)
    open_file.close()

    open_file = open(file_name2, "wb")
    pickle.dump(episode_wise_return, open_file)
    open_file.close()


if __name__ == '__main__':
    main()