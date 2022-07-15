#For the implementation of behavorial cloning we will just reduce the number of dagger iterations to 1 and 
#remove the data set aggregation step from the dagger implementation


from random import sample
import pickle
#import tensorflow as tf
import numpy as np
import tf_util
from tensorflow import keras
import gym
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
    # param
    envname = 'CartPole-v1'
    render = 0
    num_rollouts = 50
    dagger_iters=1
    # policy_fn contains expert policy
    policy_fn = keras.models.load_model("PPOexpertCartPole")
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(envname)
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
    
    # architecture of the MLP policy function
    x = tf.placeholder(tf.float32, shape=[None, obs_dim])
    yhot = tf.placeholder(tf.float32, shape=[None, act_dim])
    
    h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
    yhat = tf.layers.dense(inputs=h3, units=act_dim, activation=None)
    
    loss_l2 = tf.reduce_mean(tf.square(yhot - yhat))
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
            print('Behavorial Cloning iteration')
            # train a policy by fitting the MLP
            batch_size = 25
            for step in range(1000):
                batch_i = np.random.randint(0, obs_data.shape[0], size=batch_size)
                train_step.run(feed_dict={x: obs_data[batch_i, ], yhot: act_data[batch_i, ]})
                if (step % 10 == 0):
                    print('opmization step ', step)
                    print('obj value is ', loss_l2.eval(feed_dict={x:obs_data, yhot:act_data}))
            print('Optimization Finished!')
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
            
            train_size = obs_data.shape[0]
            save_mean = np.append(save_mean, np.mean(returns))
            save_std = np.append(save_std, np.std(returns))
            save_train_size = np.append(save_train_size, train_size)
            
    bc_results = {'means': returns, 'stds': save_std, 'train_size': save_train_size,
                      'expert_mean':save_expert_mean, 'expert_std':save_expert_std}
    print('Behavorial Cloning finished!')
    print(bc_results)

    file_name = "BCCartPole.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(returns, open_file)
    open_file.close()

    file_name2 = "BCCartPoleEPR.pkl"
    open_file = open(file_name2, "wb")
    pickle.dump(episode_wise_return, open_file)
    open_file.close()

if __name__ == '__main__':
    main()