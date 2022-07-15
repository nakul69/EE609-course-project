import numpy as np
from random import sample
#import tensorflow as tf
import numpy as np
import tf_util
from tensorflow import keras
import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#D_prev is a Nx2 array of obs and action

# policy_prev = expert


####################################################################################################
class policy_NN :
    def __init__(self, input_shape,output_shape):

        self.obs_dim = input_shape;
        self.act_dim = output_shape;
    #self.model.add(tf.keras.layers.Dense(input_shape[0], activation='tanh'));
        self.x = tf.placeholder(tf.float32, shape=[None, self.obs_dim])
        self.yhot = tf.placeholder(tf.float32, shape=[None, self.act_dim])
        self.h1 = tf.layers.dense(inputs=self.x, units=128, activation=tf.nn.relu)
        self.h2 = tf.layers.dense(inputs=self.h1, units=64, activation=tf.nn.relu)
        self.h3 = tf.layers.dense(inputs=self.h2, units=32, activation=tf.nn.relu)
        self.probhat = tf.layers.dense(inputs=self.h3, units=2,activation=tf.nn.softmax)
        self.yhat = tf.layers.dense(inputs=self.h3, units=self.act_dim, activation=None)
        #self.model = tf.keras.Model(inputs=self.x,outputs=self.yhat)
        self.loss_l2 = tf.reduce_mean(tf.square(self.yhot - self.yhat))
    
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss_l2)


####################### Dataset Generation #####################################################

def sample_expert_action(observation,expert_policy):
    observation = observation.reshape(1, -1)
    logits = expert_policy(observation)
    action = tf.random.categorical(logits, 1)
    #action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return action.eval()[0,0]

def dataset_generation(m,alpha,D_obs_prev,D_act_prev,T,policy_fn,policy_fn_est,policy_fn_idx,policy_fn_est_idx):

    N = D_obs_prev.shape[0]
    envname = 'CartPole-v1'
    render = 0
    num_rollouts = 1
    observations = []
    actions = []
    while(len(actions)<=m):
        t = np.random.randint(1,T+1);
        import gym
        env = gym.make(envname)
        max_steps = T

        returns = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                if(steps!=t):
                    if(policy_fn_idx==1):
                        action = sample_expert_action(obs,policy_fn)
                        # action using expert policy poli_fn
                        observations.append(obs)
                        actions.append(action)
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1
                    else:
                        action = policy_fn.yhat.eval(feed_dict={policy_fn.x:obs.reshape(1,-1)})    
                        if action>=0.5:
                            action=1
                        else:
                            action=0
                        observations.append(obs)
                        actions.append(action)
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1    
                else :
                    if(policy_fn_est_idx==1):
                        action = sample_expert_action(obs,policy_fn_est)
                        # action using expert policy policy_fn
                        observations.append(obs)
                        actions.append(action)
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1
                    else : 
                        action = policy_fn_est.yhat.eval(feed_dict={policy_fn_est.x:obs.reshape(1,-1)})    
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

        #print('returns', returns)
        #print('mean return', np.mean(returns))
        #print('std of return', np.std(returns))

        # pass observations, actions to imitation learning
        obs_data = np.array(observations)
        act_data = np.array(actions)
        act_data=act_data.reshape(-1,1)

        save_mean = np.mean(returns)
        save_std = np.std(returns)

    return obs_data,act_data,save_mean,save_std
###############################################################################################################

#===========================================================================
# generate expert data
#===========================================================================
# param
envname = 'CartPole-v1'
render = 0
num_rollouts = 1
dagger_iters=3
# policy_fn contains expert policy
policy_fn = keras.models.load_model("PPOexpert")
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
            action = sample_expert_action(obs,policy_fn)
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

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    # pass observations, actions to imitation learning
    obs_data = np.array(observations)
    print(np.shape(obs_data))
    act_data = np.array(actions)
    act_data=act_data.reshape(-1,1)

save_expert_mean = np.mean(returns)
save_expert_std = np.std(returns)

########################################################################################################


################## SMILE ################################################################################
num_rollouts_mean = []
policy_exec = []
#params not sure
N = 15
smile_iter = N
m = 500
D_obs_prev = obs_data
D_act_prev = act_data
T = 100
expert_policy = keras.models.load_model("PPOexpert")
policy_list = [keras.models.load_model("PPOexpert")]
policy_num=[1]
alpha = 0.1

obs_dim = obs_data.shape[1]
act_dim = act_data.shape[1]

for i in range(1, N+1):
    policy_list += [policy_NN(obs_dim, act_dim)]
    policy_num += [0]

policy_curr = policy_list[0]
policy_curr_idx = 1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # record return and std for plotting
    save_mean = []
    save_std = []
    save_train_size = []

    # loop for dagger alg
    for i_smile in range(0,smile_iter):

        print('Smile iteration ', i_smile)
        prob_lst = [pow((1-alpha),i_smile+1)]
        for j in range(0,i_smile+1):
            prob_lst+= [alpha*pow(1-alpha,j)]

        policy_prev = policy_curr
        policy_prev_idx = policy_curr_idx

        temp = np.random.choice(i_smile+2,p=prob_lst)
    
        policy_curr = policy_list[temp]
        policy_curr_idx = policy_num[temp]
    
        if(i_smile==0):
            obsi_data = D_obs_prev
            acti_data = D_act_prev

        # train a policy by fitting the MLP
        batch_size = 50
        if(policy_curr_idx==0):
            for step in range(10000):
                batch_i = np.random.randint(0, obsi_data.shape[0], size=batch_size)
                policy_curr.train_step.run(feed_dict={policy_curr.x: obsi_data[batch_i,], policy_curr.yhot: acti_data[batch_i,]})
                if (step % 1000 == 0):
                    print('opmization step ', step)
                    print('obj value is ', policy_curr.loss_l2.eval(feed_dict={policy_curr.x:obsi_data, policy_curr.yhot:acti_data}))
        print('Optimization Finished!')
        # use trained MLP to perform
        max_steps = env.spec.max_episode_steps
        num_rollouts_test=100
        observations_test = []
        actions_test = []
        returns_test = []

        D_obs_prev = obsi_data
        D_act_prev = acti_data

        obsi_data,acti_data,save_mean,save_std = dataset_generation(m,alpha,D_obs_prev,D_act_prev,T,policy_curr,policy_prev,policy_curr_idx,policy_prev_idx)

        if(policy_curr_idx==0):
            policy_exec.append(temp)
            for i in range(num_rollouts_test):
                print('test iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = policy_curr.yhat.eval(feed_dict={policy_curr.x:obs[None, :]})
                    if action>=0.5:
                       action=1
                    else:
                       action=0
                    observations_test.append(obs)
                    actions_test.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1   
                    if render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                    returns_test.append(totalr)
                print('mean return', np.mean(returns_test))
                num_rollouts_mean.append(np.mean(returns_test))
                #print('std of return', np.std(returns_test))
        
Smile_results = {'means': save_mean, 'stds': save_std}
                #'expert_mean':save_expert_mean, 'expert_std':save_expert_std}
print('Smile iterations finished!')
print(Smile_results)
policy_exec_index = np.array(policy_exec)
num_rollouts_mean_array = np.array(num_rollouts_mean)
np.save('policy_exec_index.npy',policy_exec_index)
np.save('num_rollouts_mean_array.npy',num_rollouts_mean_array)
