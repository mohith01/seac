import torch
import random
import numpy as np

from sumtree import SumTree


class PrioritizedReplayBuffer(object):
    def __init__(self, size, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        # self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        # self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        # self.reward = torch.empty(buffer_size, dtype=torch.float)
        # self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        # self.done = torch.empty(buffer_size, dtype=torch.int)

        # self._maxsize = size
        # self._next_idx = 0

        self.count = 0
        self.real_size = 0
        self.size = size
        # print(self.size)
        self._storage = [1]*size


    def __len__(self):
      return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        # print(20*"--")
        # print("Count:",self.count)
        data = (state, action, reward, next_state, done)

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)
        

        # if self._next_idx >= len(self._storage):
        #   self._storage.append(data)
        # else:
        # print(self.count, self._storage, self.size)
        self._storage[self.count] = data
        # print(len(self._storage))

        # # store transition in the buffer
        # self.state[self.count] = torch.as_tensor(state)
        # self.action[self.count] = torch.as_tensor(action)
        # self.reward[self.count] = torch.as_tensor(reward)
        # self.next_state[self.count] = torch.as_tensor(next_state)
        # self.done[self.count] = torch.as_tensor(done)
        # update counters

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample_new(self, batch_size):
        # assert self.real_size >= batch_size, "buffer contains less samples than batch size"
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            # print(30*"+++")
            # if 
            # print(type(priority), sample_idx, cumsum, a,b, self.count,len(self._storage))
            if isinstance(priority,np.ndarray):
                priority = priority.item()
                # print(priority)



            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
            # print("Priorities",priorities)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()
        # print(23*"+++")
        # print("Sample_idxs",sample_idxs, self.count )

        sample_idxs = [i for i in sample_idxs if i!=None]

        # print(sample_idxs)
        x,y,z,w,u = self._encode_sample(sample_idxs)
        # print(type(weights))
        # print(type(tree_idxs))
        # print(self._storage)
        return x,y,z,w,u, weights, tree_idxs

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def update_priorities(self, data_idxs, priorities):
        # print(type(data_idxs), type(priorities))
        # print(isinstance(priorities, torch.Tensor))
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class MARLPriorityBuffer(object):
  def __init__(self, size, num_agents):
    self.size = size
    self.num_agents = num_agents
    self.buffers = [PrioritizedReplayBuffer(size) for _ in range(num_agents)]

  def __len__(self):
    return max(len(self.buffers[0]),len(self.buffers[1]))

  def add(self, observations, actions, rewards, next_observations, dones):
    for i, (o, a, r, no, d) in enumerate(zip(observations, actions, rewards, next_observations, dones)):
      self.buffers[i].add(o, a, r, no, d)

  def sample(self, batch_size, agent_i):
    cast = lambda x: torch.from_numpy(x).float()
    obs, act, rew, next_obs, done, weights, tree_idx = self.buffers[agent_i].sample_new(batch_size)
    obs = cast(obs).squeeze()
    act = cast(act)
    rew = cast(rew)
    next_obs = cast(next_obs).squeeze()
    done = cast(done)
    # print(type(weights), type(tree_idx))
    return obs, act, rew, next_obs, done, weights, tree_idx
    
  def update_batch(self, agent_i,data_idxs, priorities):
    self.buffers[agent_i].update_priorities(data_idxs, priorities)


  def sample_shared(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        batch_size_each = batch_size // self.num_agents
        obs = []
        act = []
        rew = []
        next_obs = []
        done = []
        for agent_i in range(self.num_agents):
            o, a, r, no, d = self.buffers[agent_i].sample(batch_size_each)
            obs.append(o)
            act.append(a)
            rew.append(r)
            next_obs.append(no)
            done.append(d)
        cast = lambda x: torch.from_numpy(x).float()
        obs = cast(np.vstack(obs)).squeeze()
        act = cast(np.vstack(act))
        rew = cast(np.vstack(rew))
        next_obs = cast(np.vstack(next_obs)).squeeze()
        done = cast(np.vstack(done))
        return obs, act, rew, next_obs, done


