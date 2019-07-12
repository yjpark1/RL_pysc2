import torch
import shutil
import copy
import numpy as np
from pysc2.lib import actions
from torch.nn.functional import gumbel_softmax
from utils import arglist
from agent.agent import Agent
from networks.acnetworks_newchallenge import FullyConvNet


class AcerAgent(Agent):
    def __init__(self, ActorCritic, memory):
        """
        Acer learning for seperated actor & critic networks.
        """
        self.device = arglist.DEVICE
        self.nb_actions = arglist.NUM_ACTIONS

        self.iter = 0
        self.ActorCritic = ActorCritic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.ActorCritic.parameters(), arglist.ACER.LEARNINGRATE)

        self.memory = memory

    def select_action(self, obs, valid_actions):
        '''
        from logit to pysc2 actions
        :param logits: {'categorical': [], 'screen1': [], 'screen2': []}
        :return: FunctionCall form of action
        '''
        obs_torch = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for o in obs:
            x = obs[o].astype('float32')
            x = np.expand_dims(x, 0)
            obs_torch[o] = torch.from_numpy(x).to(arglist.DEVICE)

        logits = self.actor(obs_torch)
        logits[0] = self._mask_unavailable_actions(logits['categorical'], valid_actions)
        tau = 1.0
        function_id = gumbel_softmax(logits['categorical'], tau=tau, hard=True)
        function_id = function_id.argmax().item()

        # select an action until it is valid.
        is_valid_action = self._test_valid_action(function_id, valid_actions)
        while not is_valid_action:
            tau *= 10
            function_id = gumbel_softmax(logits['categorical'], tau=tau, hard=True)
            function_id = function_id.argmax().item()
            is_valid_action = self._test_valid_action(function_id, valid_actions)

        pos_screen1 = gumbel_softmax(logits['screen1'].view(1, -1), hard=True).argmax().item()
        pos_screen2 = gumbel_softmax(logits['screen2'].view(1, -1), hard=True).argmax().item()

        pos = [[int(pos_screen1 % arglist.FEAT2DSIZE), int(pos_screen1 // arglist.FEAT2DSIZE)],
               [int(pos_screen2 % arglist.FEAT2DSIZE), int(pos_screen2 // arglist.FEAT2DSIZE)]]  # (x, y)

        args = []
        cnt = 0
        for arg in actions.FUNCTIONS[function_id].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append(pos[cnt])
                cnt += 1
            else:
                args.append([0])

        action = actions.FunctionCall(function_id, args)
        return action, logits

    def process_batch(self, mode):
        """
        Transforms numpy replays to torch tensor
        :return: dict of torch.tensor
        """
        if mode == 'on-policy':
            replays = self.memory.sample(batch_size=1, batch_idxs=[self.memory.nb_entries - 1])
        else:
            replays = self.memory.sample(arglist.ACER.BatchSize)

        # make torch tensor
        for i, ep in enumerate(replays):
            for step in ep:
                for key in step.observation.keys():
                    x = torch.as_tensor(step.observation[key], dtype=torch.float32)
                    step.observation[key] = x.to(self.device)

                for key in step.action.keys():
                    x = torch.as_tensor(step.action[key], dtype=torch.float32)
                    step.action[key] = x.to(self.device)

                for key in step.policy.keys():
                    x = torch.as_tensor(step.policy[key], dtype=torch.float32)
                    step.policy[key] = x.to(self.device)

                x = torch.as_tensor(step.reward, dtype=torch.float32)
                step = step._replace(reward=x.to(self.device))

                x = torch.as_tensor(step.terminal, dtype=torch.float32)
                step = step._replace(terminal=x.to(self.device))
            replays[i] = step

        return replays

    @staticmethod
    def gumbel_softmax_hard(x):
        shape = x.shape
        if len(shape) == 4:
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(shape[0], -1)
            y = torch.nn.functional.gumbel_softmax(x_reshape, hard=True, dim=-1)
            # We have to reshape Y
            y = y.contiguous().view(shape)
        else:
            y = torch.nn.functional.gumbel_softmax(x, hard=True, dim=-1)

        return y

    def optimize(self, mode):
        """
        Conduct a single discrete learning iteration. Analogue of Algorithm 2 in the paper.
        """
        trajectory = self.process_batch(mode)

        ActorCritic = FullyConvNet()
        ActorCritic.copy_parameters_from(self.ActorCritic)

        # on-policy
        trajectory = replays

        for s in trajectory:
            if mode == 'on-policy':
                rho = 1
            elif mode == 'off-policy':
                rho = policies[i].detach() / old_policies[i]
            else:
                NotImplementedError()

        ###########
        _, _, _, next_states, _, _ = trajectory[-1]
        action_probabilities, action_values = ActorCritic(next_states)
        retrace_action_value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1)

        for states, actions, rewards, _, done, exploration_probabilities in reversed(trajectory):
            action_probabilities, action_values = ActorCritic(Variable(states))
            average_action_probabilities, _ = self.brain.average_actor_critic(Variable(states))
            value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1) * (1. - done)
            action_indices = Variable(actions.long())

            importance_weights = action_probabilities.data / exploration_probabilities

            naive_advantage = action_values.gather(-1, action_indices).data - value
            retrace_action_value = rewards + arglist.ACER.DISCOUNT_FACTOR * retrace_action_value * (1. - done)
            retrace_advantage = retrace_action_value - value

            # Actor
            actor_loss = - arglist.ACER.ACTOR_LOSS_WEIGHT * Variable(
                importance_weights.gather(-1, action_indices.data).clamp(max=arglist.ACER.TRUNCATION_PARAMETER) * retrace_advantage) \
                         * action_probabilities.gather(-1, action_indices).log()
            bias_correction = - arglist.ACER.ACTOR_LOSS_WEIGHT * Variable(
                (1 - arglist.ACER.TRUNCATION_PARAMETER / importance_weights).clamp(min=0.) *
                naive_advantage * action_probabilities.data) * action_probabilities.log()
            actor_loss += bias_correction.sum(-1).unsqueeze(-1)
            actor_gradients = torch.autograd.grad(actor_loss.mean(), action_probabilities, retain_graph=True)
            actor_gradients = self.discrete_trust_region_update(actor_gradients, action_probabilities,
                                                                Variable(average_action_probabilities.data))
            action_probabilities.backward(actor_gradients, retain_graph=True)

            # Critic
            critic_loss = (action_values.gather(-1, action_indices) - Variable(retrace_action_value)).pow(2)
            critic_loss.mean().backward(retain_graph=True)

            # Entropy
            entropy_loss = arglist.ACER.ENTROPY_REGULARIZATION * (action_probabilities * action_probabilities.log()).sum(-1)
            entropy_loss.mean().backward(retain_graph=True)

            retrace_action_value = importance_weights.gather(-1, action_indices.data).clamp(max=1.) * \
                                   (retrace_action_value - action_values.gather(-1, action_indices).data) + value
        self.ActorCritic.copy_gradients_from(ActorCritic)
        self.optimizer.step()
        self.brain.average_actor_critic.copy_parameters_from(self.ActorCritic, decay=arglist.ACER.TRUST_REGION_DECAY)

        return loss_actor, loss_critic

    @staticmethod
    def discrete_trust_region_update(actor_gradients, action_probabilities, average_action_probabilities):
        """
        Update the actor gradients so that they satisfy a linearized KL constraint with respect
        to the average actor-critic network. See Section 3.3 of the paper for details.

        Parameters
        ----------
        actor_gradients : tuple of torch.Tensor's
            The original gradients.
        action_probabilities
            The action probabilities according to the current actor-critic network.
        average_action_probabilities
            The action probabilities according to the average actor-critic network.

        Returns
        -------
        tuple of torch.Tensor's
            The updated gradients.
        """
        negative_kullback_leibler = - ((average_action_probabilities.log() - action_probabilities.log())
                                       * average_action_probabilities).sum(-1)
        kullback_leibler_gradients = torch.autograd.grad(negative_kullback_leibler.mean(),
                                                         action_probabilities, retain_graph=True)
        updated_actor_gradients = []
        for actor_gradient, kullback_leibler_gradient in zip(actor_gradients, kullback_leibler_gradients):
            scale = actor_gradient.mul(kullback_leibler_gradient).sum(-1).unsqueeze(-1) - arglist.ACER.TRUST_REGION_CONSTRAINT
            scale = torch.div(scale, actor_gradient.mul(actor_gradient).sum(-1).unsqueeze(-1)).clamp(min=0.)
            updated_actor_gradients.append(actor_gradient - scale * kullback_leibler_gradient)
        return updated_actor_gradients

    def soft_update(self, target, source, tau):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_models(self, fname):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), str(fname) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), str(fname) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, fname):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(str(fname) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(str(fname) + '_critic.pt'))
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')

    def save_training_checkpoint(self, state, is_best, episode_count):
        """
        Saves the models, with all training parameters intact
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = str(episode_count) + 'checkpoint.path.rar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
