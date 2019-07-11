import torch
import shutil
import copy
from torch.nn.functional import gumbel_softmax
from utils import arglist
from agent.agent import Agent


class AcerAgent(Agent):
    def __init__(self, actor, critic, memory):
        """
        Acer learning for seperated actor & critic networks.
        """
        self.device = arglist.DEVICE
        self.nb_actions = arglist.NUM_ACTIONS

        self.iter = 0
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.DDPG.LEARNINGRATE)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.DDPG.LEARNINGRATE)

        self.memory = memory

        self.target_actor.eval()
        self.target_critic.eval()

    def process_batch(self):
        """
        Transforms numpy replays to torch tensor
        :return: dict of torch.tensor
        """
        replays = self.memory.sample(arglist.DDPG.BatchSize)

        # initialize batch experience
        batch = {'state0': {'minimap': [], 'screen': [], 'nonspatial': []},
                 'action': {'categorical': [], 'screen1': [], 'screen2': []},
                 'reward': [],
                 'state1': {'minimap': [], 'screen': [], 'nonspatial': []},
                 'terminal1': [],
                 }
        # append experience to list
        for e in replays:
            # state0
            for k, v in e.state0[0].items():
                batch['state0'][k].append(v)
            # action
            for k, v in e.action.items():
                batch['action'][k].append(v)
            # reward
            batch['reward'].append(e.reward)
            # state1
            for k, v in e.state1[0].items():
                batch['state1'][k].append(v)
            # terminal1
            batch['terminal1'].append(0. if e.terminal1 else 1.)

        # make torch tensor
        for key in batch.keys():
            if type(batch[key]) is dict:
                for subkey in batch[key]:
                    x = torch.tensor(batch[key][subkey], dtype=torch.float32)
                    batch[key][subkey] = x.to(self.device)
            else:
                x = torch.tensor(batch[key], dtype=torch.float32)
                x = torch.squeeze(x)
                batch[key] = x.to(self.device)

        return batch['state0'], batch['action'], batch['reward'], batch['state1'], batch['terminal1']

    def gumbel_softmax_hard(self, x):
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

    def optimize(self):
        """
        Conduct a single discrete learning iteration. Analogue of Algorithm 2 in the paper.
        """
        actor_critic = DiscreteActorCritic()
        actor_critic.copy_parameters_from(self.brain.actor_critic)

        _, _, _, next_states, _, _ = trajectory[-1]
        action_probabilities, action_values = actor_critic(Variable(next_states))
        retrace_action_value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1)

        for states, actions, rewards, _, done, exploration_probabilities in reversed(trajectory):
            action_probabilities, action_values = actor_critic(Variable(states))
            average_action_probabilities, _ = self.brain.average_actor_critic(Variable(states))
            value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1) * (1. - done)
            action_indices = Variable(actions.long())

            importance_weights = action_probabilities.data / exploration_probabilities

            naive_advantage = action_values.gather(-1, action_indices).data - value
            retrace_action_value = rewards + DISCOUNT_FACTOR * retrace_action_value * (1. - done)
            retrace_advantage = retrace_action_value - value

            # Actor
            actor_loss = - ACTOR_LOSS_WEIGHT * Variable(
                importance_weights.gather(-1, action_indices.data).clamp(max=TRUNCATION_PARAMETER) * retrace_advantage) \
                         * action_probabilities.gather(-1, action_indices).log()
            bias_correction = - ACTOR_LOSS_WEIGHT * Variable(
                (1 - TRUNCATION_PARAMETER / importance_weights).clamp(min=0.) *
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
            entropy_loss = ENTROPY_REGULARIZATION * (action_probabilities * action_probabilities.log()).sum(-1)
            entropy_loss.mean().backward(retain_graph=True)

            retrace_action_value = importance_weights.gather(-1, action_indices.data).clamp(max=1.) * \
                                   (retrace_action_value - action_values.gather(-1, action_indices).data) + value
        self.brain.actor_critic.copy_gradients_from(actor_critic)
        self.optimizer.step()
        self.brain.average_actor_critic.copy_parameters_from(self.brain.actor_critic, decay=TRUST_REGION_DECAY)

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
            scale = actor_gradient.mul(kullback_leibler_gradient).sum(-1).unsqueeze(-1) - TRUST_REGION_CONSTRAINT
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
