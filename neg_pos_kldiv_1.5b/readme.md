will be training on /DeepSeek-R1-Distill-Qwen-1.5B
still modifying kl divergence
try these values: 0.001, 0.01, 0.05, 0.1
and negative of the above values
and just try kl divergence of 5 to see what happens

later - try making this true: algorithm.use_kl_in_reward=False \
you will also need to change the kl coef and other parameters for the above^ kl reward

maybe also try playing around with this parameter:
actor_rollout_ref.actor.kl_loss_type - k1, mse, abs, k3, etc...

Essentially, as discussed in the 2 paragraphs above, whether modifying the kl in the reward or the kl in the loss, you can play around with both of these parameters to see how it affects results:
  algorithm.kl_penalty=low_var_kl \         (kl_loss_type and kl_penalty are basically the same just that one is for kl reward the other is for kl loss)
  algorithm.kl_ctrl.kl_coef=0.001 \

reference for kl divergence control in verl:
https://verl.readthedocs.io/en/latest/algo/ppo.html

made a bunch of changes to the ppo script in this iteration
USE THIS PPO SCRIPT GOING FORWARD!

made some changes to setup.sh in this iteration
USE THIS SETUP.SH GOING FORWARD!