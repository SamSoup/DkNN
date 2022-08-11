"""
This File is intended to take a pretrained model through an inference set,
saving its logits for analytic purposes.

This should really be done at evaluation for Trainer, but I am not sure if there's
a particularly easy way to add a Callback that only logs at evalution time
"""
