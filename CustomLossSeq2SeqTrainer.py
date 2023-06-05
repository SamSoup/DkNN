from transformers import Seq2SeqTrainer


class BartTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        # implement custom logic here
        custom_loss = ...
        return custom_loss
