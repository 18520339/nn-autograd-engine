class WarmUpAndDecayScheduler:
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        if self.decay_steps <= 0: raise ValueError('Decay steps must be > 0')

    def __call__(self, step):
        warmup_lr = self.initial_learning_rate * (step / self.warmup_steps)
        decay_lr = self.initial_learning_rate * self.decay_rate ** ((step - self.warmup_steps) / self.decay_steps)
        return warmup_lr if step < self.warmup_steps else decay_lr