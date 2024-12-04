import numpy as np
from tqdm import tqdm
from rich.table import Table
from rich.console import Console


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.history = {}

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def compile(self, loss, metrics={}):
        self.loss_func = loss
        self.metric_funcs = metrics
        self.history = {'epoch': [], 'step': [], 'loss': [], **{name: [] for name in metrics}}

    def summary(self):
        table = Table(title='Model Summary', show_header=True, header_style='bold magenta')
        total_params = 0

        # Define columns with alignment and style
        table.add_column('Layer', justify='left', style='cyan', no_wrap=True)
        table.add_column('Output Shape', justify='center', style='green')
        table.add_column('Param #', justify='right', style='yellow')

        # Populate table rows with each layer's information
        for layer in self.layers:
            num_params = len(layer.parameters())
            table.add_row(layer.name, str(layer.output_size), f'{num_params:,}')
            total_params += num_params

        console = Console()
        console.print(table)
        console.print(f'[bold green]Total params:[/bold green] {total_params:,}')


    def train(self, X_train, y_train, epochs=100, learning_rate=0.01, batch_size=1, clip_value=None):
        # Custom tqdm format with emojis and more details
        tqdm_format = "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [⏳{elapsed}<{remaining}]{postfix}"
        data_size, step = len(X_train), 1
        steps_per_epoch = data_size // batch_size
        lr_scheduler = learning_rate if callable(learning_rate) else None

        for epoch in range(epochs):
            self.history['epoch'].append(epoch + 1)
            with tqdm(total=steps_per_epoch, desc=f'🚀 Epoch {epoch + 1}/{epochs}', bar_format=tqdm_format, ncols=500) as pbar:
                for i in range(0, data_size, batch_size):
                    last_idx = min(i + batch_size, data_size) # Prevents index out of range when batch_size doesn't divide the data size
                    X_batch, y_batch = X_train[i:last_idx], y_train[i:last_idx]

                    predictions = [self(inputs) for inputs in X_batch]
                    loss = self.loss_func(y_batch, predictions)
                    loss.backward()

                    learning_rate = lr_scheduler(step) if lr_scheduler else learning_rate
                    for param in self.parameters():
                        # Clip gradients to ensure gradients stay within a manageable range, especially for ReLU
                        # For example: np.clip(500, -10, 10) = 10
                        if clip_value: param.gradient = np.clip(param.gradient, -clip_value, clip_value)
                        param.data -= learning_rate * param.gradient
                        param.gradient = 0.0

                    metrics = {name: func(y_batch, predictions) for name, func in self.metric_funcs.items()}
                    results = {'loss': loss.data, **metrics, **({'learning_rate': learning_rate} if lr_scheduler else {})}

                    self.history['step'].append(step)
                    for key, value in results.items():
                        self.history.setdefault(key, []).append(value)

                    pbar.set_postfix(results)
                    pbar.update(1)
                    step += 1


    def predict(self, X):
        return np.array([
            [output.data for output in self(inputs)]
            if self.layers[-1].output_size > 1 else self(inputs).data for inputs in X
        ])