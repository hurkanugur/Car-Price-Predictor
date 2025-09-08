import matplotlib.pyplot as plt

class LossMonitor:
    def __init__(self):
        # Enable interactive mode for live updating of the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.train_line, = self.ax.plot([], [], label="Train Loss")
        self.val_line, = self.ax.plot([], [], label="Val Loss")
        
        # Store epochs and corresponding loss values for plotting
        self.train_epochs, self.train_losses = [], []
        self.val_epochs, self.val_losses = [], []

        # Configure plot labels, title, legend, and grid
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training & Validation Loss")
        self.ax.legend()
        self.ax.grid(True)

    def update(self, epoch, train_loss, val_loss=None):
        # Append current epoch and train loss for plotting
        self.train_epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_line.set_data(self.train_epochs, self.train_losses)

        # Append validation loss only if it is provided
        if val_loss is not None:
            self.val_epochs.append(epoch)
            self.val_losses.append(val_loss)
            self.val_line.set_data(self.val_epochs, self.val_losses)

        # Rescale axes and redraw the plot
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)  # brief pause for interactive update

    def close(self):
        # Disable interactive mode and display the final plot
        plt.ioff()
        plt.show()
