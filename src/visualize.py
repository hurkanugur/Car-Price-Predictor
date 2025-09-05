import matplotlib.pyplot as plt

class LossMonitor:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.train_line, = self.ax.plot([], [], label="Train Loss")
        self.val_line, = self.ax.plot([], [], label="Val Loss")
        self.train_losses, self.val_losses = [], []
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training & Validation Loss")
        self.ax.legend()
        self.ax.grid(True)

    def update(self, train_loss, val_loss=None):
        self.train_losses.append(train_loss)
        self.train_line.set_data(range(len(self.train_losses)), self.train_losses)
        if val_loss is not None:
            self.val_losses.append(val_loss)
            self.val_line.set_data(range(len(self.val_losses)), self.val_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()
