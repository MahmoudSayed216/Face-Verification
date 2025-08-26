import torch
from logger import Logger

class CheckpointHandler:
    def __init__(self, model, optim, save_every, output_path):
        self.past_value = -float('inf')
        self.model = model
        self.optim = optim
        self.save_every = save_every
        self.output_path = output_path
        # self.writer = writer
    
    def _save_data(self, current_value, epoch, _type):
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'entire_model': self.model,
            'optimizer_state_dict': self.optim.state_dict(),
            'metric': current_value
            },
            f'{self.output_path}/weights/{_type}.pth')

        Logger.CHECKPOINT(f"{_type} updated")
        # self.writer.add_line(f"{_type} updated")

    def _save_if_improved(self, current_value, epoch):
        if current_value > self.past_value:
            
            self._save_data(current_value, epoch, 'best')
            self.past_value = current_value

    def _save_periodic(self, current_value, epoch):
        if epoch%self.save_every == 0:
            self._save_data(current_value, epoch, 'periodic')

    def save(self, current_value, epoch):
        self._save_data(current_value, epoch, 'last')

    def save_model(self, current_value, epoch):
        self._save_if_improved(current_value, epoch)
        self._save_periodic(current_value, epoch)
        # self._save_anyway(current_value, epoch)