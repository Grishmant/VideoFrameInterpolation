from torchvision.datasets import UCF101
import av
import torch
from torchvision import transforms
import torch.nn.functional as F
# NOT IN USE, USING READ_FRAMES.PY INSTEAD FOR FRAME EXTRACTION

ucf_data_dir = "C:\\Users\\grish\\Desktop\\VFI\\UCF101\\UCF-101"
ucf_label_dir = "C:\\Users\\grish\\Desktop\\VFI\\ucftrainlist\\ucfTrainTestlist"
frames_per_clip = 5
step_between_clips = 5
batch_size = 32

tfs = transforms.Compose([
    transforms.Lambda(lambda x: x / 255),
    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    transforms.Lambda(lambda x: F.interpolate(x, (240, 320)))
])

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)




# test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=False, transform=tfs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

if __name__ == '__main__':

    train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=True, transform=tfs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print(len(train_dataset))