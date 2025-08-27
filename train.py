import configs
import os
import torch
import torchvision.transforms as T
import timeit
from dataset import FacesDataset
from model import CringeNet
from tripletLoss import TripletLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from TripletsFormer import compute_distance_matrix, form_triplets, get_a_n_p_pairs
from logger import Logger 
from metrics import compute_accuracy_score, compute_triplet_loss
from checkpointer import CheckpointHandler


def get_loaders():
    test_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FacesDataset(configs.DATA_PATH, split="train", per_subject_samples=configs.PER_SUBJECT_SAMPLES, transforms=train_transform)
    test_dataset = FacesDataset(configs.DATA_PATH, split="val", per_subject_samples=configs.PER_SUBJECT_SAMPLES, transforms=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size = configs.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = configs.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader









def train(session_path, train_loader, test_loader):
    #TODO: tensorboard
    tensorboard_logs_path = f"{session_path}/runs/experiment1"
    writer = SummaryWriter(tensorboard_logs_path)
    model = CringeNet(configs.EMBEDDING_DIM).to(configs.DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=configs.LEARNING_RATE)
    checkpointer = CheckpointHandler(model, optim, configs.SAVE_EVERY, session_path)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode=configs.MODE, factor=configs.LR_REDUCTION_FACTOR, patience=configs.PATIENCE, min_lr=configs.MIN_LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, mode=configs.MODE, factor=configs.LR_REDUCTION_FACTOR, patience=configs.PATIENCE, min_lr=configs.MIN_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=configs.T_MAX, eta_min=configs.MIN_LR)
    loss_fn = TripletLoss(configs.ALPHA)
    print(configs.DEVICE)
    for epoch in range(configs.EPOCHS):
        print(f"EPOCH: {epoch}")
        model.train()
        for i, input in enumerate(train_loader):
            optim.zero_grad()
            input = input.to(configs.DEVICE)
            embeddings = model(input)
            B, N, L = embeddings.shape
            embeddings = embeddings.view(B * N, L)  # (n, d)
            distance_matrix = compute_distance_matrix(embeddings)
            triplets = get_a_n_p_pairs(distance_matrix)
            anc, ps, ns = form_triplets(embeddings, triplets)
            loss = loss_fn(anc, ps, ns)
            writer.add_scalar("Loss/train", loss.item(), epoch)
            loss.backward()
            optim.step()
            if i%configs.PRINT_EVERY ==0:
                print("train batch loss: ", loss.item())

        test_loss = compute_triplet_loss(model, test_loader, loss_fn)
        accuracy_scores =  compute_accuracy_score(model, test_loader)
        print("test loss: ",test_loss)
        i = 0
        for th, acc in accuracy_scores:
            print(f"accuracy @{th:.2f}: ", acc, " [dynamic threhsold]" if i == 0 else "")
            i+=1
        checkpointer.save_model(accuracy_scores[0][1], epoch)
        scheduler.step()
        print("LEARNING RATE: ", scheduler.get_last_lr())
        writer.add_scalar("Loss/test", test_loss)
        writer.add_scalar("accuracy/test", accuracy_scores[0][1])
        print("*"*30)
    checkpointer.save(accuracy_scores[0][1], epoch)
    writer.close()

    



def prepare_training_session_dir():
    os.makedirs(os.path.join(configs.OUTPUT_DIR, "output"), exist_ok=True)
    session_folder_path = os.path.join(configs.OUTPUT_DIR, "output")
    current_attemps_count = len(os.listdir(session_folder_path))
    current_training_attempt_index = current_attemps_count+1
    final_path = os.path.join(session_folder_path,"session"+str(current_training_attempt_index)) 
    os.mkdir(final_path)
    return final_path



if __name__ == "__main__":
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=350, threshold=10_000)
    Logger.debug_active = configs.DEBUG
    Logger.checkpoint_active = configs.CHECKPOINT
    Logger.log_active = configs.LOG
    path = prepare_training_session_dir()
    train_loader, test_loader = get_loaders()
    train(path, train_loader, test_loader)