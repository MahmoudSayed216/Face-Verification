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
from distancesComputer import compute_distance_matrix
from logger import Logger 
import webbrowser


def get_loaders():
    test_transforms = T.Compose([
        T.Resize(256),              # resize smaller side to 256
        T.CenterCrop(224),          # crop to 224×224
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
    ])

    train_transform = T.Compose([
        T.Resize(256),                   # resize while keeping aspect ratio
        T.RandomCrop(224),               # small spatial jitter
        T.RandomHorizontalFlip(p=0.5),   # okay for faces (symmetry)
        # Optional, but light jitter only:
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = FacesDataset(configs.DATA_PATH, 
                                 split="train", 
                                 per_subject_samples=configs.PER_SUBJECT_SAMPLES, 
                                 transforms=train_transform)
    
    test_dataset = FacesDataset(configs.DATA_PATH, 
                                 split="val", 
                                 per_subject_samples=configs.PER_SUBJECT_SAMPLES, 
                                 transforms=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size = configs.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = configs.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader




def compute_loss(model, loader, loss_fn):
    model.eval()
    losses = []
    for i, input in enumerate(loader):
        input = input.to(configs.DEVICE)
        embeddings = model(input)
        B, N, L = embeddings.shape
        embeddings = embeddings.view(B * N, L)  # (n, d)
        distance_matrix = compute_distance_matrix(embeddings)
        pairs = get_n_p_pairs(distance_matrix)
        ps, ns = form_triplets(embeddings, pairs)
        loss = loss_fn(embeddings, ps, ns)
    
        losses.append(loss.item())

    return sum(losses)/len(losses)




def get_n_p_pairs(distance_matrix):
    n_p_pairs = []
    for i in range(len(distance_matrix)):
        row :torch.Tensor = distance_matrix[i].clone()
        lower_bound = i - i%configs.PER_SUBJECT_SAMPLES
        upper_bound = lower_bound + configs.PER_SUBJECT_SAMPLES

        row[lower_bound:upper_bound] = float('inf')
        neg_idx = row.argmin(0)
        pos_idx = i-1 if i%configs.PER_SUBJECT_SAMPLES != 0 else 1
        n_p_pairs.append((pos_idx, neg_idx))
    return n_p_pairs        

def form_triplets(embeddings, n_p_pairs):
    # anchors = embeddings
    positives = []
    negatives = []
    for embedding, pair in zip(embeddings, n_p_pairs):
        p, n = pair
        positives.append(embeddings[p])
        negatives.append(embeddings[n])
        
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    return positives, negatives


def train(session_path, train_loader, test_loader):
    #TODO: tensorboard
    tensorboard_logs_path = f"{session_path}/runs/experiment1"
    writer = SummaryWriter(tensorboard_logs_path)
    # os.system(f"tensorboard --logdir={session_path}/runs &")
    # webbrowser.open("http://localhost:6006")

    model = CringeNet().to(configs.DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=configs.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode=configs.MODE, factor=configs.LR_REDUCTION_FACTOR, patience=configs.PATIENCE, min_lr=configs.MIN_LR)
    loss_fn = TripletLoss(configs.ALPHA)
    print(configs.DEVICE)
    for epoch in range(configs.EPOCHS):
        model.train()
        for i, input in enumerate(train_loader):
            optim.zero_grad()
            input = input.to(configs.DEVICE)
            embeddings = model(input)
            B, N, L = embeddings.shape
            embeddings = embeddings.view(B * N, L)  # (n, d)
            distance_matrix = compute_distance_matrix(embeddings)
            pairs = get_n_p_pairs(distance_matrix)
            ps, ns = form_triplets(embeddings, pairs)
            loss = loss_fn(embeddings, ps, ns)
            writer.add_scalar("Loss/train", loss.item(), epoch)
            loss.backward()
            optim.step()
            if i%20 ==0:
                print("train batch loss: ", loss.item())


        test_loss = compute_loss(model, test_loader, loss_fn)
        print("test loss: ",test_loss)
        scheduler.step(test_loss)
        print("LEARNING RATE: ", scheduler.get_last_lr())
        writer.add_scalar("Loss/test", test_loss)

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