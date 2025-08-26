import torch
import configs


from TripletsFormer import compute_distance_matrix, form_triplets, get_a_n_p_pairs

def compute_triplet_loss(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, input in enumerate(loader):
            input = input.to(configs.DEVICE)
            embeddings = model(input)
            B, N, L = embeddings.shape
            embeddings = embeddings.view(B * N, L)
            distance_matrix = compute_distance_matrix(embeddings)
            tripletes = get_a_n_p_pairs(distance_matrix)
            anc, ps, ns = form_triplets(embeddings, tripletes)
            loss = loss_fn(anc, ps, ns)
        
            losses.append(loss.item())

    return sum(losses)/len(losses)




def compute_accuracy_score(model, loader):
    """
    Compute verification accuracy for a Siamese/triplet network.
    Returns the accuracy (0-1).
    """
    model.eval()
    distances = []
    labels = []
    with torch.no_grad():
        for input in loader:
            input = input.to(configs.DEVICE)
            embeddings = model(input)
            B, N, L = embeddings.shape
            embeddings = embeddings.view(B * N, L)
            distance_matrix = compute_distance_matrix(embeddings)
            triplets = get_a_n_p_pairs(distance_matrix)
            anchors, ps, ns = form_triplets(embeddings, triplets)
            
            # distances
            pos_dist = torch.norm(anchors - ps, dim=1)
            neg_dist = torch.norm(anchors - ns, dim=1)
            
            distances.extend(pos_dist.cpu().numpy())
            labels.extend([1]*len(pos_dist))  # 1 = same
            distances.extend(neg_dist.cpu().numpy())
            labels.extend([0]*len(neg_dist))  # 0 = different
    
    distances = torch.tensor(distances)
    labels = torch.tensor(labels)
    
    # simple threshold selection: midpoint between mean pos/neg distances
    pos_mean = distances[labels==1].mean()
    neg_mean = distances[labels==0].mean()
    dynamic_threshold = (pos_mean + neg_mean) / 2
    th1 = 0.7
    th2 = 0.8
    th3 = 0.9
    th4 = 1.0
    th5 = 1.1
    th6 = 1.2
    
    preds_at_dynamic_th = (distances < dynamic_threshold).int()
    preds_at_th1 = (distances < th1).int()
    preds_at_th2 = (distances < th2).int()
    preds_at_th3 = (distances < th3).int()
    preds_at_th4 = (distances < th4).int()
    preds_at_th5 = (distances < th5).int()
    preds_at_th6 = (distances < th6).int()

    accuracy_at_dynamic_th = (preds_at_dynamic_th == labels).float().mean().item()
    accuracy_th1 = (preds_at_th1 == labels).float().mean().item()
    accuracy_th2 = (preds_at_th2 == labels).float().mean().item()
    accuracy_th3 = (preds_at_th3 == labels).float().mean().item()
    accuracy_th4 = (preds_at_th4 == labels).float().mean().item()
    accuracy_th5 = (preds_at_th5 == labels).float().mean().item()
    accuracy_th6 = (preds_at_th6 == labels).float().mean().item()
    return ((dynamic_threshold,accuracy_at_dynamic_th), (th1,accuracy_th1), (th2,accuracy_th2), (th3,accuracy_th3), (th4,accuracy_th4), (th5,accuracy_th5), (th6,accuracy_th6))