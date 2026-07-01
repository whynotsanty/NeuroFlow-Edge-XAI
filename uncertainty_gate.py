import torch
import torch.nn.functional as F

def calculate_entropy(probs, epsilon=1e-9):
    """
    Calcula a entropia de Shannon em bits (base 2).
    """
    entropy = -torch.sum(probs * torch.log2(probs + epsilon), dim=1)
    return entropy

def apply_uncertainty_gate(logits, entropy_threshold=1.2):
    """
    Avalia a incerteza da previsão para detetar tráfego Out-of-Distribution (OOD).
    Retorna a previsão, probabilidades, entropia e um booleano de fiabilidade.
    """
    probs = F.softmax(logits, dim=1)
    entropies = calculate_entropy(probs)
    
    predictions = torch.argmax(probs, dim=1)
    
    # True se Incerteza < Threshold (Confiar) | False se Incerteza >= Threshold (Rejeitar/OOD)
    reliable_predictions = entropies < entropy_threshold
    
    return predictions, probs, entropies, reliable_predictions