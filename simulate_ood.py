import torch
from model_cnn import get_model
from uncertainty_gate import apply_uncertainty_gate

def run_simulation():
    print("Iniciando simulação do NeuroFlow Uncertainty Gate...\n")
    
    # Instanciar o modelo usando o teu motor de seleção (4 classes)
    model = get_model(model_name="custom", num_classes=4)
    model.eval()

    # 1. Simular pacote conhecido (In-Distribution)
    # Exemplo: O modelo reconhece perfeitamente a classe 0 (streaming_video)
    # Adicionámos o 4º valor ao tensor porque agora temos 4 classes
    mock_logits_id = torch.tensor([[6.0, -1.0, -2.5, -0.5]]) 
    
    # 2. Simular pacote desconhecido (Out-of-Distribution - ex: tráfego IoT cifrado)
    # A rede não reconhece o padrão e gera logits fracos e uniformes
    mock_logits_ood = torch.tensor([[1.0, 1.1, 0.9, 1.0]])

    print("\n--- Teste 1: Tráfego Conhecido (In-Distribution) ---")
    pred_id, prob_id, ent_id, rel_id = apply_uncertainty_gate(mock_logits_id)
    print(f"Probabilidades: {prob_id.detach().numpy()}")
    print(f"Entropia: {ent_id.item():.3f} bits")
    print(f"Ação: {'✅ Aplicar regra QoS' if rel_id.item() else '❌ Desviar tráfego (OOD)'}\n")

    print("--- Teste 2: Tráfego Desconhecido (Out-of-Distribution) ---")
    pred_ood, prob_ood, ent_ood, rel_ood = apply_uncertainty_gate(mock_logits_ood)
    print(f"Probabilidades: {prob_ood.detach().numpy()}")
    print(f"Entropia: {ent_ood.item():.3f} bits")
    print(f"Ação: {'✅ Aplicar regra QoS' if rel_ood.item() else '❌ Desviar tráfego (OOD)'}\n")

if __name__ == "__main__":
    run_simulation()