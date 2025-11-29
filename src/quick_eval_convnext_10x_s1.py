from scripts.train import CONFIG, evaluate_split  # train.py está em src/scripts

def main():
    # Usa a configuração padrão e ajusta para o experimento desejado
    cfg = CONFIG
    cfg.model  = "convnext"   # ou "effnet", "swin", "deit" – deve bater com o nome do checkpoint
    cfg.mag    = "10x"        # magnificação do checkpoint (5x, 10x ou 20x)
    cfg.schema = "s1"         # "s1" ou "s5", de acordo com o checkpoint
    cfg.split  = "A.2"        # estamos usando os splits A.2

    # Avalia no split de teste padrão (A.2-test)
    res = evaluate_split(cfg, split_name="test")

    print("\nResultado da avaliação em A.2-test:")
    print(res)

if __name__ == "__main__":
    main()
