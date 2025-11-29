from scripts.train import CONFIG, evaluate_split

def main():
    cfg = CONFIG
    cfg.model  = "convnext"   # ajuste se o ckpt for de outro modelo
    cfg.mag    = "10x"
    cfg.schema = "s5"         # importante: S5
    cfg.split  = "A.2"

    # Se o nome do checkpoint for diferente, ajuste aqui ou passe ckpt_path
    res = evaluate_split(cfg, split_name="test")

    print("\nResultados em A.2-test (S5 + colapsos):")
    for scheme, vals in res.items():
        print(f"\n=== {scheme} ===")
        for k, v in vals.items():
            if k == "cm":
                print("cm:")
                print(v)
            else:
                print(f"{k}: {v}")

if __name__ == "__main__":
    main()
