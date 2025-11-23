# Projeto DIAG – Experimentos do TCC

Este diretório (`D:\DIAG\src`) contém os scripts e configurações usados para os
experimentos de classificação de patches na base **DiagSet-A** (A.1 e A.2),
relacionados ao meu TCC.

O objetivo é **replicar o protocolo do artigo “DiagSet: a dataset for prostate
cancer histopathological image classification”** trocando apenas os modelos
de classificação de patch por arquiteturas mais leves e modernas.

---

## 1. Estrutura geral do projeto (resumo)

Raiz do repositório: `D:\DIAG`

- `partitions/DiagSet-A.1/*.csv`  
  Splits em nível de *slide* (train/validation/test) para A.1.

- `partitions/DiagSet-A.2/*.csv`  
  Splits em nível de *slide* (train/validation/test) para A.2 – **mesmos splits do paper**.

- `distributions/S/{5x,10x,20x,40x}/*.json`  
  Distribuições por slide/magnificação (usadas para análises em nível de WSI).

- `src/`  
  Scripts de construção de índices, datasets e treino.

- `src/data/`  
  - `patch_index*.csv`: índices de patches.  
  - `splits_A.{1,2}_{5x,10x,20x,40x}/*.csv`: splits em nível de **patch** por magnificação.

- `src/checkpoints/`  
  Checkpoints por magnificação/modelo/esquema de rótulo (S1, S5).

- `src/logs/`  
  Logs CSV dos treinos (histórico de épocas, métricas, etc).

- `Checkpoints_and_data/TCC/`  
  Organização adicional de checkpoints e logs por GPU (GTX_1060, RTX_3060, RTX_4060)
  e coleções “All_” com os melhores modelos.

---

## 2. Scripts principais (versão atual)

Os scripts centrais usados no fluxo do TCC são:

- `build_patch_index_blobs.py`  
  Cria o índice de patches a partir dos blobs `.npy` da DiagSet-A.

- `expand_splits.py`  
  Expande os splits em nível de slide (`partitions/DiagSet-A.*/*.csv`) para
  splits em nível de patch por magnificação (`src/data/splits_A.*_{mag}/*.csv`).

- `diagset_data.py`  
  Implementa o `Dataset` de patches da DiagSet-A para PyTorch, incluindo:
  - carregamento via `memmap` dos blobs de patches;
  - augmentations (RandomCrop 224, flip, rotação 90°);
  - normalização estilo ImageNet;
  - mapeamento de rótulos para os esquemas S1 e S5 (será ajustado).

- `train.py`  
  Script de treino e validação dos modelos, com:
  - escolha de modelo (`--model`),
  - escolha de magnificação (`--mag`),
  - escolha de esquema de rótulo (`--schema` = s1, s5),
  - logging em CSV e Weights & Biases,
  - suporte a mixed precision e gradient accumulation.

Outros scripts auxiliares:
- `inspect.py`, `inspect_shapes.py`, `teste_loader.py`: inspeções e testes de loaders.
- `loginWand.py`: autenticação no Weights & Biases.
- `wandb_api_resume.py`: consolidação de runs do W&B.

---

## 3. Modelos que serão usados no TCC

Modelos modernos escolhidos para substituir as arquiteturas clássicas do paper:

- **EfficientNetV2-S**
- **ConvNeXt-Tiny**
- **Swin-Tiny**
- **DeiT-Small**

Outros modelos podem aparecer em testes, mas o foco do TCC será a comparação
desses quatro modelos com os resultados originais do artigo.

---

## 4. Magnificações que serão usadas

Magnificações disponíveis na DiagSet-A:

- 5×, 10×, 20×, 40×

Magnificações definidas para o TCC:

- **Usar**: 5×, 10×, 20×  
- **Não usar inicialmente**: 40× (pode ser considerada apenas se sobrar tempo,
pois é a mais pesada em número de patches e tempo de treino).

Os splits por magnificação usados no TCC são:

- `src/data/splits_A.2_5x/{train,validation,test}.csv`
- `src/data/splits_A.2_10x/{train,validation,test}.csv`
- `src/data/splits_A.2_20x/{train,validation,test}.csv`

---

## 5. Situação atual dos experimentos (resumo)

- Já existem checkpoints e logs para:
  - modelos EfficientNetV2-S, ConvNeXt-Tiny, Swin-Tiny, DeiT-S;
  - magnificações 5×, 10×, 20×;
  - esquemas S1 e S5 (versão antiga de S1, será corrigida).

- O arquivo `DiagSetA_wandb_runs_flat.csv` consolida os resultados de runs
anteriores no Weights & Biases.

Esses resultados antigos **não serão descartados**, mas serão considerados
como “primeira versão”. A partir de agora, os experimentos serão refeitos
com:

- mapeamento de S1 corrigido para bater com o paper (BG/T/N/A vs R1–R5),
- `learning rate = 1e-4` com scheduler em 20 e 40 épocas,
- foco nas mags 5×, 10×, 20×.

---

## 6. Próximos passos (ligados ao plano do TCC)

1. **Ajustar S1 e S5 em `diagset_data.py`**  
   - Corrigir S1 para: BG/T/N/A = não-câncer; R1–R5 = câncer.
   - Confirmar que S5 (9 classes) está batendo com o artigo.

2. **Treinar S1 e S5 com os modelos modernos**  
   - Para cada modelo e mag (5×, 10×, 20×):
     - treinar S1 (binário) com o mapeamento corrigido;
     - treinar S5 (9 classes) e depois derivar S2–S4 a partir de S5.

3. **Comparar com o artigo**  
   - Gerar tabelas com Acc e AvAcc por modelo/mag/esquema;
   - Comparar com os resultados de VGG/ResNet/Inception/ViT-B/32 do paper.

4. **(Opcional) Subir para nível de WSI (DiagSet-B)**  
   - Usar o melhor modelo para gerar mapas de probabilidade;
   - Replicar a lógica de % de tecido tumoral e limiares TL/TU.

---

## 7. Ambiente e hardware

- Ambiente principal:
  - Windows + Conda + Python + PyTorch.
  - IDE frequentemente usada: **Spyder**.

- Hardware disponível:
  - **RTX 3060 Laptop** (GPU principal para os experimentos do TCC).
  - Resultados antigos também incluem execuções em GTX 1060 e RTX 4060,
    guardadas em `Checkpoints_and_data\TCC\{GTX_1060,RTX_3060,RTX_4060}`.