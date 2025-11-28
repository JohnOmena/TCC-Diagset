# Projeto DIAG (pasta `src`)

Guia rapido do que existe em `D:\DIAG\src` e do que sera usado na Fase 0 do TCC.

## Estrutura geral do projeto
- `partitions/DiagSet-A.{1,2}/`: splits em nivel de slide (train/validation/test) fornecidos junto com a base.
- `distributions/S/{5x,10x,20x,40x}/`: distribuicoes por slide e magnificacao para analises em nivel de WSI.
- `src/`: scripts principais de construcao de indices, datasets e treino.
- `src/data/`: indices de patches (`patch_index*.csv`) e splits em nivel de patch por magnificacao (`splits_A.{1,2}_{5x,10x,20x,40x}/`).
- `src/checkpoints/`: checkpoints antigos organizados por magnificacao (5x, 10x, 20x, 40x).
- `src/logs/`: logs CSV gerados durante os treinos (historia de epocas e metricas).
- `src/wandb/`: artefatos offline/exportados de execucoes do W&B.
- `Checkpoints_and_data/TCC/`: pastas por GPU (`GTX_1060`, `RTX_3060`, `RTX_4060`) e colecoes `All_`/`checkpoints` com os melhores modelos e logs consolidados; inclui `src.7z` como snapshot.
- Arquivo plano de resultados de runs W&B: `DiagSetA_wandb_runs_flat.csv` na raiz `D:\DIAG`.

## Modelos modernos que serao usados
- EfficientNetV2-S
- ConvNeXt-Tiny
- Swin-Tiny
- DeiT-Small

## Magnificacoes foco do TCC
- Usar: **5x**, **10x**, **20x**
- Nao usar inicialmente: **40x** (fica em espera por custo/tempo)

## Papel dos scripts principais
- `build_patch_index_blobs.py`: le os blobs `.npy` de patches (`blobs/S/{mag}/{slide}/{label}/*.npy`) e gera o CSV `data/patch_index_blobs_4mag.csv` com caminho do blob, indice do patch, slide, magnificacao e rotulo.
- `expand_splits.py`: pega os splits por slide (`partitions/DiagSet-A.{1,2}`) e expande para splits por patch por magnificacao, salvando em `src/data/splits_{A.1,A.2}_{5x,10x,20x,40x}/`.
- `diagset_data.py`: dataset PyTorch que usa memmap para ler patches 256x256, aplica augmentations (RandomCrop/CenterCrop 224, flip horizontal, rotacao em multiplos de 90 graus), normalizacao ImageNet e mapeamento de rotulos para S5 (9 classes) ou S1 (binario alinhado ao paper).
- `train.py`: loop de treino/validacao com escolha de modelo/magnificacao/esquema de rotulo, acumulacao de batch, mixed precision, logging em CSV e W&B, retomada de runs (resume/download de checkpoints) e salvamento em `src/checkpoints/{mag}`.

## Situacao atual dos experimentos
- Ja existem checkpoints e logs de uma rodada anterior para ConvNeXt, Swin, DeiT (mags 5x, 10x, 20x; S1 e S5). Eles ficam em `src/checkpoints/` e nos backups `Checkpoints_and_data/TCC/` (por GPU e em `All_`).
- Logs CSV correspondentes estao em `src/logs/` (e copias/variacoes em `Checkpoints_and_data/TCC`).
- O arquivo `DiagSetA_wandb_runs_flat.csv` agrega as runs antigas do W&B; os dados brutos das runs estao em `src/wandb/`.
- Esses resultados servem como versao 1. A partir daqui os novos treinos devem focar nas mags 5x/10x/20x com mapeamento S1 corrigido e mesmo LR (1e-4 + scheduler em 20/40 epocas) para refazer as comparacoes com os modelos modernos.
