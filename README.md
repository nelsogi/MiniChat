# MiniChat — une petite IA générative en C#

Un mini-GPT (Transformer) écrit de zéro avec **TorchSharp**, qui apprend à discuter
à partir de `data/dialogues.txt`, puis tourne en chat dans la console.

## Lancer

```bash
dotnet run                 # entraîne (la 1re fois) puis ouvre le chat
dotnet run -- --train      # refait tout : pré-entraînement + fine-tuning
dotnet run -- --finetune   # refait seulement le fine-tuning, depuis model-pretrain/
```

L'entraînement se fait en deux phases (réglages dans `TrainingConfig.Pretraining` / `FineTuning`) :

1. **Pré-entraînement** sur `data/texte.txt` (beaucoup de texte général) -> sauvegardé dans `model-pretrain/`.
2. **Fine-tuning** sur `data/dialogues.txt` -> sauvegardé dans `model/`.

Le tokenizer BPE (2000 tokens, `TokenizerVocabSize` dans `Program.cs`) est entraîné au début
de la phase 1 sur les deux fichiers, puis sauvegardé avec le modèle (`tokenizer.json`).
Si tu changes beaucoup les dialogues, refais un `--train` complet : `--finetune` garde
l'ancien tokenizer (les caractères nouveaux seront ignorés).

Le modèle (EmbedDim 256, 6 couches, 8 têtes, ~5,3 M paramètres) est lent sur CPU :
compte environ 2 s par étape. Avec une carte NVIDIA, c'est beaucoup plus rapide.

Le modèle entraîné est sauvegardé dans `bin/.../model/` : les lancements suivants
démarrent directement le chat.

Carte NVIDIA ? Dans `MiniChat.csproj`, remplace `TorchSharp-cpu` par `TorchSharp-cuda-windows`.

## Suivre les entraînements

Chaque entraînement (`--train` ou `--finetune`) laisse deux fichiers avec la même date dans `bin/.../results/` :

- `<date>.txt` : les réponses aux questions de `data/questions-test.txt`, posées à la fin avec
  une température (0.7) et une graine (1234) fixes. Seul le modèle change : compare les fichiers entre eux.
- `<date>-loss.csv` : les loss mesurées pendant l'entraînement (s'ouvre dans Excel pour tracer les courbes).

Pendant l'entraînement, une réponse d'exemple à la première question s'affiche tous les 1000 pas
(`SampleInterval` dans `TrainingConfig`).

## Préparer les données (DataBuilder)

Un petit projet console à part, dans `DataBuilder/` :

```bash
# Dialogues OpenAssistant (fichier *.messages.jsonl ou .jsonl.gz) -> format Toi:/IA:
dotnet run --project DataBuilder -- oasst Data/dialogues.txt 2023-04-12_oasst_ready.messages.jsonl.gz --lang fr --max 300

# Livres du Projet Gutenberg (.txt) -> texte propre pour le pré-entraînement
dotnet run --project DataBuilder -- gutenberg Data/texte.txt pg4650.txt pg17489.txt
```

- **oasst** suit la réponse la mieux classée de chaque conversation, ignore les messages supprimés,
  refusés, trop longs, avec du code ou des caractères exotiques (emoji...).
- **gutenberg** retire l'en-tête et la licence, recolle les paragraphes (un par ligne),
  simplifie la typographie (’ … — -> ' ... -) et lit aussi les vieux fichiers en Latin-1.

Attention : la sortie est **écrasée**. Garde une copie de tes `dialogues.txt` faits à la main.

## Organisation du code

| Dossier          | Classe                | Rôle |
|------------------|-----------------------|------|
| `Config/`        | `ModelConfig`         | Taille du réseau (couches, têtes, mémoire) |
|                  | `TrainingConfig`      | Nombre d'étapes, vitesse d'apprentissage |
| `Tokenization/`  | `BpeTokenizer`        | Texte ↔ nombres, par morceaux de mots (BPE appris sur nos textes) |
| `Data/`          | `TextDataset`         | Découpe le texte en extraits pour l'entraînement |
| `Model/`         | `CausalSelfAttention` | L'attention : regarder les tokens d'avant |
|                  | `FeedForward`         | Petit réseau appliqué à chaque position |
|                  | `TransformerBlock`    | Attention + FeedForward + connexions résiduelles |
|                  | `MiniGpt`             | Le modèle complet |
| `Training/`      | `Trainer`             | La boucle d'apprentissage |
| `Generation/`    | `TextGenerator`       | Écrit du texte token par token (top-k, top-p) |
| `Chat/`          | `ChatSession`         | Transforme la génération en conversation |
| `Evaluation/`    | `QuestionTest`        | Questions de test après l'entraînement, exemples pendant |
| `Storage/`       | `ModelStore`          | Sauvegarde / chargement du modèle |

## Ce qu'il faut savoir

- Avec ~6 Ko de dialogues, l'IA va surtout **réciter** et mélanger ce qu'elle a vu.
  C'est normal : la qualité dépend d'abord de la **quantité de données**.
- Surveille les deux loss pendant l'entraînement. Si la loss de vérification
  remonte alors que l'autre baisse, le modèle apprend par cœur.
- `/temp 0.5` dans le chat : réponses plus sages. `/temp 1.2` : plus créatives.

## Prochaines étapes

1. Ajouter beaucoup plus de texte dans `data/texte.txt` (plusieurs livres, vise quelques Mo).
2. Passer sur GPU (`TorchSharp-cuda-windows`) pour entraîner plus longtemps.
3. Brancher le modèle dans une appli WinForms ou un jeu.
