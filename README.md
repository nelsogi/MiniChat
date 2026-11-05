# MiniChat — une petite IA générative en C#

Un mini-GPT (Transformer) écrit de zéro avec **TorchSharp**, qui apprend à discuter
à partir de `data/dialogues.txt`, puis tourne en chat dans la console.

## Lancer

```bash
dotnet run                 # entraîne (la 1re fois) puis ouvre le chat
dotnet run -- --train      # refait tout : pré-entraînement + fine-tuning
dotnet run -- --finetune   # refait seulement le fine-tuning, depuis model-pretrain/
dotnet run -- --resume     # reprend un entraînement interrompu (coupure, plantage, Ctrl+C)
```

Pendant l'entraînement, l'avancement est sauvegardé tous les 250 pas dans `bin/.../checkpoint/`
(`CheckpointInterval`) : poids, état de l'optimiseur, meilleurs poids. `--resume` repart de là.
Le dossier est supprimé quand l'entraînement va jusqu'au bout.

Arrêt anticipé : si la loss de vérification ne s'améliore plus pendant 2000 pas
(`EarlyStoppingPatience`, 0 = jamais), la phase s'arrête et garde le meilleur modèle.

L'entraînement se fait en deux phases (réglages dans `TrainingConfig.Pretraining` / `FineTuning`) :

1. **Pré-entraînement** sur `data/texte.txt` (beaucoup de texte général) -> sauvegardé dans `model-pretrain/`.
2. **Fine-tuning** sur `data/dialogues.txt` -> sauvegardé dans `model/`.

Le tokenizer BPE (2000 tokens, `TokenizerVocabSize` dans `Program.cs`) est entraîné au début
de la phase 1 sur les deux fichiers, puis sauvegardé avec le modèle (`tokenizer.json`).
Si tu changes beaucoup les dialogues, refais un `--train` complet : `--finetune` garde
l'ancien tokenizer (les caractères nouveaux seront ignorés).

Le modèle (EmbedDim 384, 6 couches, 6 têtes, mémoire de 256 tokens, ~11,5 M paramètres) apprend
à chaque étape sur 64 extraits, envoyés au GPU par paquets de 16 (accumulation de gradients,
voir `BatchSize` / `EffectiveBatchSize`). Sur une NVIDIA T400 4 Go : ~2 Go de mémoire GPU,
~3,5 s par étape. La mémoire GPU utilisée s'affiche à chaque mesure de loss.

Le modèle entraîné est sauvegardé dans `bin/.../model/` : les lancements suivants
démarrent directement le chat.

GPU NVIDIA utilisé automatiquement s'il y en a un (pilote récent), CPU sinon : rien à configurer.

Autres dossiers, sans recompiler :

```bash
dotnet run -- --data D:\MesDonnees      # texte.txt, dialogues.txt, questions-test.txt
dotnet run -- --models D:\MesModeles    # model/, model-pretrain/ (+ checkpoint/ et results/ y sont écrits)
```

## Lancer sur un autre PC

### Préparer le paquet (sur ton PC)

```powershell
powershell -ExecutionPolicy Bypass -File publish.ps1
```

Le script fait `dotnet publish -c Release -r win-x64 --self-contained true -o publish`, puis zippe
le dossier dans `MiniChat-win-x64.zip`. Le dossier `publish/` contient :

- `MiniChat.exe` et le runtime .NET (rien à installer côté .NET sur l'autre PC),
- TorchSharp et libtorch (CUDA et CPU),
- `data/`,
- `model/` et `model-pretrain/` s'ils existent (pris dans `bin\Debug\net8.0`, là où `dotnet run`
  les crée ; autre source : `publish.ps1 -ModelsSource D:\MesModeles`).

Attention à la taille : libtorch CUDA fait plusieurs Go (voir la taille affichée à la fin du script).

### Sur l'autre PC

1. Dézippe l'archive où tu veux (par exemple `C:\MiniChat`).
2. Lance `MiniChat.exe` depuis ce dossier, ou depuis n'importe où :
   `C:\MiniChat\MiniChat.exe`. Les options marchent pareil :
   `MiniChat.exe --train`, `--finetune`, `--resume`, `--data <dossier>`, `--models <dossier>`.
3. Pré-requis sur l'autre PC :
   - Windows 10/11 64 bits.
   - [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) (libtorch en a besoin ;
     il est déjà présent sur la plupart des PC).
   - Pour le GPU : une carte NVIDIA avec un pilote récent (version 528 ou plus, CUDA 12).
     Sans ça, MiniChat tourne quand même, sur CPU.
   - Si `MiniChat.exe` est dans un dossier protégé (ex. `C:\Program Files`), donne-lui `--models`
     vers un dossier où il peut écrire (entraînement, `results/`, `checkpoint/`).

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

# Colonne de texte de fichiers .parquet (Wikipedia, FineWeb... sur Hugging Face), arrêt à 50 Mo
dotnet run --project DataBuilder -- parquet Data/texte.txt train-00000-of-00002.parquet --column text --max-mo 50
```

- **oasst** suit la réponse la mieux classée de chaque conversation, ignore les messages supprimés,
  refusés, trop longs, avec du code ou des caractères exotiques (emoji...).
- **gutenberg** retire l'en-tête et la licence, recolle les paragraphes (un par ligne),
  simplifie la typographie (’ … — -> ' ... -) et lit aussi les vieux fichiers en Latin-1.
- **parquet** lit la colonne `--column` (par défaut `text`) : un document par valeur, nettoyé comme
  pour Gutenberg, une ligne du document = un paragraphe. `--max-mo` arrête avant le document qui
  ferait dépasser la taille ; sans cette option, tout est gardé. Affiche le nombre de documents gardés.

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
