# MiniChat — une petite IA générative en C#

Un mini-GPT (Transformer) écrit de zéro avec **TorchSharp**, qui apprend à discuter
à partir de `data/dialogues.txt`, puis tourne en chat dans **Nel Code**, une interface console
faite avec [Spectre.Console](https://spectreconsole.net/).

## Lancer

```bash
dotnet run                 # ouvre Nel Code (sans modèle entraîné : il propose /train)
```

Au lancement, un cadre d'accueil affiche le modèle chargé (taille), GPU ou CPU et les dossiers.
Écris un message pour discuter, ou une commande :

| Commande                                    | Ce qu'elle fait |
|---------------------------------------------|-----------------|
| `/help`                                     | La liste des commandes |
| `/train`                                    | Nouveau modèle : pré-entraînement + fine-tuning |
| `/finetune`                                 | Refait seulement le fine-tuning, depuis `model-pretrain/` |
| `/continue <étapes> [finetune] [--lr 3e-4]` | Entraîne plus longtemps un modèle existant (voir plus bas) |
| `/resume`                                   | Reprend un entraînement interrompu |
| `/test`                                     | Questions de test sur le modèle chargé (réponses dans `results/<date>-test.txt`) |
| `/stats`                                    | Courbes de loss (entraînement et vérification) du dernier entraînement, meilleur point marqué ★ |
| `/temp 0.8`, `/topk 20`, `/topp 0.9`        | Réglages de génération (sans nombre : affiche la valeur) |
| `/info`                                     | Modèle, GPU, réglages, conversation |
| `/reset`                                    | Oublie la conversation |
| `/clear`                                    | Efface l'écran et la conversation |
| `/quit`                                     | Quitte |

Pendant l'entraînement, une barre de progression montre le pas, les deux loss, la vitesse
d'apprentissage et le temps restant ; chaque mesure laisse une ligne d'historique au-dessus.

Dans la zone de saisie :
- tape `/` : la liste des commandes qui correspondent s'affiche dessous. `↑` `↓` pour choisir,
  `Tab` pour compléter, `Entrée` pour lancer, `Échap` pour fermer la liste ;
- liste fermée, `↑` `↓` rappellent les messages et commandes déjà envoyés ;
- `Échap` pendant une réponse l'arrête. Pendant un entraînement (ou `/test`), il l'arrête proprement :
  les loss sont mesurées au pas atteint, le meilleur modèle est sauvegardé, et `/resume` reprend
  exactement à ce pas.

Les options en ligne de commande marchent toujours : elles lancent la commande qui va avec au démarrage.

```bash
dotnet run -- --train      # refait tout : pré-entraînement + fine-tuning
dotnet run -- --finetune   # refait seulement le fine-tuning, depuis model-pretrain/
dotnet run -- --resume     # reprend un entraînement interrompu (coupure, plantage, Ctrl+C)
dotnet run -- --continue 2000            # 2000 pas de pré-entraînement en plus, depuis model-pretrain/
dotnet run -- --continue-finetune 200    # 200 pas de fine-tuning en plus, depuis model/
dotnet run -- --continue 2000 --lr 2e-4  # vitesse de départ choisie
```

### Continuer un modèle (`--continue`, `--continue-finetune`)

Pour entraîner plus longtemps, ou après avoir ajouté des données dans `data/` (relues à chaque lancement) :

- Le modèle repart de ses poids **et** de l'état d'AdamW, sauvegardés avec lui (`optimizer.dat`), et les pas
  continuent sa numérotation (`training.json`) : `--continue 2000` sur un modèle arrêté au pas 9000 va de 9000 à 11000.
- Le tokenizer n'est jamais réentraîné : le modèle garde le sien, les caractères qu'il ne connaît pas sont ignorés.
- Nouveau warmup court (200 pas, moitié des pas si tu en demandes moins de 400), puis cosinus depuis une vitesse
  plus basse : 3e-4 en pré-entraînement, 1e-4 en fine-tuning (à 3e-4 il apprend les dialogues par cœur).
  `--lr` change ce départ ; la fin du cosinus est toujours dix fois plus bas.
- Le modèle sauvegardé n'est remplacé que si la loss de vérification est meilleure. L'ancienne est mesurée au
  premier pas, sur les données actuelles (comparaison juste même si elles ont changé) ; les deux sont affichées.
- Les loss s'ajoutent au CSV de l'entraînement d'origine (`results/<date>-loss.csv`). Les réponses aux
  questions de test vont dans un nouveau `<date>.txt`.
- `--continue` ne refait pas le fine-tuning : relance ensuite `--finetune` pour que le chat en profite.
- Une continuation interrompue se reprend avec `--resume`, avec les mêmes réglages.
- Un modèle sauvegardé avant cette option n'a ni `optimizer.dat` ni `training.json` : AdamW repart de zéro
  et les pas sont comptés depuis 0. Son CSV d'origine n'est pas connu : un nouveau est créé.

Pendant l'entraînement, l'avancement est sauvegardé tous les 250 pas dans `bin/.../checkpoint/`
(`CheckpointInterval`) : poids, état de l'optimiseur, meilleurs poids. `--resume` repart de là.
Le dossier est supprimé quand l'entraînement va jusqu'au bout.

Arrêt anticipé : si la loss de vérification ne s'améliore plus pendant 2000 pas (100 en fine-tuning)
(`EarlyStoppingPatience`, 0 = jamais), la phase s'arrête et garde le meilleur modèle.

L'entraînement se fait en deux phases (réglages dans `TrainingConfig.Pretraining` / `FineTuning`) :

1. **Pré-entraînement** sur `data/texte.txt` (beaucoup de texte général) -> sauvegardé dans `model-pretrain/`.
   20 000 étapes. `texte.txt` peut faire plusieurs Go : il est lu ligne par ligne, et ses tokens
   prennent 2 octets chacun en mémoire (1,5 Go de texte -> ~460 M tokens -> ~0,9 Go de RAM, ~1 min de découpage).
2. **Fine-tuning** sur `data/dialogues.txt` -> sauvegardé dans `model/`. Court (300 étapes, vitesse 1e-4) :
   les dialogues sont petits et le modèle les apprend vite par cœur.

Le tokenizer BPE (4000 tokens, `TokenizerVocabSize` dans `Program.cs`) est entraîné au début
de la phase 1 sur les dialogues et un échantillon de 20 Mo de `texte.txt` réparti dans tout le fichier
(`TokenizerSampleMb`, ~1 min 30), puis sauvegardé avec le modèle (`tokenizer.json`).
Si tu changes beaucoup les dialogues, refais un `--train` complet : `--finetune` garde
l'ancien tokenizer (les caractères nouveaux seront ignorés).

Le modèle (EmbedDim 512, 8 couches, 8 têtes, mémoire de 256 tokens, dropout 0,1, ~27 M paramètres) apprend
à chaque étape sur 64 extraits, envoyés au GPU par paquets de 16 (accumulation de gradients,
voir `BatchSize` / `EffectiveBatchSize`). Sur une RTX 3060 Ti 8 Go : ~3,2 Go de mémoire GPU (toute la carte),
0,46 s par étape, soit ~2 h 35 de pré-entraînement. Des paquets de 32 ne vont pas plus vite.
La mémoire GPU utilisée s'affiche à chaque mesure de loss.
Les modèles déjà entraînés gardent leur taille (`config.json`) : ces réglages valent pour le prochain `--train`.

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
   `MiniChat.exe --train`, `--finetune`, `--resume`, `--continue <pas>`, `--continue-finetune <pas>`, `--lr <vitesse>`,
   `--data <dossier>`, `--models <dossier>`.
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
- **parquet** lit la colonne `--column` (par défaut `text`) ; un fichier sans cette colonne mais avec une
  seule colonne de texte (TinyStories : `french-tinystories`) utilise celle-là. Un document par valeur,
  nettoyé comme pour Gutenberg, une ligne du document = un paragraphe. `--max-mo` arrête avant le document qui
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
| `App/`           | `Workbench`           | Le modèle chargé et ce qui le change : entraîner, continuer, reprendre, tester |
|                  | `NelShell`            | La boucle de Nel Code : commandes ou messages pour le modèle |
| `Commands/`      | `ICommand`            | Une commande (`Name`, `Description`, `Usage`, `Execute`) ; une classe par commande |
|                  | `CommandRegistry`     | Trouve toutes les commandes toute seule (ajouter une classe suffit) |
| `UI/`            | `Theme`               | Les couleurs et symboles, à un seul endroit |
|                  | `WelcomeBanner`       | Le cadre d'accueil |
|                  | `ChatView`            | La conversation : spinner, puis réponse au fur et à mesure |
|                  | `InputBox`            | La zone de saisie |
|                  | `TrainingProgressView`| La barre de progression de l'entraînement |

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
