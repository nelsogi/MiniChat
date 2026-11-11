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
| `/train [--size small\|medium\|large] [--epochs 1,1]` | Nouveau modèle : pré-entraînement + fine-tuning (taille, voir plus bas ; longueur en passages sur `texte.txt`). Montre les paramètres et la durée estimée, puis demande confirmation |
| `/finetune [--epochs 2,5]`                  | Refait seulement le fine-tuning, depuis `model-pretrain/` (passages sur les dialogues) |
| `/continue <étapes> [finetune] [--lr 3e-4]` | Entraîne plus longtemps un modèle existant (voir plus bas) |
| `/resume`                                   | Reprend un entraînement interrompu |
| `/test`                                     | Questions de test sur le modèle chargé (réponses dans `results/<date>-test.txt`) |
| `/models`                                   | Les modèles sauvegardés : taille, loss de vérification, pas, dates |
| `/save <nom>`                               | Garde le modèle chargé sous un nom (`saved/<nom>/`), avant un `/train` ou un `/continue` par exemple |
| `/load <nom>`                               | Charge un modèle (`model`, `model-pretrain` ou un nom de `/save`) pour le chat |
| `/export <nom> [description] [--from model-pretrain]` | Exporte un modèle (par défaut `model`) en `exports/<nom>.nelmodel`, pour `/import` dans Nel Code (format : `NELMODEL-FORMAT.md`) |
| `/compare <nom1> <nom2>`                    | Les questions de test posées aux deux modèles, réponses côte à côte |
| `/stats`                                    | Courbes de loss (entraînement et vérification) du dernier entraînement, meilleur point marqué ★ |
| `/temp 0.8`, `/topk 20`, `/topp 0.9`        | Réglages de génération pour la conversation en cours (sans nombre : affiche la valeur) |
| `/config [réglage valeur]`                  | Affiche ou modifie les réglages enregistrés dans `settings.json` |
| `/save-chat [fichier]`                      | Sauvegarde la conversation en texte (par défaut `chats/<date>.txt`) |
| `/info`                                     | Modèle, GPU, réglages, conversation |
| `/reset`                                    | Oublie la conversation |
| `/clear`                                    | Efface l'écran et la conversation |
| `/quit`                                     | Quitte |

Pendant l'entraînement, une barre de progression montre le pas, les deux loss, la vitesse
d'apprentissage et le temps restant ; chaque mesure laisse une ligne d'historique au-dessus.

### Réglages (`settings.json`)

Dans le dossier des modèles (à côté de `model/`), créé au premier lancement et lu à chaque démarrage :

```json
{
  "temperature": 0.8,
  "topK": 20,
  "topP": 0.9,
  "defaultModel": "model"
}
```

`defaultModel` : le modèle chargé au démarrage, `model` (celui qui discute) ou `model-pretrain`
(pour voir ce que donne le pré-entraînement seul). Modifiable à la main ou avec `/config`, par exemple
`/config temperature 0.6` (enregistré et appliqué tout de suite). `/temp`, `/topk` et `/topp`, eux,
ne changent que la conversation en cours. Les questions de test gardent toujours leurs propres réglages fixes.

### Clavier

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
dotnet run -- --train --epochs 0.5       # pré-entraînement plus court : un demi-passage sur texte.txt
dotnet run -- --train --size medium      # un plus gros modèle (voir « Tailles du modèle »)
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
- La continuation a sa propre courbe complète sur exactement le nombre de pas demandé : warmup court
  (200 pas, moitié des pas si tu en demandes moins de 400), puis cosinus jusqu'au dernier pas demandé,
  depuis une vitesse plus basse : 3e-4 en pré-entraînement, 2e-4 en fine-tuning.
  `--lr` change ce départ ; la fin du cosinus est toujours dix fois plus bas (LR/10 au dernier pas).
- Le dropout, les betas d'AdamW et le weight decay sont ceux de la phase (voir plus bas). Un `optimizer.dat`
  sauvegardé avant le weight decay séparé de l'embedding n'a pas les mêmes groupes : AdamW repart de zéro (message).
- Le modèle sauvegardé n'est remplacé que si la loss de vérification est meilleure. L'ancienne est mesurée au
  premier pas, sur les données actuelles (comparaison juste même si elles ont changé) ; les deux sont affichées.
- Les loss s'ajoutent au CSV de l'entraînement d'origine (`results/<date>-loss.csv`). Les réponses aux
  questions de test vont dans un nouveau `<date>.txt`.
- `--continue` ne refait pas le fine-tuning : relance ensuite `--finetune` pour que le chat en profite.
- Une continuation interrompue se reprend avec `--resume`, avec les mêmes réglages.
- Un modèle sauvegardé avant cette option n'a ni `optimizer.dat` ni `training.json` : AdamW repart de zéro
  et les pas sont comptés depuis 0. Son CSV d'origine n'est pas connu : un nouveau est créé.

Pendant l'entraînement, l'avancement est sauvegardé tous les 1000 pas dans `bin/.../checkpoint/`
(`CheckpointInterval`) : poids, état de l'optimiseur, meilleurs poids. `--resume` repart de là.
Le dossier est supprimé quand l'entraînement va jusqu'au bout.

Vérification : la loss de vérification est mesurée sur 200 extraits fixes de la partie vérification
(`ValidationExtracts`), tirés une fois avec une graine fixe : toujours les mêmes, à chaque mesure, reprise ou
continuation. Deux mesures ne diffèrent que parce que le modèle a changé (avant, 20 batchs tirés au hasard
faisaient bouger la loss de ±0,05 à modèle égal). Mesure tous les 500 pas en pré-entraînement, 50 en fine-tuning
(`EvalInterval`).

Arrêt anticipé : si la loss de vérification ne s'améliore plus pendant 2000 pas (300 en fine-tuning)
(`EarlyStoppingPatience`, 0 = jamais), la phase s'arrête et garde le meilleur modèle. En pré-entraînement,
jamais avant un passage complet sur les données (`EarlyStoppingMinEpochs`) : avant, chaque extrait est du texte
neuf. Le nombre de passages faits s'affiche à chaque mesure (et dans la colonne `passages` du CSV).

L'entraînement se fait en deux phases (réglages dans `TrainingConfig.Pretraining` / `FineTuning`,
adaptés à la taille du modèle par `PretrainingFor` / `FineTuningFor`). Les durées ci-dessous sont celles du modèle small.
Leur longueur se donne en passages sur les données (`Epochs`, option `--epochs`) : le nombre de pas en est
calculé une fois les données découpées, et le cosinus du learning rate va toujours jusqu'au dernier pas prévu.

1. **Pré-entraînement** sur `data/texte.txt` (beaucoup de texte général) -> sauvegardé dans `model-pretrain/`.
   1,1 passage : ~56 000 étapes estimées pour les ~830 M tokens d'entraînement de `texte.txt`, ~6 h 30
   (`--epochs 0,6` : ~30 000 étapes, ~3 h 30).
   Dropout 0 (le modèle ne revoit presque jamais le même texte), AdamW betas (0,9 ; 0,95), weight decay 0,1
   sur les matrices des couches mais pas sur l'embedding des tokens.
   `texte.txt` peut faire plusieurs Go : il est lu ligne par ligne, et ses tokens
   restent sur le GPU, 2 octets chacun (2,9 Go de texte -> ~920 M tokens -> ~1,9 Go de mémoire GPU).
   Le découpage en tokens (~1 min) n'est fait qu'une fois : il est gardé dans `bin/.../cache/` et relu
   en ~1 s tant que `texte.txt` et le tokenizer ne changent pas (même chose pour `dialogues.txt`).
2. **Fine-tuning** sur `data/dialogues.txt` et `data/identite.txt` -> sauvegardé dans `model/`. 2,5 passages
   (~4 500 étapes estimées, ~35 min) à 2e-4 sur les ~292 000 conversations (OpenAssistant, French Alpaca, French Instruct,
   Luth-SFT : voir DataBuilder). Dropout 0,1 (`ModelConfig.Dropout`), betas (0,9 ; 0,999), weight decay aussi sur l'embedding.
   Avec de petits dialogues, l'arrêt anticipé s'arrête avant qu'il les apprenne par cœur.

Identité : `data/identite.txt` contient 50 conversations sur qui est MiniChat (nom, ce qu'il sait faire, ses
limites), au même format que `dialogues.txt`. Chacune est glissée 10 fois parmi les dialogues, à des places
tirées au hasard (`IdentityRepeats` dans `Workbench`) : ~0,3 % des dialogues. Le chat n'ajoute plus de phrase
d'identité au début de chaque conversation : toute la mémoire du modèle reste pour la discussion. Après avoir
modifié `identite.txt`, refais `/finetune`.

Initialisation d'un nouveau modèle (comme GPT-2) : embeddings et couches `Linear` selon N(0 ; 0,02), biais à 0,
et les deux projections qui écrivent dans la connexion résiduelle (sortie de l'attention et du SwiGLU) selon
N(0 ; 0,02/√(2 × Layers)), pour que la somme des 2 × Layers corrections ne grandisse pas avec la profondeur.

Le tokenizer BPE (8000 tokens, `TokenizerVocabSize` dans `App/Workbench.cs`) est entraîné au début
de la phase 1 sur un échantillon de 60 Mo de `texte.txt` et de 20 Mo de `dialogues.txt` (plus `identite.txt`),
chacun pris en blocs de 1 Mo répartis dans tout le fichier (`TokenizerTextSampleMb`, `TokenizerDialogueSampleMb` ;
quelques minutes), puis sauvegardé avec le modèle (`tokenizer.json`). Avec 8000 tokens au lieu de 4000, les mots
fréquents tiennent plus souvent en un seul token : un même texte fait moins de tokens, et un extrait de 512 tokens
contient plus de texte.
Un nouveau tokenizer a une autre empreinte : le cache des tokens (`cache/`) est redécoupé au premier `/train`
(~1 min pour `texte.txt`), et l'ancien fichier remplacé. Les modèles déjà entraînés gardent leur tokenizer.
Si tu changes beaucoup les dialogues, refais un `--train` complet : `--finetune` garde
l'ancien tokenizer (les caractères nouveaux seront ignorés).

### Tokens spéciaux

Le tokenizer a quatre tokens spéciaux, jamais découpés ni fusionnés : `<|doc|>`, `<|user|>`,
`<|assistant|>` et `<|end|>`. Le modèle voit les conversations ainsi (format dans `Chat/ChatFormat.cs`) :

```
<|doc|><|user|>Bonjour !<|end|><|assistant|>Bonjour, comment vas-tu ?<|end|><|user|>...
```

- Le fine-tuning convertit `dialogues.txt`, qui reste au format lisible `Toi:` / `IA:`. **Une ligne vide
  sépare deux conversations** (chacune commence par `<|doc|>`). Une ligne qui ne commence ni par `Toi:`
  ni par `IA:` continue le message d'avant.
- La loss ne compte que les réponses de l'IA et le `<|end|>` qui les termine : le modèle apprend à répondre
  et à s'arrêter. La génération s'arrête quand il écrit `<|end|>`.
- Dans `texte.txt`, `<|doc|>` devant chaque document dit au modèle qu'un nouveau texte commence.
  DataBuilder l'écrit (un livre Gutenberg, une valeur Parquet = un document). Un `texte.txt` fait avant
  n'en a pas : il marche, mais refais-le avec DataBuilder pour en profiter.
- À l'écran, dans les questions de test et `/save-chat`, les messages gardent les noms Toi et IA.
- Un modèle entraîné avant les tokens spéciaux ne peut plus être chargé (son vocabulaire n'a pas de place
  pour eux) : réentraîne-le avec `/train`.

### Tailles du modèle (`/train --size`)

Trois tailles (`ModelConfig.Sizes`), toutes avec des têtes d'attention de 64 dimensions :

| Taille   | EmbedDim | Couches | Têtes | Contexte | Paramètres* | Batch par étape | LR max | Warmup |
|----------|----------|---------|-------|----------|-------------|-----------------|--------|--------|
| `small` (par défaut) | 512 | 8 | 8 | 256 | ~30 M | 16 x 4 extraits = 16 384 tokens | 1e-3 | 100 pas |
| `medium` | 640 | 10 | 10 | 512 | ~55 M | 8 x 8 extraits = 32 768 tokens | 6e-4 | 1000 pas |
| `large`  | 768 | 12 | 12 | 512 | ~91 M | 4 x 16 extraits = 32 768 tokens | 6e-4 | 1000 pas |

\* avec 8000 tokens de vocabulaire. La vitesse descend en cosinus jusqu'à LR max / 10. Le fine-tuning garde
ses réglages (2e-4, 2,5 passages) avec les mêmes paquets que le pré-entraînement (8 ou 4 extraits) ; le texte
général y est arrondi à 2 extraits sur 8 (1 sur 4 pour large), soit 25 %.

Avant de commencer, `/train` affiche le nombre exact de paramètres, le nombre d'étapes de chaque phase et la
durée estimée, puis demande confirmation (`o` / `n`, Entrée = oui). L'estimation (`Training/TrainingEstimate.cs`)
compte les calculs d'une étape (~6 opérations par paramètre et par token, plus l'attention) au rythme mesuré
avec small sur une RTX 3060 Ti, et les tokens d'après la taille des fichiers : un ordre de grandeur
(medium : ~14 h de pré-entraînement pour 1,1 passage du `texte.txt` actuel, large : ~24 h).

Mémoire GPU : si elle dépasse 7 Go (toute la carte, `GpuMemoryLimitMb`) après le premier pas, les paquets
sont divisés par deux et l'accumulation doublée (8 x 8 -> 4 x 16) : même nombre de tokens par étape, même
nombre d'étapes prévu. Un message le dit. Les tokens de `texte.txt` (~1,8 Go) sont sur le GPU et comptent dans
ces 7 Go.

Les modèles déjà entraînés gardent leur taille (`config.json`). `/finetune`, `/continue` et `/resume`
reconnaissent la taille d'un modèle à ses dimensions et reprennent ses réglages.

Le modèle small apprend à chaque étape sur 64 extraits de 256 tokens (16 384 tokens), envoyés au GPU par
paquets de 16 (accumulation de gradients, voir `BatchSize` / `EffectiveBatchSize`). Sur une RTX 3060 Ti 8 Go :
~3,2 Go de mémoire GPU (toute la carte) plus ~1,9 Go pour les tokens de `texte.txt`, 0,42 s par étape avec TF32
(~39 000 tokens/s, 0,53 s sans), soit ~6 h 30 de pré-entraînement pour 1,1 passage du `texte.txt` actuel
(mesuré avec le tokenizer de 4000 tokens). Des paquets de 32 ne vont pas plus vite.
La mémoire GPU utilisée s'affiche à chaque mesure de loss.

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
   `MiniChat.exe --train`, `--finetune`, `--resume`, `--continue <pas>`, `--continue-finetune <pas>`, `--lr <vitesse>`, `--epochs <passages>`,
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
- `<date>-loss.csv` : les loss mesurées pendant l'entraînement (s'ouvre dans Excel pour tracer les courbes),
  avec la vitesse (`tokens_par_seconde`, entre deux mesures) et la mémoire GPU utilisée (`memoire_gpu_mo`, toute la carte)
  et les passages faits sur les données (`passages`).

Pendant l'entraînement, une réponse d'exemple à la première question s'affiche tous les 1000 pas
(`SampleInterval` dans `TrainingConfig`).

## Préparer les données (DataBuilder)

Un petit projet console à part, dans `DataBuilder/` :

```bash
# Dialogues OpenAssistant (fichier *.messages.jsonl ou .jsonl.gz) -> format Toi:/IA:, une ligne vide entre deux conversations
dotnet run --project DataBuilder -- oasst Data/dialogues.txt 2023-11-05_oasst2_ready.messages.jsonl.gz --lang fr --max 300

# Conversations en JSON (.jsonl, .jsonl.gz ou .json), ajoutées à la suite du fichier précédent
dotnet run --project DataBuilder -- chat Data/dialogues.txt conversations-fr.jsonl --append oui

# Livres du Projet Gutenberg (.txt) -> texte propre pour le pré-entraînement
dotnet run --project DataBuilder -- gutenberg Data/texte.txt pg4650.txt pg17489.txt

# Colonne de texte de fichiers .parquet (Wikipedia, FineWeb... sur Hugging Face), arrêt à 50 Mo
dotnet run --project DataBuilder -- parquet Data/texte.txt train-00000-of-00002.parquet --column text --max-mo 50
```

- **oasst** suit la réponse la mieux classée de chaque conversation, ignore les messages supprimés,
  refusés, trop longs, avec du code ou des caractères exotiques (emoji...). OASST2 (2023-11-05) contient
  déjà tout OASST1 : inutile de donner les deux. Il n'a que ~300 conversations françaises utilisables.
- **chat** lit des `.jsonl`, `.jsonl.gz`, `.json` ou `.parquet` aux formats de conversation courants sur Hugging Face : `{"messages": [{"role", "content"}]}`,
  ShareGPT `{"conversations": [{"from", "value"}]}` et Alpaca `{"instruction", "input", "output"}`.
  Les messages `system` sont ignorés, les conversations en double rejetées, et le nettoyage est celui
  de **oasst** (comme lui, le Markdown est retiré et les formules LaTeX rejetées). `--field` choisit le champ des messages s'il a un autre nom (s'il ne reconnaît rien, il
  affiche les champs trouvés). `--append oui` ajoute à la fin du fichier au lieu de l'écraser.
  `--skip "champ=a,b;champ2=c"` écarte les conversations dont le champ contient une de ces valeurs.
  Le « œ » mal encodé de certains jeux (`cÅ“ur`, `Åufs`) est réparé.

Le `dialogues.txt` actuel (~292 000 conversations, 92 Mo) se refait ainsi :

```bash
# OpenAssistant 2 : https://huggingface.co/datasets/OpenAssistant/oasst2 (Apache 2.0)
dotnet run --project DataBuilder -- oasst Data/dialogues.txt 2023-11-05_oasst2_ready.messages.jsonl.gz
# French Alpaca 110K : https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K (Apache 2.0)
# French Instruct : https://huggingface.co/datasets/angeluriot/french_instruct (MIT), sans la logique formelle,
#   le code, les problèmes de maths en plusieurs étapes et les réponses Reddit (style "human")
dotnet run --project DataBuilder -- chat Data/dialogues.txt Acquiesce_data_110k_instructions.json dataset.parquet --append oui \
    --skip "source=Logic Inference,Evol Instruct,Grade School Math;style=human"
# Luth-SFT : https://huggingface.co/datasets/kurakurai/luth-sft (ODC-BY), fichiers de data/, sans luth_scholar
#   ni luth_tulu3_persona_math (maths) : ~157 000 conversations de plus
dotnet run --project DataBuilder -- chat Data/dialogues.txt luth_aya_dataset-00000-of-00001.parquet luth_croissantllm-00000-of-00001.parquet \
    luth_smoltalk2-00000-of-00001.parquet luth_tulu3_persona_instruct-00000-of-00001.parquet \
    luth_openhermes-00000-of-00002.parquet luth_openhermes-00001-of-00002.parquet --append oui
```

Le `texte.txt` actuel (2,9 Go, 565 611 documents) est la mise bout à bout de deux sorties de **parquet** : l'ancien
`texte.txt` (~0,1 Go de web FineWeb-2, puis ~1,3 Go de Wikipédia FR, 192 872 documents : ses 1 468 518 614 premiers
octets), puis 1,4 Go de web de qualité (372 739 documents). Les extraits sont tirés au hasard dans tout le fichier :
l'ordre ne compte pas.

```bash
# FineWeb2-HQ : https://huggingface.co/datasets/epfml/FineWeb2-HQ (ODC-BY), les 10 % meilleures pages françaises
# de FineWeb-2. fra_Latn/000_00000 à 000_00002 : ~2 Go chacun, mais seulement ~0,5 Go de texte (le reste : d'autres colonnes)
dotnet run --project DataBuilder -- parquet texte-hq.txt 000_00000.parquet 000_00001.parquet 000_00002.parquet
cat texte-hq.txt >> Data/texte.txt
```
- **gutenberg** retire l'en-tête et la licence, recolle les paragraphes (un par ligne),
  simplifie la typographie (’ … — -> ' ... -) et lit aussi les vieux fichiers en Latin-1.
- **parquet** lit la colonne `--column` (par défaut `text`) ; un fichier sans cette colonne mais avec une
  seule colonne de texte (TinyStories : `french-tinystories`) utilise celle-là. Un document par valeur,
  nettoyé comme pour Gutenberg, une ligne du document = un paragraphe. `--max-mo` arrête avant le document qui
  ferait dépasser la taille ; sans cette option, tout est gardé. Affiche le nombre de documents gardés.

Attention : la sortie est **écrasée** (sauf avec `chat --append oui`). Garde une copie de tes `dialogues.txt` faits à la main.

## Organisation du code

| Dossier          | Classe                | Rôle |
|------------------|-----------------------|------|
| `Config/`        | `ModelConfig`         | Taille du réseau (couches, têtes, mémoire) |
|                  | `TrainingConfig`      | Nombre d'étapes, vitesse d'apprentissage |
| `Tokenization/`  | `BpeTokenizer`        | Texte ↔ nombres, par morceaux de mots (BPE appris sur nos textes) |
| `Data/`          | `TextDataset`         | Découpe le texte en extraits pour l'entraînement |
| `Model/`         | `CausalSelfAttention` | L'attention : regarder les tokens d'avant |
|                  | `FeedForward`         | Petit réseau appliqué à chaque position, GELU (Classic) |
|                  | `SwiGlu`              | Le même rôle, avec une « vanne » SiLU (Modern) |
|                  | `RmsNorm`             | Normalisation sans moyenne ni biais (Modern ; Classic : LayerNorm) |
|                  | `Rope`                | Positions par rotation de Q et K dans l'attention (Modern) |
|                  | `KvCache`             | Keys/Values déjà calculées, pour générer sans tout recalculer |
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

## Architecture (`ModelConfig.Architecture`)

| | `Modern` (par défaut) | `Classic` |
|---|---|---|
| Position | RoPE : Q et K tournés selon la position, dans l'attention | Embedding de position appris, ajouté à l'entrée |
| Normalisation | RMSNorm | LayerNorm |
| FeedForward | SwiGLU (3 matrices, largeur 8/3 x EmbedDim) | GELU (2 matrices, largeur 4 x EmbedDim) |
| Biais | aucun | dans la projection de l'attention et le FeedForward |

Les deux ont à peu près le même nombre de paramètres. Pour comparer : `/train`, `/save modern`, passer
`Architecture` à `ModelArchitecture.Classic` dans `Config/ModelConfig.cs`, `/train`, puis `/compare model modern`
et `/stats`. Le choix est écrit dans le `config.json` de chaque modèle ; un `config.json` sans ce réglage
(modèle d'avant) est relu en `Classic`.

La génération utilise un **cache KV** : le prompt passe une fois dans le modèle, puis chaque nouveau token
n'y fait passer que lui-même. La vitesse (tokens/s) s'affiche sous chaque réponse, et dans `/info`.

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
