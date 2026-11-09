# Rapport technique — MiniChat / Nel Code (entraînement)

État au 2026-11-09, commit `171f09d`. Lecture seule : aucun code modifié.
Sources des chiffres : code, `bin/Debug/net8.0/{model,model-pretrain}/*.json`, `bin/Debug/net8.0/results/*`,
`nvidia-smi`, et un comptage de tokens fait hors du dépôt avec le tokenizer de `model/` (`BpeTokenizer`, `ChatFormat` du projet).

Légende : **mesuré** = lu ou calculé sur les fichiers ; **estimé** = déduit ; **non enregistré** = l'info n'existe nulle part sur disque.

---

## 1. Matériel et vitesse

| Point | Valeur | Où |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 192 Mo, pilote 621.01 (mesuré, `nvidia-smi`) | `Program.cs` (`cuda.is_available() ? CUDA : CPU`) |
| Pile logicielle | .NET 8, TorchSharp-cuda-windows 0.105.*, CUDA 12 | `MiniChat.csproj` |
| Mémoire GPU max | **Non enregistrée** pour le dernier entraînement : affichée en console seulement (`_peakGpuMb`), absente du CSV. README : ~3,2 Go (carte entière, Windows compris) | `Training/Trainer.cs` › `Trainer.GpuMemoryNow`, `Training/GpuMemory.cs` › `GpuMemory.Read` (`cudaMemGetInfo`) |
| Tokens par étape | 64 extraits × 256 tokens = 16 384 | `Config/TrainingConfig.cs` › `EffectiveBatchSize`, `ModelConfig.BlockSize` |

Vitesse mesurée (colonne `secondes` des CSV ; inclut évaluations, checkpoints et exemples) :

| Entraînement | Pas | Durée | s / étape | tokens / s |
|---|---|---|---|---|
| Fine-tuning `2026-11-09_16-23-41` (le dernier) | 0 → 2 000 | 1 078 s | **0,539** | **~30 400** |
| Continuation pré-entraînement (dernière, `2026-11-08_18-07-56-loss.csv`) | 16 000 → 23 200 | 3 780 s | 0,525 | ~31 200 |
| Pré-entraînement initial (`/train`) | 0 → 9 000 | 4 777 s | 0,531 | ~30 900 |

Le README et `TrainingConfig.Steps` annoncent 0,46 s/étape (mesure plus ancienne).

## 2. Données

Dossier lu à l'exécution : `bin/Debug/net8.0/Data/` (copie identique de `Data/` ; `Program.cs`, défaut `AppContext.BaseDirectory/data`).

| Fichier | Taille | Tokens (tokenizer de `model/`) | Rôle |
|---|---|---|---|
| `texte.txt` | 1 400,5 Mo (1 468 518 614 o), 1 426,5 M caractères | **459 093 369** (mesuré) | Pré-entraînement + texte général du fine-tuning |
| `dialogues.txt` | 34,9 Mo (36 558 479 o), 35,6 M caractères | **11 341 631** (mesuré, après `ChatFormat.FromDialogueFile`) | Fine-tuning |
| `000_00000.parquet` | 60,2 Mo | — (source brute) | Texte web (nommage FineWeb-2 ; contenu : articles web FR) |
| `train-00000-of-00017.parquet` | 733,5 Mo | — (source brute) | Wikipédia FR (URLs `fr.wikipedia.org` dans le pied du fichier) |
| `train-00000-of-00001-73cdb762aecc853c.parquet` | 0,55 Mo | — (source brute) | Non identifié avec certitude (texte narratif ; probablement TinyStories FR, cf. `ParquetConverter`) |
| `questions-test.txt` | 490 o | — | 10 questions de test |

Sources et combinaison :

| Point | Détail | Où |
|---|---|---|
| `texte.txt` | Sortie de `DataBuilder parquet` : un document par valeur Parquet, précédé de `<|doc|>`, un paragraphe par ligne. 192 872 documents (mesuré). Ordre estimé (densité des « Notes et références ») : ~0,1 Go de web au début, puis ~1,3 Go de Wikipédia. Part TinyStories non vérifiable. Concaténation bout à bout, sans pondération | `DataBuilder/ParquetConverter.cs` › `ParquetConverter`, `DataBuilder/Program.cs` |
| `dialogues.txt` | OpenAssistant 2 (`oasst`) + French Alpaca + French Instruct (`chat --append oui`), format `Toi:` / `IA:`, ligne vide entre conversations. 135 606 conversations, 137 160 tours (mesuré). Réponses coupées à `--max 300` caractères / `--sentences 2` par défaut | `DataBuilder/OasstConverter.cs`, `DataBuilder/ChatConverter.cs`, commit `171f09d` |
| Lecture `texte.txt` | Ligne par ligne, ids en `ushort[]` (~0,9 Go RAM), ~54 s de tokenisation (mesuré en Release) ; refait à chaque phase | `Data/TextDataset.cs` › `TextDataset.FromFile` |

Séparation entraînement / vérification (`Data/TextDataset.cs` › `TextDataset.Split`) :

- Morceaux de 10 000 tokens (`ChunkTokens`) ; dans chaque groupe de 10 morceaux, 1 tiré au hasard va en vérification (~10 %).
- Graine fixe `SplitSeed = 1234`, indépendante de la graine des batchs : découpage identique à chaque lancement.
- Un extrait ne chevauche jamais les deux côtés (segments + `StartsIndex` / `PickStart`).
- Volumes estimés : `texte.txt` ~413 M tokens entraînement / ~46 M vérification ; `dialogues.txt` ~10,2 M / ~1,1 M.

## 3. Tokenizer

`Tokenization/BpeTokenizer.cs` › `BpeTokenizer` ; construit dans `App/Workbench.cs` › `Workbench.CreateModel`.

| Point | Valeur |
|---|---|
| Type | BPE sur **caractères** (pas sur octets), pré-découpage regex (mot avec espace devant, nombres, ponctuation, `\n` seul) ; caractères inconnus ignorés |
| Vocabulaire | **4 000** = 155 caractères de base + 3 841 fusions + 4 spéciaux (`TokenizerVocabSize`) |
| Tokens spéciaux | `<|doc|>`, `<|user|>`, `<|assistant|>`, `<|end|>` (ids en fin de vocabulaire) |
| Données d'entraînement | Échantillon de 20 Mo de `texte.txt` (20 blocs de 1 Mo répartis, `ReadSample`, `TokenizerSampleMb`) + **tout** `dialogues.txt` formaté (~40 M caractères) |
| Caractères / token (mesuré) | `texte.txt` : **3,11** ; `dialogues.txt` : **3,14** |
| Algorithme | Recompte de toutes les paires à chaque fusion (~1 min 30 selon le commentaire) |

## 4. Modèle

`Config/ModelConfig.cs` › `ModelConfig` (valeurs de `model/config.json` = valeurs par défaut) :

| Champ | Valeur |
|---|---|
| VocabSize | 4 000 |
| BlockSize | 256 |
| EmbedDim | 512 |
| Heads | 8 (HeadDim 64) |
| Layers | 8 |
| Dropout | 0,1 |
| Architecture | Modern |
| SwiGluHidden (calculé) | 1 408 |

Paramètres : **27 746 816** (mesuré, ligne d'en-tête des résultats ; recalculé : embedding 2 048 000 + 8 × 3 212 288 + norme finale 512).

| Élément | Implémentation | Où |
|---|---|---|
| Structure | Pré-norm, résiduel `x + attn(norm(x))`, `x + ffn(norm(x))` | `Model/TransformerBlock.cs` › `TransformerBlock` |
| Normalisation | RMSNorm (eps 1e-5, poids appris, sans biais), aussi en sortie | `Model/RmsNorm.cs` › `RmsNorm` |
| Attention | Multi-têtes causale, QKV fusionné, sans biais ; **`functional.scaled_dot_product_attention` utilisé** (`is_causal`, dropout 0,1 à l'entraînement) | `Model/CausalSelfAttention.cs` › `CausalSelfAttention` |
| Positions | RoPE, θ = 10 000, rotation par moitiés, table cos/sin précalculée | `Model/Rope.cs` › `Rope` |
| Feed-forward | SwiGLU (gate/up/down, sans biais) | `Model/SwiGlu.cs` › `SwiGlu` |
| Weight tying | Oui : logits = `linear(norm(x), tokenEmbedding.weight)` | `Model/MiniGpt.cs` › `MiniGpt.Forward` |
| Dropout | Embeddings, attention (SDPA), sortie de l'attention, sortie du FFN | `MiniGpt`, `CausalSelfAttention`, `SwiGlu` |
| Initialisation | Embedding N(0, 0,02) ; couches `Linear` : init par défaut de TorchSharp | `MiniGpt` (constructeur) |
| Génération | Cache KV (hors entraînement) | `Model/KvCache.cs` › `KvCache` |

## 5. Boucle d'entraînement

Boucle : `Training/Trainer.cs` › `Trainer.Train` ; réglages : `Config/TrainingConfig.cs` › `TrainingConfig` (`Pretraining`, `FineTuning`, `Continuation`) ; orchestration : `App/Workbench.cs` › `Pretrain`, `FineTune`, `ContinuePhase`.

| Réglage | Pré-entraînement | Fine-tuning | Continuation (`/continue`) |
|---|---|---|---|
| Optimiseur | AdamW (TorchSharp) | idem | idem, état rechargé depuis `optimizer.dat` |
| Betas / eps | défaut TorchSharp : (0,9 ; 0,999), 1e-8 | idem | idem |
| Weight decay | 0,1 sur les paramètres de dim ≥ 2 (toutes les `Linear` + embedding lié) ; 0 sur les vecteurs (poids RMSNorm) | idem | idem |
| LR max → min | 1e-3 → 1e-4 | 2e-4 → 2e-5 | `--lr` ou min(3e-4, LR de la phase) → LR/10 |
| Warmup | 100 pas, linéaire | 50 pas | min(200, pas/2) |
| Schedule | Cosinus jusqu'à `Steps` | idem | Cosinus de `FirstStep` à `FirstStep + pas` |
| Batch / accumulation | 16 × 4 = 64 extraits | idem (13 dialogues + 3 texte général par micro-batch) | idem |
| Contexte | 256 tokens | 256 | 256 |
| Gradient clipping | norme 1,0 (`clip_grad_norm_`) | idem | idem |
| Étapes | 9 000 (`Steps`) | 2 000 | nombre demandé |
| EvalInterval / EvalBatches | 100 / 20 (batchs de 16) | 50 / 20 | ceux de la phase |
| Arrêt anticipé (patience) | 2 000 pas | 300 pas | celle de la phase |
| Checkpoint / exemple | tous les 250 / 1 000 pas | 250 / 250 | ceux de la phase |
| Graine | 42 + pas de départ | idem | idem |

Points communs :

- **Précision** : float32 partout (aucun autocast, float16 ou bfloat16 dans le code).
- **Batchs** : construits sur **CPU** (tableaux `long[]`/`float[]`, début d'extrait **aléatoire uniforme**, avec remise), puis `tensor(...).to(device)` à chaque micro-batch — `Data/TextDataset.cs` › `TextDataset.GetBatch`.
- **Loss** : cross-entropy par token × masque, divisée par la somme du masque — `Trainer.ComputeLoss`.
- **Évaluation** : mesure au premier pas, tous les `EvalInterval` pas et au dernier ; vérification sur le jeu principal seulement — `Trainer.Report`, `Trainer.EstimateLoss`.
- **Meilleur modèle** : poids écrits dans `checkpoint/best.dat` à chaque amélioration, remis à la fin ; une continuation ne remplace le modèle que si la loss de vérification baisse — `Trainer.Train`, `Workbench.SaveResult`.

Contenu sauvegardé (`Training/Checkpoint.cs` › `Checkpoint`, `Storage/ModelStore.cs` › `ModelStore`) :

| Dossier | Fichiers |
|---|---|
| `checkpoint/` (en cours, supprimé à la fin) | `config.json`, `tokenizer.json`, `weights.dat` (poids actuels), `optimizer.dat`, `best.dat` + `best.json`, `state.json` (phase, pas, RunId, StartValLoss), `continue.json` (si continuation), `final-optimizer.dat` |
| `model/`, `model-pretrain/` | `config.json`, `tokenizer.json`, `weights.dat` (111 Mo, meilleurs poids), `optimizer.dat` (222 Mo, état AdamW du **dernier** pas), `training.json` (pas, loss de vérification, RunId) |

## 6. Fine-tuning

| Point | Détail | Où |
|---|---|---|
| Format | `<|doc|><|user|>question<|end|><|assistant|>réponse<|end|>…` ; `<|doc|>` par conversation | `Chat/ChatFormat.cs` › `ChatFormat.FromDialogueFile` |
| Masquage | Poids 1 sur les tokens de la réponse et le `<|end|>` qui la termine ; 0 sur `<|doc|>`, `<|user|>`, la question, `<|assistant|>`. Un token prend le poids du dernier caractère de son mot. **60,8 %** des tokens de `dialogues.txt` sont notés (mesuré) | `ChatFormat.ReplyWeights`, `TextDataset` (constructeur, `onlyReplies`) |
| Texte général | `GeneralTextRatio = 0,2` → `round(16 × 0,2) = 3` extraits sur 16 par micro-batch (**18,75 %** des extraits ; ~27 % des tokens notés, le texte général étant noté à 100 %) | `Trainer.GetTrainingBatch` |
| Taille des réponses (mesuré) | moyenne **49,3** tokens, médiane 50, P90 81, max 232 | — |
| Taille des questions (mesuré) | moyenne 28,4 tokens ; conversation moyenne ~84 tokens (une fenêtre de 256 en contient ~3) | — |
| Passages | 2 000 pas × 52 extraits × 256 ≈ 26,6 M tokens de dialogues ≈ 2,6 passages | `TrainingConfig.FineTuning` |

## 7. Résultats

Mesures des CSV `results/*-loss.csv` (loss de vérification sauf mention).

| Phase (dernier entraînement) | Fichier / pas | Loss de départ | Meilleure (pas) | Finale (entraîn. / vérif.) | Arrêt anticipé |
|---|---|---|---|---|---|
| Pré-entraînement (`/train`) | `2026-11-08_18-07-56`, 0 → 9 000 | 8,4006 | **2,5929** (8 900) | 2,6211 / 2,6690 | Non (fin des pas) |
| Continuation pré-entraînement (`/continue 100000`) | même CSV, 16 000 → 23 200 (prévu 116 000) | 2,5914 | **2,4647** (21 200) → `model-pretrain/` | 2,6140 / 2,5596 | **Oui** (23 200 − 21 200 = 2 000) |
| Fine-tuning (`/finetune`) | `2026-11-09_16-23-41`, 0 → 2 000 | 3,1145 | **1,8174** (1 750) → `model/` | 1,7370 / 1,8492 | Non (250 < 300) |

Aucune continuation de fine-tuning (`/continue … finetune`) dans les CSV.

Dernières réponses aux questions de test : aucun fichier `*-test.txt` (commande `/test`) dans `results/` ; les plus récentes sont celles générées à la fin du fine-tuning, `results/2026-11-09_16-23-41.txt` (température 0,7, top-k 20, top-p 0,9, graine 1234) :

| # | Question | Réponse |
|---|---|---|
| 1 | Bonjour ! | Je suis MiniChat, un assistant qui répond en français à des questions sur l'amitié, la compétence et la compétence de chacun. |
| 2 | Comment tu t'appelles ? | Je suis programmé pour comprendre les avantages de l'assistant qui répond en français. Tu peux répondre à vos besoins et à vos besoins en fonction de vos besoins et de vos besoins. |
| 3 | Tu es qui ? | Je suis Mini Chat, un assistant qui répond en français à des questions sur l'amélioration de la santé physique et mentale. |
| 4 | Comment ça va aujourd'hui ? | Il y a plusieurs façons de faire un plan pour lire un livre, y compris les livres et les livres, de faire un bon travail et de partager un livre avec des amis. |
| 5 | Qu'est-ce que tu aimes faire ? | Je suis MiniChat, un assistant qui répond en français en anglais. |
| 6 | Peux-tu m'expliquer ce qu'est une étoile ? | Une étoile est une planète qui appartient à la même planète et à sa propre planète. Elle est composée de huit planètes, chacune ayant un corps unique et un corps unique. |
| 7 | Quel temps fait-il chez toi ? | Je suis désolé, mais je n'ai pas la capacité de t'aider à répondre à cette question. |
| 8 | Donne-moi un conseil pour bien dormir. | Prendre des pauses régulières, organiser une journée avec un plat de pâtes, faire de l'exercice régulièrement, préparer des pâtes avec des pâtes à la pâtes, faire de l'exercice régulièrement, faire de l'exercice régulièrement, faire de l' *(coupée)* |
| 9 | Raconte-moi une petite histoire. | Il était une fois, dans une forêt lointaine, un jeune garçon qui avait l'habitude de se promener dans le parc. En chemin, il se promenait dans le parc, mais il avait tout à fait la chance de se promener dans le parc. |
| 10 | Merci et au revoir ! | Merci ! Nous sommes ravis de vous aider à vous assister à des actions de sociétés privées et de soutien pour l'entreprise. |

Constat : format respecté et arrêt sur `<|end|>`, mais contenu peu factuel et répétitions (2, 8, 9).

## 8. Points faibles ou lents (non corrigés)

- **float32 seulement** (`Trainer`, `MiniGpt`) : pas de bfloat16 / autocast alors que la RTX 3060 Ti (Ampere) le gère ; SDPA en float32 ne peut pas utiliser le noyau Flash.
- **Batchs sur CPU** (`TextDataset.GetBatch`) : boucle C# token par token, `long[]` (int64), copie synchrone vers le GPU à chaque micro-batch, sans mémoire épinglée.
- **Peu de données vues** : 9 000 pas × 16 384 = 147 M tokens au pré-entraînement initial (~0,36 passage de `texte.txt`), ~380 M (23 200 pas, < 1 passage) avec les continuations gardées.
- **Évaluation bruitée** (`EvalBatches = 20` × 16 extraits ≈ 82 k tokens) : la loss de vérification oscille de ±0,05 d'une mesure à l'autre ; le « meilleur » pas et l'arrêt anticipé dépendent de ce bruit.
- **Continuation de 100 000 pas** : cosinus étalé sur 100 000 pas → LR quasi constant (~3e-4) jusqu'à l'arrêt anticipé, aucune phase de décroissance.
- **Fine-tuning** : vérification sur les dialogues seulement (l'oubli du texte général n'est pas mesuré) ; part réelle de texte général 18,75 % des extraits mais ~27 % des tokens notés.
- **Fenêtres à cheval sur les documents** : un extrait de 256 tokens mélange ~3 conversations (ou plusieurs documents) sans masque d'attention entre elles.
- **Tokenizer** : entraîné sur plus de dialogues (~36 Mo) que de `texte.txt` (20 Mo) ; algorithme naïf (recompte complet à chaque fusion) ; caractères hors des 155 de base supprimés silencieusement.
- **Tokenisation de `texte.txt` refaite à chaque phase** (~1 min, deux fois pour `/train`), sans cache des ids.
- **Checkpoints lourds et synchrones** : ~333 Mo (poids + AdamW) tous les 250 pas, plus 111 Mo à chaque nouveau meilleur score.
- **AdamW β2 = 0,999** (défaut) et weight decay appliqué à l'embedding lié ; init des `Linear` par défaut, sans mise à l'échelle des projections résiduelles.
- **Dropout 0,1 au pré-entraînement** alors qu'on fait moins d'un passage sur les données.
- **Suivi incomplet** : pic de mémoire GPU et tokens/s non écrits dans le CSV ; `optimizer.dat` correspond au dernier pas, pas aux poids gardés.
- **Documentation désynchronisée** : `TrainingConfig.Steps` et README parlent de 20 000 pas et 0,46 s/étape (valeur réelle : 9 000, ~0,53 s) ; README situe `TokenizerVocabSize` dans `Program.cs` (il est dans `Workbench`) ; `Trainer` parle encore de « caractère suivant ».
