# Format .nelmodel

Un fichier `.nelmodel` contient un modèle prêt à discuter, exporté par MiniChat et installé dans Nel Code
avec `/import <fichier.nelmodel> [nom]`.

## Contenu

Une archive **zip**, avec ces quatre fichiers **à la racine** (pas dans un sous-dossier) :

| Fichier | Contenu |
|---|---|
| `config.json` | la forme du réseau (`ModelConfig` de MiniChat) |
| `tokenizer.json` | le tokenizer BPE |
| `weights.dat` | les poids (format TorchSharp, `module.save`) |
| `model-info.json` | la description du modèle, voir plus bas |

Les quatre sont obligatoires. Il n'y a **pas d'état d'optimiseur** (`optimizer.dat`) : un `.nelmodel` sert
à générer, pas à reprendre un entraînement. Les autres fichiers éventuels de l'archive sont ignorés.

### config.json

```json
{
  "VocabSize": 4000,
  "BlockSize": 256,
  "EmbedDim": 512,
  "Heads": 8,
  "Layers": 8,
  "Dropout": 0.1,
  "Architecture": "Modern"
}
```

`Architecture` vaut `"Modern"` (RoPE, RMSNorm, SwiGLU) ou `"Classic"` (comme GPT-2). Si la clé manque, le
modèle est lu comme `Classic`.

### tokenizer.json

```json
{
  "Alphabet": "\n !\"'(),-.0123…",
  "Merges": [[12, 5], [40, 3], …],
  "Special": ["<|doc|>", "<|user|>", "<|assistant|>", "<|end|>"]
}
```

- `Alphabet` : les caractères de base, ids `0` à `Alphabet.Length - 1`.
- `Merges[i]` : la fusion n° i, `[gauche, droite]`, qui crée le token `Alphabet.Length + i`.
- `Special` : les tokens spéciaux, ids juste après les fusions. **Obligatoire** : sans lui, le modèle ne
  connaît pas le format de conversation et Nel Code le refuse.

Le nombre total de tokens (`Alphabet` + `Merges` + `Special`) doit être égal à `VocabSize` de `config.json`.

### weights.dat

Les poids du `MiniGpt`, écrits par TorchSharp. Leurs noms (`_tokenEmbedding`, `_blocks`, `_qkv`, `_gate`,
`weight`…) sont ceux des champs des modules de MiniChat. Le chargement est strict : un poids manquant, en trop
ou de mauvaise forme est une erreur.

### model-info.json

```json
{
  "formatVersion": 1,
  "name": "minichat-v2",
  "createdAt": "2026-11-10T14:00:00+01:00",
  "parameters": 27746816,
  "architecture": "Modern",
  "valLoss": 4.671,
  "trainingSteps": 400,
  "description": "Pré-entraîné sur texte.txt, fine-tuné sur dialogues.txt"
}
```

| Clé | Type | Rôle |
|---|---|---|
| `formatVersion` | entier | **obligatoire**, version de ce format : `1` |
| `name` | texte | nom proposé à l'import (sinon : le nom du fichier sans `.nelmodel`) |
| `createdAt` | date ISO 8601 | date de l'export |
| `parameters` | entier | nombre de paramètres |
| `architecture` | texte | `"Modern"` ou `"Classic"` |
| `valLoss` | nombre | loss de vérification des poids exportés |
| `trainingSteps` | entier | nombre de pas d'entraînement |
| `description` | texte | une phrase libre |

Seul `formatVersion` est obligatoire. Les autres clés sont affichées à l'import, mais ce qui compte pour
charger le modèle reste `config.json` : en cas de désaccord, c'est lui qui fait foi. Seul `parameters` est
contrôlé : s'il ne correspond pas au vrai nombre de paramètres, Nel Code installe le modèle mais affiche un
avertissement.

## Import dans Nel Code

`/import <fichier.nelmodel> [nom] [--replace]` :

1. ouvre le zip (refus s'il n'est pas un zip valide) ;
2. vérifie que les quatre fichiers sont présents (refus qui les nomme sinon) ;
3. lit `formatVersion` : refus si elle manque ou est plus récente que celle que Nel Code connaît
   (« mets Nel Code à jour ») ;
4. affiche les infos de `model-info.json` ;
5. si le nom est déjà pris, demande : remplacer, choisir un autre nom, ou annuler (`--replace` remplace
   sans demander) ;
6. extrait les quatre fichiers dans un dossier temporaire et charge le modèle pour le tester ;
7. si tout va bien, installe le modèle dans `%APPDATA%\NelCode\models\<nom>`. Sinon, rien n'est gardé, et un
   modèle qu'on voulait remplacer reste tel quel ;
8. avertit si `parameters` ne correspond pas au modèle chargé.

Le nom vient de l'argument `[nom]`, sinon de `name`, sinon du nom du fichier. L'ancien modèle n'est supprimé
qu'une fois le nouveau vérifié. Si c'était le modèle actif, il est rechargé.

Pour les tests, `/import` accepte aussi directement un dossier de modèle MiniChat (`config.json`,
`tokenizer.json`, `weights.dat`).

## Versions

| Version | Changement |
|---|---|
| 1 | première version |

Une nouvelle version n'est nécessaire que si un ancien Nel Code ne peut pas lire correctement le nouveau
contenu. Ajouter une clé facultative à `model-info.json` ne change pas la version.
