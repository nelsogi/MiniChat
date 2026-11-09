namespace MiniChat.Config;

/// <summary>
/// Réglages de l'entraînement. Les valeurs par défaut sont celles du pré-entraînement (Pretraining) ;
/// FineTuning et Continuation en changent une partie.
/// </summary>
public sealed record TrainingConfig
{
    /// <summary>Vitesse d'apprentissage par défaut pour continuer un modèle déjà entraîné (--continue).</summary>
    public const double ContinueLearningRate = 3e-4;

    /// <summary>Warmup d'une continuation : court, le modèle et AdamW sont déjà "chauds".</summary>
    public const int ContinueWarmupSteps = 200;

    /// <summary>
    /// Continuer un modèle déjà entraîné (--continue, --continue-finetune) : exactement steps pas de plus,
    /// numérotés à partir de firstStep (le pas où il s'était arrêté), avec leur propre courbe complète :
    /// warmup court jusqu'à learningRate, puis cosinus jusqu'à learningRate / 10 au dernier pas demandé.
    /// Vitesse plus basse qu'au premier entraînement : il s'agit d'affiner, pas de bousculer ce qui est appris.
    /// Sans learningRate : ContinueLearningRate, ou la vitesse de la phase si elle est déjà plus basse
    /// (fine-tuning, 2e-4). Warmup réduit à la moitié des pas pour une petite continuation (sinon il ne finirait
    /// jamais). Epochs = null : c'est steps qui fixe la longueur, pas la taille des données.
    /// </summary>
    public TrainingConfig Continuation(int firstStep, int steps, double? learningRate = null)
    {
        double lr = learningRate ?? Math.Min(ContinueLearningRate, LearningRate);
        return this with
        {
            Epochs = null,
            FirstStep = firstStep,
            Steps = firstStep + steps,
            LearningRate = lr,
            MinLearningRate = lr / 10,
            WarmupSteps = Math.Min(ContinueWarmupSteps, steps / 2),
        };
    }

    /// <summary>
    /// Phase 1 : apprendre la langue sur un gros texte général (data/texte.txt).
    /// texte.txt (~413 M tokens d'entraînement) x 1,1 passage / 16 384 tokens par étape = ~27 700 étapes,
    /// soit ~3 h 15 à 0,42 s par étape (RTX 3060 Ti, TF32). Interruptible, reprise avec /resume.
    /// </summary>
    public static TrainingConfig Pretraining => new();

    /// <summary>
    /// Phase 2 : spécialiser le modèle sur les dialogues (data/dialogues.txt + data/identite.txt).
    /// Vitesse plus basse : on ajuste ce qui est déjà appris, sans tout effacer. Dropout du modèle
    /// (ModelConfig.Dropout) : les dialogues sont relus plusieurs fois, il faut éviter de les apprendre par cœur.
    ///
    /// dialogues.txt (OpenAssistant + French Alpaca + French Instruct, ~135 000 conversations) fait
    /// ~10 millions de tokens d'entraînement ; chaque étape en lit 13 312 (13 extraits sur 16, le reste est du
    /// texte général) : 2,5 passages = ~1 900 étapes, ~14 min. L'arrêt anticipé reste un garde-fou si les
    /// dialogues sont petits (un fichier de 30 000 tokens était appris par cœur en 50 étapes).
    /// </summary>
    public static TrainingConfig FineTuning => new()
    {
        Epochs = 2.5,
        LearningRate = 2e-4,
        MinLearningRate = 2e-5,
        WarmupSteps = 50,
        GeneralTextRatio = 0.2,
        Dropout = null,
        Beta2 = 0.999,
        EmbeddingWeightDecay = true,
        EvalInterval = 50,
        EarlyStoppingPatience = 300,
        EarlyStoppingMinEpochs = 0,
        SampleInterval = 250,
    };

    /// <summary>
    /// Part de chaque batch tirée de texte.txt pendant le fine-tuning (0.2 = 20 %, arrondi : 3 extraits sur 16).
    /// Sans ça, le modèle oublie peu à peu le français général appris en phase 1.
    /// </summary>
    public double GeneralTextRatio { get; init; } = 0;

    /// <summary>
    /// Longueur de la phase en passages sur les données d'entraînement (1 = chaque token vu une fois en moyenne).
    /// Steps en est calculé au chargement des données (voir ForData) : la longueur suit la taille de texte.txt,
    /// et le cosinus va toujours jusqu'au dernier pas prévu. --epochs le change. null : Steps tel quel.
    /// 1,1 au pré-entraînement : un passage complet, un peu plus pour finir la descente du cosinus.
    /// </summary>
    public double? Epochs { get; init; } = 1.1;

    /// <summary>
    /// Numéro du dernier pas : la phase va de FirstStep à Steps. Calculé depuis Epochs (voir ForData) ;
    /// cette valeur ne sert que si Epochs est null (continuation : FirstStep + le nombre de pas demandé).
    /// </summary>
    public int Steps { get; init; } = 5000;

    /// <summary>
    /// Numéro du premier pas : 0 pour un nouvel entraînement, le pas atteint avant pour une continuation.
    /// Le warmup et le cosinus partent d'ici.
    /// </summary>
    public int FirstStep { get; init; } = 0;

    /// <summary>
    /// Nombre d'extraits envoyés au GPU en une fois. Limité par la mémoire du GPU :
    /// si "out of memory", baisse-le (EffectiveBatchSize doit rester un multiple).
    /// </summary>
    public int BatchSize { get; init; } = 16;

    /// <summary>
    /// Nombre d'extraits dont le modèle apprend à chaque étape (le "vrai" batch).
    /// Si c'est plus que BatchSize, on accumule les gradients de plusieurs petits batchs avant
    /// d'ajuster les poids : 64 / 16 = 4 petits batchs par étape, comme un batch de 64 d'un coup.
    /// </summary>
    public int EffectiveBatchSize { get; init; } = 64;

    /// <summary>Vitesse d'apprentissage : trop haut = instable, trop bas = lent.</summary>
    public double LearningRate { get; init; } = 1e-3;

    /// <summary>
    /// Weight decay (AdamW) : à chaque étape, les poids sont un peu rapprochés de 0.
    /// Empêche quelques poids de devenir énormes pour apprendre le texte par cœur.
    /// Seulement sur les matrices des couches (voir EmbeddingWeightDecay pour l'embedding).
    /// </summary>
    public double WeightDecay { get; init; } = 0.1;

    /// <summary>
    /// Weight decay aussi sur l'embedding des tokens (qui sert aussi de couche de sortie, weight tying).
    /// Non au pré-entraînement : les tokens rares, peu vus, verraient leur vecteur tiré vers 0 entre deux
    /// apparitions. Oui au fine-tuning, comme avant (et l'état d'AdamW des modèles déjà fine-tunés reste valable).
    /// </summary>
    public bool EmbeddingWeightDecay { get; init; } = false;

    /// <summary>
    /// Betas d'AdamW : mémoire de la moyenne des gradients (Beta1) et de leur taille (Beta2).
    /// 0,95 au pré-entraînement (comme GPT-3, LLaMA) : AdamW s'adapte plus vite quand les gradients changent,
    /// plus stable avec un grand learning rate. 0,999 (valeur par défaut d'AdamW) au fine-tuning.
    /// </summary>
    public double Beta1 { get; init; } = 0.9;

    public double Beta2 { get; init; } = 0.95;

    /// <summary>
    /// Dropout de cette phase (voir DropoutRate). 0 au pré-entraînement : moins d'un passage et demi sur les
    /// données, le modèle ne revoit presque jamais le même texte, il n'y a rien à apprendre par cœur.
    /// null : celui du modèle (ModelConfig.Dropout, 0,1).
    /// </summary>
    public double? Dropout { get; init; } = 0;

    /// <summary>
    /// Gradient clipping : si la "force" totale d'une correction (norme du gradient) dépasse cette valeur,
    /// on la réduit. Évite qu'un batch bizarre fasse faire un pas énorme et casse le modèle.
    /// </summary>
    public double GradClip { get; init; } = 1.0;

    /// <summary>Vitesse d'apprentissage atteinte à la fin (la courbe cosinus descend jusqu'ici).</summary>
    public double MinLearningRate { get; init; } = 1e-4;

    /// <summary>Étapes de "chauffe" : la vitesse monte de 0 à LearningRate pour éviter un départ brutal.</summary>
    public int WarmupSteps { get; init; } = 100;

    /// <summary>
    /// Tous les combien de pas on sauvegarde l'avancement dans checkpoint/ (pour --resume).
    /// Une sauvegarde écrit ~330 Mo (poids + AdamW) : 1000 pas, c'est au plus ~7 min à refaire après une coupure.
    /// Un arrêt avec Échap sauvegarde de toute façon au pas atteint.
    /// </summary>
    public int CheckpointInterval { get; init; } = 1000;

    /// <summary>
    /// Arrêt anticipé : on s'arrête si la loss de vérification ne s'est pas améliorée depuis ce nombre de pas.
    /// Le modèle n'apprend alors plus rien d'utile (ou commence à apprendre par cœur). 0 = jamais.
    /// </summary>
    public int EarlyStoppingPatience { get; init; } = 2000;

    /// <summary>
    /// Pas d'arrêt anticipé tant que le modèle n'a pas fait au moins ce nombre de passages sur les données
    /// (comptés depuis le pas 0). 1 au pré-entraînement : avant un passage complet, chaque extrait est du texte
    /// neuf, une loss qui stagne est du bruit ou un palier, pas le signe qu'il apprend par cœur.
    /// </summary>
    public double EarlyStoppingMinEpochs { get; init; } = 1;

    /// <summary>Tous les combien de pas on affiche une réponse d'exemple (première question de questions-test.txt).</summary>
    public int SampleInterval { get; init; } = 1000;

    /// <summary>
    /// Tous les combien de pas on mesure et affiche la loss (et on sauvegarde le modèle s'il est le meilleur).
    /// 500 en pré-entraînement : la loss y bouge lentement sur ~27 000 pas, et chaque mesure coûte du calcul.
    /// </summary>
    public int EvalInterval { get; init; } = 500;

    /// <summary>
    /// Nombre d'extraits de la vérification : toujours les mêmes, tirés une fois avec une graine fixe
    /// (voir TextDataset.FixedValidation). Deux mesures ne diffèrent que parce que le modèle a changé.
    /// </summary>
    public int ValidationExtracts { get; init; } = 200;

    /// <summary>Nombre de batchs tirés au hasard pour estimer la loss d'entraînement affichée.</summary>
    public int EvalBatches { get; init; } = 20;

    /// <summary>Graine aléatoire, pour avoir des résultats reproductibles.</summary>
    public int Seed { get; init; } = 42;

    /// <summary>Extraits de texte général dans chaque petit batch (3 sur 16 au fine-tuning, 0 sinon).</summary>
    public int GeneralExtracts => (int)Math.Round(BatchSize * GeneralTextRatio);

    /// <summary>Tokens du texte principal (texte.txt, ou les dialogues) lus à chaque étape.</summary>
    public long MainTokensPerStep(int blockSize) =>
        (long)(BatchSize - GeneralExtracts) * (EffectiveBatchSize / BatchSize) * blockSize;

    /// <summary>
    /// Steps calculé depuis Epochs et la taille des données d'entraînement (trainTokens) :
    /// Epochs x trainTokens / tokens lus par étape, compté depuis FirstStep. Sans Epochs : inchangé.
    /// </summary>
    public TrainingConfig ForData(long trainTokens, int blockSize) => Epochs is { } epochs
        ? this with { Steps = FirstStep + Math.Max(1, (int)Math.Ceiling(epochs * trainTokens / MainTokensPerStep(blockSize))) }
        : this;
}
