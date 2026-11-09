namespace MiniChat.Config;

/// <summary>Réglages de l'entraînement.</summary>
public sealed record TrainingConfig
{
    /// <summary>Vitesse d'apprentissage par défaut pour continuer un modèle déjà entraîné (--continue).</summary>
    public const double ContinueLearningRate = 3e-4;

    /// <summary>Warmup d'une continuation : court, le modèle et AdamW sont déjà "chauds".</summary>
    public const int ContinueWarmupSteps = 200;

    /// <summary>
    /// Continuer un modèle déjà entraîné (--continue, --continue-finetune) : steps pas de plus,
    /// numérotés à partir de firstStep (le pas où il s'était arrêté). Nouveau warmup court, puis
    /// cosinus depuis une vitesse plus basse qu'au premier entraînement : il s'agit d'affiner,
    /// pas de bousculer ce qui est appris. Sans learningRate : ContinueLearningRate, ou la vitesse
    /// de la phase si elle est déjà plus basse (fine-tuning, 2e-4).
    /// Warmup réduit à la moitié des pas pour une petite continuation (sinon il ne finirait jamais).
    /// </summary>
    public TrainingConfig Continuation(int firstStep, int steps, double? learningRate = null)
    {
        double lr = learningRate ?? Math.Min(ContinueLearningRate, LearningRate);
        return this with
        {
            FirstStep = firstStep,
            Steps = firstStep + steps,
            LearningRate = lr,
            MinLearningRate = lr / 10,
            WarmupSteps = Math.Min(ContinueWarmupSteps, steps / 2),
        };
    }

    /// <summary>Phase 1 : apprendre la langue sur un gros texte général (data/texte.txt).</summary>
    public static TrainingConfig Pretraining => new();

    /// <summary>
    /// Phase 2 : spécialiser le modèle sur les dialogues (data/dialogues.txt).
    /// Vitesse plus basse et moins d'étapes : on ajuste ce qui est déjà appris, sans tout effacer.
    ///
    /// dialogues.txt (OpenAssistant + French Alpaca + French Instruct, ~135 000 conversations) fait
    /// ~10 millions de tokens ; chaque étape en lit ~16 000, dont 80 % de dialogues : un passage complet
    /// prend ~800 étapes. 2 000 étapes = ~2,5 passages. Avec l'ancien fichier de 30 000 tokens, le modèle
    /// l'apprenait par cœur en 50 étapes : l'arrêt anticipé garde ce garde-fou si les dialogues sont petits.
    /// </summary>
    public static TrainingConfig FineTuning => new()
    {
        Steps = 2000,
        LearningRate = 2e-4,
        MinLearningRate = 2e-5,
        WarmupSteps = 50,
        GeneralTextRatio = 0.2,
        EvalInterval = 50,
        EarlyStoppingPatience = 300,
        SampleInterval = 250,
    };

    /// <summary>
    /// Part de chaque batch tirée de texte.txt pendant le fine-tuning (0.2 = 20 %).
    /// Sans ça, le modèle oublie peu à peu le français général appris en phase 1.
    /// </summary>
    public double GeneralTextRatio { get; init; } = 0;

    /// <summary>
    /// Nombre d'étapes d'apprentissage. Pré-entraînement : 20 000 x 64 extraits x 256 tokens = 330 millions
    /// de tokens, ~70 % de texte.txt sans jamais le relire. Mesuré sur RTX 3060 Ti avec le modèle de
    /// 27 M de paramètres : 0,46 s par étape, soit ~2 h 35 (interruptible, reprise avec --resume).
    /// C'est le numéro du dernier pas : la phase va de FirstStep à Steps.
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
    /// </summary>
    public double WeightDecay { get; init; } = 0.1;

    /// <summary>
    /// Gradient clipping : si la "force" totale d'une correction (norme du gradient) dépasse cette valeur,
    /// on la réduit. Évite qu'un batch bizarre fasse faire un pas énorme et casse le modèle.
    /// </summary>
    public double GradClip { get; init; } = 1.0;

    /// <summary>Vitesse d'apprentissage atteinte à la fin (la courbe cosinus descend jusqu'ici).</summary>
    public double MinLearningRate { get; init; } = 1e-4;

    /// <summary>Étapes de "chauffe" : la vitesse monte de 0 à LearningRate pour éviter un départ brutal.</summary>
    public int WarmupSteps { get; init; } = 100;

    /// <summary>Tous les combien de pas on sauvegarde l'avancement dans checkpoint/ (pour --resume).</summary>
    public int CheckpointInterval { get; init; } = 250;

    /// <summary>
    /// Arrêt anticipé : on s'arrête si la loss de vérification ne s'est pas améliorée depuis ce nombre de pas.
    /// Le modèle n'apprend alors plus rien d'utile (ou commence à apprendre par cœur). 0 = jamais.
    /// </summary>
    public int EarlyStoppingPatience { get; init; } = 2000;

    /// <summary>Tous les combien de pas on affiche une réponse d'exemple (première question de questions-test.txt).</summary>
    public int SampleInterval { get; init; } = 1000;

    /// <summary>
    /// Tous les combien de pas on mesure et affiche la loss (et on sauvegarde le modèle s'il est le meilleur).
    /// 100 en pré-entraînement : la loss y bouge lentement, et chaque mesure coûte du temps de calcul.
    /// </summary>
    public int EvalInterval { get; init; } = 100;

    /// <summary>Nombre de batchs moyennés pour mesurer la loss proprement.</summary>
    public int EvalBatches { get; init; } = 20;

    /// <summary>Graine aléatoire, pour avoir des résultats reproductibles.</summary>
    public int Seed { get; init; } = 42;
}
