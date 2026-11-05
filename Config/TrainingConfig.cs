namespace MiniChat.Config;

/// <summary>Réglages de l'entraînement.</summary>
public sealed class TrainingConfig
{
    /// <summary>Phase 1 : apprendre la langue sur un gros texte général (data/texte.txt).</summary>
    public static TrainingConfig Pretraining => new();

    /// <summary>
    /// Phase 2 : spécialiser le modèle sur les dialogues (data/dialogues.txt).
    /// Vitesse plus basse et moins d'étapes : on ajuste ce qui est déjà appris, sans tout effacer.
    /// </summary>
    public static TrainingConfig FineTuning => new()
    {
        Steps = 800,
        LearningRate = 3e-4,
        MinLearningRate = 3e-5,
        WarmupSteps = 50,
        GeneralTextRatio = 0.2,
    };

    /// <summary>
    /// Part de chaque batch tirée de texte.txt pendant le fine-tuning (0.2 = 20 %).
    /// Sans ça, le modèle oublie peu à peu le français général appris en phase 1.
    /// </summary>
    public double GeneralTextRatio { get; init; } = 0;

    /// <summary>Nombre d'étapes d'apprentissage.</summary>
    public int Steps { get; init; } = 10000;

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

    /// <summary>Tous les combien de pas on affiche une réponse d'exemple (première question de questions-test.txt).</summary>
    public int SampleInterval { get; init; } = 1000;

    /// <summary>Tous les combien de pas on affiche la loss.</summary>
    public int EvalInterval { get; init; } = 50;

    /// <summary>Nombre de batchs moyennés pour mesurer la loss proprement.</summary>
    public int EvalBatches { get; init; } = 20;

    /// <summary>Graine aléatoire, pour avoir des résultats reproductibles.</summary>
    public int Seed { get; init; } = 42;
}
