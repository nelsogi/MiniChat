namespace MiniChat.Training;

/// <summary>
/// Une mesure de loss pendant l'entraînement (tous les EvalInterval pas).
/// Remaining = temps restant estimé, null au premier pas (rien pour l'estimer encore).
/// IsBest = meilleure loss de vérification vue jusqu'ici : ces poids sont gardés.
/// </summary>
public sealed record TrainingReport(int Step, int LastStep, float TrainLoss, float ValLoss, double LearningRate,
    TimeSpan Elapsed, TimeSpan? Remaining, bool IsBest, (long UsedMb, long TotalMb)? Gpu);

/// <summary>
/// Ce que le Trainer montre pendant qu'il tourne. Il ne sait pas comment c'est affiché :
/// c'est l'interface (UI/TrainingProgressView) qui choisit, une barre de progression dans Nel Code.
/// </summary>
public interface ITrainingProgress
{
    /// <summary>Appelé avant chaque pas : le pas en cours, sa vitesse d'apprentissage, le temps restant estimé.</summary>
    void Advance(int step, double learningRate, TimeSpan? remaining);

    /// <summary>Une nouvelle mesure des deux loss.</summary>
    void Report(TrainingReport report);

    /// <summary>Un message (reprise, arrêt anticipé, réponse d'exemple...).</summary>
    void Log(string message);
}
