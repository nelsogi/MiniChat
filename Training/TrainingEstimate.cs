using MiniChat.Config;

namespace MiniChat.Training;

/// <summary>
/// Durée d'un entraînement, estimée avant de le lancer (/train la montre avant de demander confirmation).
/// Les calculs d'une étape se comptent : ~6 opérations par paramètre et par token (forward + backward), plus
/// l'attention, qui grandit avec le contexte. La vitesse du GPU en opérations par seconde est celle mesurée
/// avec le petit modèle : l'estimation vaut pour cette carte, et c'est un ordre de grandeur (les mesures
/// de loss et les sauvegardes s'y ajoutent, une plus grosse matrice va un peu plus vite par opération).
/// </summary>
public static class TrainingEstimate
{
    // small : 0,42 s par étape de 16 384 tokens sur une RTX 3060 Ti en TF32 (39 000 tokens/s),
    // à ~173 millions d'opérations par token.
    private const double FlopsPerSecond = 6.7e12;

    // Octets de texte.txt par token : 3,13 mesuré avec 4000 tokens de vocabulaire. Avec 8000, les mots
    // fréquents tiennent plus souvent en un token : un peu moins de tokens pour le même texte.
    private const double BytesPerToken = 3.4;

    /// <summary>Tokens estimés d'un fichier de données, d'après sa taille.</summary>
    public static long TokensIn(string path) => (long)(new FileInfo(path).Length / BytesPerToken);

    /// <summary>Nombre de pas d'une phase sur trainTokens tokens (voir TrainingConfig.ForData).</summary>
    public static int Steps(TrainingConfig config, long trainTokens, int blockSize) =>
        config.ForData(trainTokens, blockSize).Steps - config.FirstStep;

    /// <summary>Durée de steps pas d'entraînement pour ce modèle.</summary>
    public static TimeSpan Duration(ModelConfig model, long parameters, TrainingConfig config, int steps)
    {
        double flopsPerToken = 6.0 * parameters + 6.0 * model.Layers * model.BlockSize * model.EmbedDim;
        double tokens = (double)steps * config.EffectiveBatchSize * model.BlockSize;
        return TimeSpan.FromSeconds(tokens * flopsPerToken / FlopsPerSecond);
    }
}
