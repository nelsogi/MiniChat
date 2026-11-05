using MiniChat.Chat;
using MiniChat.Tokenization;
using static TorchSharp.torch;

namespace MiniChat.Data;

/// <summary>
/// Le texte d'entraînement, découpé en 90 % pour apprendre et 10 % pour vérifier.
/// La partie "vérification" n'est jamais apprise : si la loss y descend aussi,
/// le modèle généralise vraiment au lieu de juste apprendre par cœur.
///
/// Chaque token a aussi un poids (le "masque") : 1 = le modèle est noté sur ce token,
/// 0 = il le lit mais n'est pas noté dessus. Voir onlyReplies.
/// </summary>
public sealed class TextDataset
{
    private readonly Tensor _train;
    private readonly Tensor _validation;
    private readonly Tensor _trainMask;
    private readonly Tensor _validationMask;
    private readonly Random _random;

    /// <param name="onlyReplies">
    /// true pour les dialogues : le modèle n'est noté que sur les réponses "IA: ..." (et le retour
    /// à la ligne qui les termine, pour qu'il apprenne à s'arrêter). Les lignes "Toi: ..." restent
    /// lues comme contexte, mais il n'a pas à apprendre à les écrire : c'est l'humain qui les tape.
    /// false : le modèle est noté sur tout le texte.
    /// </param>
    public TextDataset(string text, BpeTokenizer tokenizer, int seed, bool onlyReplies = false, double trainRatio = 0.9)
    {
        var weights = onlyReplies ? ReplyWeights(text) : null;

        // Un token prend le poids du dernier caractère de son mot. Un mot ne traverse jamais une ligne,
        // donc " Coucou" (après "IA:") compte, alors que "IA" et ":" ne comptent pas.
        var tokens = tokenizer.EncodeWithPositions(text).ToArray();
        var ids = tokens.Select(t => t.Id).ToArray();
        var mask = tokens.Select(t => weights?[t.Position] ?? 1f).ToArray();
        int split = (int)(ids.Length * trainRatio);

        _train = tensor(ids[..split]);
        _validation = tensor(ids[split..]);
        _trainMask = tensor(mask[..split]);
        _validationMask = tensor(mask[split..]);
        _random = new Random(seed);
    }

    public long TrainLength => _train.shape[0];
    public long ValidationLength => _validation.shape[0];

    /// <summary>
    /// Tire des extraits au hasard.
    /// x = l'extrait, y = le même extrait décalé d'un token (= "la bonne réponse").
    /// mask = le poids de chaque token de y dans la loss.
    /// Exemple : x = [" Bon", "jour", " à"], y = ["jour", " à", " toi"].
    /// </summary>
    public (Tensor X, Tensor Y, Tensor Mask) GetBatch(bool validation, int batchSize, int blockSize, Device device)
    {
        var source = validation ? _validation : _train;
        var sourceMask = validation ? _validationMask : _trainMask;
        long maxStart = source.shape[0] - blockSize - 1;
        if (maxStart <= 0)
            throw new InvalidOperationException("Pas assez de texte pour ce BlockSize : ajoute des données.");

        var starts = Enumerable.Range(0, batchSize)
            .Select(_ => _random.NextInt64(maxStart))
            .ToArray();

        var x = stack(starts.Select(s => source.narrow(0, s, blockSize)).ToArray());
        var y = stack(starts.Select(s => source.narrow(0, s + 1, blockSize)).ToArray());
        var mask = stack(starts.Select(s => sourceMask.narrow(0, s + 1, blockSize)).ToArray());
        return (x.to(device), y.to(device), mask.to(device));
    }

    /// <summary>
    /// 1 pour chaque caractère d'une réponse, 0 ailleurs.
    ///   "Toi: Salut\nIA: Coucou\n"
    ///    0000000000 0000111111 1      <- le "IA: " est donné au modèle, la réponse et le \n sont notés
    /// </summary>
    private static float[] ReplyWeights(string text)
    {
        var weights = new float[text.Length];
        int lineStart = 0;

        while (lineStart < text.Length)
        {
            int lineEnd = text.IndexOf('\n', lineStart);
            if (lineEnd < 0) lineEnd = text.Length - 1;   // dernière ligne sans retour à la ligne

            if (text.AsSpan(lineStart).StartsWith(ChatSession.BotPrefix))
                for (int i = lineStart + ChatSession.BotPrefix.Length; i <= lineEnd; i++)
                    weights[i] = 1;

            lineStart = lineEnd + 1;
        }

        return weights;
    }
}
