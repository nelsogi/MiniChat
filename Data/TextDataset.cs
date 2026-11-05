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
///
/// Les tokens sont gardés en mémoire normale sur 2 octets chacun (ushort, vocabulaire &lt;= 65 536) :
/// un texte.txt de 1,5 Go donne ~460 millions de tokens, soit ~0,9 Go. Seuls les petits batchs
/// tirés au hasard deviennent des tenseurs (sur le GPU).
/// </summary>
public sealed class TextDataset
{
    private readonly ushort[] _ids;
    private readonly float[]? _mask;   // null = le modèle est noté sur tous les tokens
    private readonly int _split;       // _ids[.._split] = entraînement, _ids[_split..] = vérification
    private readonly Random _random;

    /// <param name="onlyReplies">
    /// true pour les dialogues : le modèle n'est noté que sur les réponses "IA: ..." (et le retour
    /// à la ligne qui les termine, pour qu'il apprenne à s'arrêter). Les lignes "Toi: ..." restent
    /// lues comme contexte, mais il n'a pas à apprendre à les écrire : c'est l'humain qui les tape.
    /// false : le modèle est noté sur tout le texte.
    /// </param>
    public TextDataset(string text, BpeTokenizer tokenizer, int seed, bool onlyReplies = false, double trainRatio = 0.9)
    {
        CheckVocabulary(tokenizer);
        var weights = onlyReplies ? ReplyWeights(text) : null;

        // Un token prend le poids du dernier caractère de son mot. Un mot ne traverse jamais une ligne,
        // donc " Coucou" (après "IA:") compte, alors que "IA" et ":" ne comptent pas.
        var tokens = tokenizer.EncodeWithPositions(text).ToArray();
        _ids = tokens.Select(t => (ushort)t.Id).ToArray();
        _mask = weights is null ? null : tokens.Select(t => weights[t.Position]).ToArray();
        _split = (int)(_ids.Length * trainRatio);
        _random = new Random(seed);
        Characters = text.Length;
    }

    private TextDataset(ushort[] ids, long characters, int seed, double trainRatio)
    {
        _ids = ids;
        _split = (int)(ids.Length * trainRatio);
        _random = new Random(seed);
        Characters = characters;
    }

    /// <summary>
    /// Lit un gros fichier ligne par ligne, sans jamais le charger en entier : au-delà d'environ
    /// 1 milliard de caractères, il ne tient même pas dans une chaîne .NET. Le modèle est noté sur tout le texte.
    /// Même résultat que le constructeur : un mot ne traverse jamais une ligne, on peut donc encoder ligne par ligne.
    /// </summary>
    public static TextDataset FromFile(string path, BpeTokenizer tokenizer, int seed, double trainRatio = 0.9)
    {
        CheckVocabulary(tokenizer);

        // Taille de départ : ~1 token pour 3 octets en français (mesuré). Agrandi si besoin.
        var ids = new ushort[Math.Clamp(new FileInfo(path).Length / 3, 1024, Array.MaxLength)];
        int count = 0;
        long characters = 0;

        // ReadLine enlève aussi les "\r" des fins de ligne Windows.
        using var reader = new StreamReader(path);
        while (reader.ReadLine() is { } line)
        {
            characters += line.Length + 1;
            foreach (long id in tokenizer.Encode(line + "\n"))
            {
                if (count == ids.Length)
                {
                    if (count == Array.MaxLength)
                        throw new InvalidDataException($"{Path.GetFileName(path)} : plus de {Array.MaxLength:N0} tokens, réduis le fichier (DataBuilder --max-mo).");
                    Array.Resize(ref ids, (int)Math.Min(2L * ids.Length, Array.MaxLength));
                }
                ids[count++] = (ushort)id;
            }
        }

        Array.Resize(ref ids, count);
        return new TextDataset(ids, characters, seed, trainRatio);
    }

    public long TrainLength => _split;
    public long ValidationLength => _ids.Length - _split;

    /// <summary>Nombre de caractères du texte de départ.</summary>
    public long Characters { get; }

    /// <summary>
    /// Tire des extraits au hasard.
    /// x = l'extrait, y = le même extrait décalé d'un token (= "la bonne réponse").
    /// mask = le poids de chaque token de y dans la loss.
    /// Exemple : x = [" Bon", "jour", " à"], y = ["jour", " à", " toi"].
    /// </summary>
    public (Tensor X, Tensor Y, Tensor Mask) GetBatch(bool validation, int batchSize, int blockSize, Device device)
    {
        int offset = validation ? _split : 0;
        long maxStart = (validation ? ValidationLength : TrainLength) - blockSize - 1;
        if (maxStart <= 0)
            throw new InvalidOperationException("Pas assez de texte pour ce BlockSize : ajoute des données.");

        var x = new long[batchSize * blockSize];
        var y = new long[batchSize * blockSize];
        var mask = new float[batchSize * blockSize];

        for (int b = 0; b < batchSize; b++)
        {
            int start = offset + (int)_random.NextInt64(maxStart);
            for (int t = 0; t < blockSize; t++)
            {
                int i = b * blockSize + t;
                x[i] = _ids[start + t];
                y[i] = _ids[start + t + 1];
                mask[i] = _mask?[start + t + 1] ?? 1f;
            }
        }

        long[] shape = [batchSize, blockSize];
        return (tensor(x, shape).to(device), tensor(y, shape).to(device), tensor(mask, shape).to(device));
    }

    private static void CheckVocabulary(BpeTokenizer tokenizer)
    {
        if (tokenizer.VocabSize > ushort.MaxValue + 1)
            throw new InvalidOperationException($"Vocabulaire de {tokenizer.VocabSize} tokens : TextDataset en garde au plus 65 536 (ushort).");
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
