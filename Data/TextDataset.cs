using MiniChat.Chat;
using MiniChat.Tokenization;
using static TorchSharp.torch;

namespace MiniChat.Data;

/// <summary>
/// Le texte d'entraînement, découpé en ~90 % pour apprendre et ~10 % pour vérifier.
/// La partie "vérification" n'est jamais apprise : si la loss y descend aussi,
/// le modèle généralise vraiment au lieu de juste apprendre par cœur.
///
/// La vérification n'est pas la fin du texte : texte.txt colle plusieurs sources bout à bout (Wikipedia,
/// livres...), et ses 10 derniers % n'en verraient qu'une. Le texte est coupé en morceaux de 10 000 tokens,
/// et dans chaque groupe de 10 morceaux qui se suivent, un seul, tiré au hasard, va en vérification :
///   [E E V E E E E E E E][E E E E E E V E E E][E V E E ...      E = entraînement, V = vérification
/// La vérification a donc un peu de chaque source. Le tirage a sa propre graine fixe (SplitSeed) :
/// même texte = mêmes morceaux de vérification, à chaque lancement, reprise ou continuation.
/// Un extrait tiré pour un batch reste toujours d'un seul côté : jamais à cheval sur les deux.
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
    /// <summary>Taille d'un morceau, en tokens.</summary>
    public const int ChunkTokens = 10_000;

    // Petit texte (dialogues.txt : ~35 000 tokens, soit 4 morceaux) : un morceau de vérification sur 4 ferait
    // 25 % du texte. Les morceaux rapetissent donc pour qu'il y en ait au moins MinChunks, sans descendre sous
    // MinChunkTokens (un extrait fait BlockSize + 1 tokens, il faut de la place pour en tirer plusieurs).
    private const int MinChunks = 100;
    private const int MinChunkTokens = 1_000;

    // Graine du choix des morceaux de vérification. Fixe et séparée de celle des batchs : la graine des batchs
    // change à la reprise et entre les phases, le découpage, lui, ne doit jamais changer.
    private const int SplitSeed = 1234;

    /// <summary>Une suite de morceaux voisins du même côté (entraînement ou vérification).</summary>
    private readonly record struct Segment(int Start, int Length);

    private readonly ushort[] _ids;
    private readonly float[]? _mask;   // null = le modèle est noté sur tous les tokens
    private readonly Segment[] _train;
    private readonly Segment[] _validation;
    private readonly Random _random;

    // Pour tirer un début d'extrait au hasard parmi tous les débuts possibles (voir StartsIndex).
    private (int BlockSize, long[] Train, long[] Validation)? _starts;

    /// <param name="text">Avec onlyReplies : des dialogues au format du modèle (ChatFormat.FromDialogueFile).</param>
    /// <param name="onlyReplies">
    /// true pour les dialogues : le modèle n'est noté que sur les réponses de l'IA et le &lt;|end|&gt; qui
    /// les termine, pour qu'il apprenne à s'arrêter (voir ChatFormat.ReplyWeights). Les questions restent
    /// lues comme contexte, mais il n'a pas à apprendre à les écrire : c'est l'humain qui les tape.
    /// false : le modèle est noté sur tout le texte.
    /// </param>
    /// <param name="validationEvery">Un morceau sur combien va en vérification (10 = ~10 %).</param>
    public TextDataset(string text, BpeTokenizer tokenizer, int seed, bool onlyReplies = false, int validationEvery = 10)
    {
        CheckVocabulary(tokenizer);
        var weights = onlyReplies ? ChatFormat.ReplyWeights(text) : null;

        // Un token prend le poids du dernier caractère de son mot (ou de son token spécial). Un mot ne
        // traverse jamais un token spécial, donc "Coucou" (après <|assistant|>) compte, <|assistant|> non.
        var tokens = tokenizer.EncodeWithPositions(text).ToArray();
        _ids = tokens.Select(t => (ushort)t.Id).ToArray();
        _mask = weights is null ? null : tokens.Select(t => weights[t.Position]).ToArray();
        (_train, _validation) = Split(_ids.Length, validationEvery);
        _random = new Random(seed);
        Characters = text.Length;
    }

    private TextDataset(ushort[] ids, long characters, int seed, int validationEvery)
    {
        _ids = ids;
        (_train, _validation) = Split(ids.Length, validationEvery);
        _random = new Random(seed);
        Characters = characters;
    }

    /// <summary>
    /// Lit un gros fichier ligne par ligne, sans jamais le charger en entier : au-delà d'environ
    /// 1 milliard de caractères, il ne tient même pas dans une chaîne .NET. Le modèle est noté sur tout le texte.
    /// Même résultat que le constructeur : un mot ne traverse jamais une ligne, on peut donc encoder ligne par ligne.
    /// </summary>
    public static TextDataset FromFile(string path, BpeTokenizer tokenizer, int seed, int validationEvery = 10)
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
        return new TextDataset(ids, characters, seed, validationEvery);
    }

    public long TrainLength => _train.Sum(s => (long)s.Length);
    public long ValidationLength => _validation.Sum(s => (long)s.Length);

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
        var segments = validation ? _validation : _train;
        var ends = StartsIndex(blockSize, validation);
        long possibleStarts = ends.Length == 0 ? 0 : ends[^1];
        if (possibleStarts <= 0)
            throw new InvalidOperationException("Pas assez de texte pour ce BlockSize : ajoute des données.");

        var x = new long[batchSize * blockSize];
        var y = new long[batchSize * blockSize];
        var mask = new float[batchSize * blockSize];

        for (int b = 0; b < batchSize; b++)
        {
            int start = PickStart(segments, ends, _random.NextInt64(possibleStarts));
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

    /// <summary>
    /// Coupe length tokens en morceaux, en met un sur validationEvery en vérification, et regroupe les
    /// morceaux voisins du même côté (un extrait peut passer d'un morceau d'entraînement au suivant).
    /// </summary>
    private static (Segment[] Train, Segment[] Validation) Split(int length, int validationEvery)
    {
        // Tout petit texte : des morceaux sous MinChunkTokens, pour en avoir au moins validationEvery.
        int chunk = Math.Max(1, Math.Max(Math.Min(length / MinChunks, ChunkTokens),
                                         Math.Min(MinChunkTokens, length / validationEvery)));
        int chunks = (length + chunk - 1) / chunk;

        var random = new Random(SplitSeed);
        var isValidation = new bool[chunks];
        for (int group = 0; group < chunks; group += validationEvery)
        {
            // Le dernier groupe peut être incomplet (fin du texte) : il n'a un morceau de vérification qu'avec
            // une chance size / validationEvery, pour rester à un sur validationEvery en moyenne. Sauf s'il est
            // le seul groupe : il faut toujours au moins un morceau de vérification.
            int size = Math.Min(validationEvery, chunks - group);
            if (size == validationEvery || group == 0 || random.Next(validationEvery) < size)
                isValidation[group + random.Next(size)] = true;
        }

        var train = new List<Segment>();
        var validation = new List<Segment>();
        int first = 0;
        for (int i = 1; i <= chunks; i++)
        {
            if (i < chunks && isValidation[i] == isValidation[first])
                continue;

            int start = first * chunk;
            var segment = new Segment(start, Math.Min(i * chunk, length) - start);
            (isValidation[first] ? validation : train).Add(segment);
            first = i;
        }
        return (train.ToArray(), validation.ToArray());
    }

    /// <summary>
    /// ends[i] = nombre de débuts d'extrait possibles dans les segments 0 à i (un extrait de BlockSize + 1 tokens
    /// doit tenir dans son segment). Tirer un nombre entre 0 et ends[^1] choisit donc un début au hasard,
    /// chacun avec la même chance, comme avant le découpage en morceaux. Calculé une fois par BlockSize.
    /// </summary>
    private long[] StartsIndex(int blockSize, bool validation)
    {
        if (_starts is not { } starts || starts.BlockSize != blockSize)
        {
            starts = (blockSize, Cumulate(_train, blockSize), Cumulate(_validation, blockSize));
            _starts = starts;
        }
        return validation ? starts.Validation : starts.Train;
    }

    private static long[] Cumulate(Segment[] segments, int blockSize)
    {
        var ends = new long[segments.Length];
        long total = 0;
        for (int i = 0; i < segments.Length; i++)
        {
            total += Math.Max(0, segments[i].Length - blockSize);
            ends[i] = total;
        }
        return ends;
    }

    /// <summary>Le début d'extrait n° index (voir StartsIndex) : son segment, puis la position dans le segment.</summary>
    private static int PickStart(Segment[] segments, long[] ends, long index)
    {
        // Premier segment dont ends dépasse index. BinarySearch renvoie ~position quand la valeur manque ;
        // quand elle est là (index == ends[i]), le début est dans un segment suivant.
        int i = Array.BinarySearch(ends, index);
        i = i >= 0 ? i + 1 : ~i;
        while (ends[i] == (i == 0 ? 0 : ends[i - 1]))   // segments trop courts pour un extrait : sautés
            i++;
        long before = i == 0 ? 0 : ends[i - 1];
        return segments[i].Start + (int)(index - before);
    }

    private static void CheckVocabulary(BpeTokenizer tokenizer)
    {
        if (tokenizer.VocabSize > ushort.MaxValue + 1)
            throw new InvalidOperationException($"Vocabulaire de {tokenizer.VocabSize} tokens : TextDataset en garde au plus 65 536 (ushort).");
    }
}
