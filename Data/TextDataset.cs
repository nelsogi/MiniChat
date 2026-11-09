using MiniChat.Chat;
using MiniChat.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.Data;

/// <summary>
/// Un texte découpé en tokens, avant d'être envoyé sur le GPU : ce que TokenCache garde sur le disque.
/// Ids sur 2 octets (short, vocabulaire &lt;= 32 768). Mask : le poids (0 ou 1) de chaque token dans la loss,
/// null = le modèle est noté sur tous les tokens. Characters = nombre de caractères du texte de départ.
/// </summary>
public sealed record TokenizedText(short[] Ids, byte[]? Mask, long Characters);

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
/// 0 = il le lit mais n'est pas noté dessus. Voir Tokenize (onlyReplies).
///
/// Les tokens sont gardés directement sur le device (GPU), en int16 : un texte.txt de 1,5 Go donne
/// ~460 millions de tokens, soit ~0,9 Go de mémoire GPU. Un batch y est découpé par des index :
/// seuls les débuts d'extrait (un nombre par extrait) passent du CPU au GPU. À libérer (Dispose) après la phase.
/// </summary>
public sealed class TextDataset : IDisposable
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

    private readonly Tensor _ids;      // int16, forme (tokens), sur le device
    private readonly Tensor? _mask;    // uint8, forme (tokens), sur le device ; null = noté sur tous les tokens
    private readonly Device _device;
    private readonly Segment[] _train;
    private readonly Segment[] _validation;
    private readonly Random _random;

    // Pour tirer un début d'extrait au hasard parmi tous les débuts possibles (voir StartsIndex).
    private (int BlockSize, long[] Train, long[] Validation)? _starts;

    /// <param name="text">Les tokens (Tokenize, TokenizeFile ou TokenCache), copiés sur le device.</param>
    /// <param name="validationEvery">Un morceau sur combien va en vérification (10 = ~10 %).</param>
    public TextDataset(TokenizedText text, int seed, Device device, int validationEvery = 10)
    {
        _device = device;
        _ids = ToDevice(tensor(text.Ids), device);
        _mask = text.Mask is null ? null : ToDevice(tensor(text.Mask), device);
        (_train, _validation) = Split(text.Ids.Length, validationEvery);
        _random = new Random(seed);
        Characters = text.Characters;
    }

    /// <summary>
    /// Découpe un texte en tokens.
    /// onlyReplies : true pour les dialogues (au format du modèle, ChatFormat.FromDialogueFile) : le modèle
    /// n'est noté que sur les réponses de l'IA et le &lt;|end|&gt; qui les termine, pour qu'il apprenne à
    /// s'arrêter (voir ChatFormat.ReplyWeights). Les questions restent lues comme contexte, mais il n'a pas
    /// à apprendre à les écrire : c'est l'humain qui les tape. false : le modèle est noté sur tout le texte.
    /// </summary>
    public static TokenizedText Tokenize(string text, BpeTokenizer tokenizer, bool onlyReplies)
    {
        CheckVocabulary(tokenizer);
        var weights = onlyReplies ? ChatFormat.ReplyWeights(text) : null;

        // Un token prend le poids du dernier caractère de son mot (ou de son token spécial). Un mot ne
        // traverse jamais un token spécial, donc "Coucou" (après <|assistant|>) compte, <|assistant|> non.
        var tokens = tokenizer.EncodeWithPositions(text).ToArray();
        var ids = tokens.Select(t => (short)t.Id).ToArray();
        var mask = weights is null ? null : tokens.Select(t => (byte)weights[t.Position]).ToArray();
        return new TokenizedText(ids, mask, text.Length);
    }

    /// <summary>
    /// Lit un gros fichier ligne par ligne, sans jamais le charger en entier : au-delà d'environ
    /// 1 milliard de caractères, il ne tient même pas dans une chaîne .NET. Le modèle est noté sur tout le texte.
    /// Même résultat que Tokenize : un mot ne traverse jamais une ligne, on peut donc encoder ligne par ligne.
    /// </summary>
    public static TokenizedText TokenizeFile(string path, BpeTokenizer tokenizer)
    {
        CheckVocabulary(tokenizer);

        // Taille de départ : ~1 token pour 3 octets en français (mesuré). Agrandi si besoin.
        var ids = new short[Math.Clamp(new FileInfo(path).Length / 3, 1024, Array.MaxLength)];
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
                ids[count++] = (short)id;
            }
        }

        Array.Resize(ref ids, count);
        return new TokenizedText(ids, null, characters);
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
    ///
    /// Les débuts sont tirés sur le CPU (même tirage qu'avant, donc les mêmes extraits), puis le GPU découpe :
    /// index[b, t] = début[b] + t, pour t de 0 à BlockSize, et les tokens sont lus d'un coup à ces index.
    /// </summary>
    public (Tensor X, Tensor Y, Tensor Mask) GetBatch(bool validation, int batchSize, int blockSize, Device device)
    {
        var segments = validation ? _validation : _train;
        var ends = StartsIndex(blockSize, validation);
        long possibleStarts = ends.Length == 0 ? 0 : ends[^1];
        if (possibleStarts <= 0)
            throw new InvalidOperationException("Pas assez de texte pour ce BlockSize : ajoute des données.");

        var starts = new long[batchSize];
        for (int b = 0; b < batchSize; b++)
            starts[b] = PickStart(segments, ends, _random.NextInt64(possibleStarts));

        // (batch, BlockSize + 1) : chaque extrait et le token qui le suit.
        var index = tensor(starts, device: _device).unsqueeze(1)
                    + arange(blockSize + 1, ScalarType.Int64, _device).unsqueeze(0);
        var flat = index.view(-1);
        var tokens = _ids.index_select(0, flat).view(batchSize, blockSize + 1).to(ScalarType.Int64);

        var x = tokens.narrow(1, 0, blockSize).contiguous();
        var y = tokens.narrow(1, 1, blockSize).contiguous();
        var mask = _mask is null
            ? ones(batchSize, blockSize, ScalarType.Float32, _device)
            : _mask.index_select(0, index.narrow(1, 1, blockSize).reshape(-1)).view(batchSize, blockSize).to(ScalarType.Float32);

        return (x.to(device), y.to(device), mask.to(device));
    }

    public void Dispose()
    {
        _ids.Dispose();
        _mask?.Dispose();
    }

    // Gardé hors des DisposeScope : le tenseur vit aussi longtemps que le dataset. La copie CPU est libérée.
    private static Tensor ToDevice(Tensor cpu, Device device)
    {
        if (device.type == DeviceType.CPU)
            return cpu.DetachFromDisposeScope();

        var moved = cpu.to(device).DetachFromDisposeScope();
        cpu.Dispose();
        return moved;
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
        if (tokenizer.VocabSize > short.MaxValue + 1)
            throw new InvalidOperationException($"Vocabulaire de {tokenizer.VocabSize} tokens : TextDataset en garde au plus 32 768 (int16).");
    }
}
