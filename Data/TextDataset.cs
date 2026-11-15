using MiniChat.Chat;
using MiniChat.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.Data;

/// <summary>
/// Un texte découpé en tokens, avant TextDataset : ce que TokenCache garde sur le disque.
/// Ids sur 2 octets (short, vocabulaire &lt;= 32 768). Mask : le poids (0 ou 1) de chaque token dans la loss,
/// null = le modèle est noté sur tous les tokens. Characters = nombre de caractères du texte de départ.
/// </summary>
public sealed record TokenizedText(short[] Ids, byte[]? Mask, long Characters);

/// <summary>
/// Des extraits pour le modèle. X = les tokens lus, Y = le token suivant de chacun (la bonne réponse),
/// Mask = le poids de chaque token de Y dans la loss, tous de forme (extraits, BlockSize).
/// Documents : le numéro de conversation de chaque token quand un extrait en contient plusieurs (dialogues,
/// voir MiniGpt.Forward), null pour un texte suivi.
/// </summary>
public sealed record Batch(Tensor X, Tensor Y, Tensor Mask, Tensor? Documents = null);

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
/// Dialogues (documentToken = &lt;|doc|&gt;) : un extrait ne commence plus n'importe où, mais au début d'une
/// conversation tirée au hasard, suivie des conversations d'après tant qu'elles tiennent EN ENTIER ; la place qui
/// reste est du remplissage, jamais noté. Chaque conversation ne voit qu'elle-même et ses positions repartent
/// de 0 (Batch.Documents, voir DocumentMask) : le modèle apprend chaque conversation comme si elle était seule,
/// et de son début, comme en discutant. Une conversation plus longue que l'extrait est coupée à BlockSize tokens.
///   [doc conversation 12 ......][doc conversation 13 ...][doc conversation 14 .....][remplissage]
/// Une conversation appartient au côté (entraînement ou vérification) où elle commence, et s'arrête à la fin de
/// son segment : rien ne passe d'un côté à l'autre.
///
/// Les tokens sont gardés sur le CPU, en int16 : un texte.txt de 3 Go donne ~820 millions de tokens (1,6 Go), qui
/// prendraient au GPU la place de l'entraînement. Un batch y est découpé par des index, puis seuls ses tokens
/// (quelques milliers) passent sur le device : 0,3 ms par micro-batch de 4 x 512 (mesuré, 820 M tokens, RTX 3060 Ti). À libérer (Dispose) après la phase.
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
    internal readonly record struct Segment(int Start, int Length);

    /// <summary>
    /// Une conversation : Length tokens à partir de Start, du &lt;|doc|&gt; qui l'ouvre au suivant (ou à la fin de son
    /// segment). Elle occupe Length - 1 places dans un extrait : son dernier token n'est que la réponse attendue
    /// (Y) du précédent, il n'y a rien à prédire après.
    /// </summary>
    internal readonly record struct Conversation(int Start, int Length);

    private readonly Tensor _ids;      // int16, forme (tokens), sur le CPU (voir plus bas)
    private readonly Tensor? _mask;    // uint8, forme (tokens), sur le CPU ; null = noté sur tous les tokens
    private readonly Device _device;
    private readonly Segment[] _train;
    private readonly Segment[] _validation;
    private readonly Random _random;

    // Dialogues : les conversations de chaque côté, dans l'ordre du texte. null : texte suivi (extraits n'importe où).
    private readonly Conversation[]? _trainConversations;
    private readonly Conversation[]? _validationConversations;

    // Graine des extraits de vérification (voir FixedValidation), fixe comme SplitSeed.
    private const int ValidationSeed = 4321;

    // Pour tirer un début d'extrait au hasard parmi tous les débuts possibles (voir StartsIndex).
    private (int BlockSize, long[] Train, long[] Validation)? _starts;

    // Les extraits de vérification, tirés une fois (voir FixedValidation).
    private (int Count, int BlockSize, Batch Batch)? _fixedValidation;

    // Remplissage moyen des extraits d'entraînement (voir ExtractFill), calculé une fois par BlockSize.
    private (int BlockSize, double Fill)? _fill;

    /// <param name="text">Les tokens (Tokenize, TokenizeFile ou TokenCache), gardés sur le CPU.</param>
    /// <param name="device">Où vont les extraits de vérification (FixedValidation). GetBatch prend le sien.</param>
    /// <param name="validationEvery">Un morceau sur combien va en vérification (10 = ~10 %).</param>
    /// <param name="documentToken">
    /// L'id de &lt;|doc|&gt; pour des dialogues : chaque extrait est fait de conversations entières, qui ne se voient
    /// pas entre elles (voir plus haut). null : texte suivi, les extraits commencent n'importe où.
    /// </param>
    public TextDataset(TokenizedText text, int seed, Device device, int validationEvery = 10, int? documentToken = null)
    {
        _device = device;
        _ids = tensor(text.Ids).DetachFromDisposeScope();
        _mask = text.Mask is null ? null : tensor(text.Mask).DetachFromDisposeScope();
        (_train, _validation) = Split(text.Ids.Length, validationEvery);
        _random = new Random(seed);
        Characters = text.Characters;

        if (documentToken is { } token)
        {
            _trainConversations = Conversations(text.Ids, _train, (short)token);
            _validationConversations = Conversations(text.Ids, _validation, (short)token);
        }
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
    /// Les débuts sont tirés sur le CPU (même tirage qu'avant, donc les mêmes extraits), puis le CPU découpe :
    /// index[b, t] = début[b] + t, pour t de 0 à BlockSize, et les tokens sont lus d'un coup à ces index.
    /// Dialogues : une conversation tirée au hasard par extrait, puis les suivantes (voir Pack).
    /// </summary>
    public Batch GetBatch(bool validation, int batchSize, int blockSize, Device device)
    {
        if (ConversationsOf(validation) is { } conversations)
            return To(Pack(conversations, Enumerable.Range(0, batchSize).Select(_ => _random.Next(conversations.Length)).ToArray(),
                blockSize), device);

        var segments = validation ? _validation : _train;
        var ends = StartsIndex(blockSize, validation);
        long possibleStarts = ends.Length == 0 ? 0 : ends[^1];
        if (possibleStarts <= 0)
            throw new InvalidOperationException("Pas assez de texte pour ce BlockSize : ajoute des données.");

        var starts = new long[batchSize];
        for (int b = 0; b < batchSize; b++)
            starts[b] = PickStart(segments, ends, _random.NextInt64(possibleStarts));

        return To(Gather(starts, blockSize), device);
    }

    /// <summary>
    /// La vérification : count extraits de la partie vérification, tirés une seule fois avec une graine fixe
    /// (ValidationSeed, à part du tirage des batchs) puis gardés sur le device. Toujours les mêmes, à chaque
    /// mesure, reprise ou continuation sur le même texte : la loss de vérification ne bouge que si le modèle change.
    /// </summary>
    public Batch FixedValidation(int count, int blockSize)
    {
        if (_fixedValidation is { } cached && cached.Count == count && cached.BlockSize == blockSize)
            return cached.Batch;

        var random = new Random(ValidationSeed);
        Batch Draw()
        {
            if (_validationConversations is { } conversations)
                return conversations.Length > 0
                    ? Pack(conversations, Enumerable.Range(0, count).Select(_ => random.Next(conversations.Length)).ToArray(), blockSize)
                    : throw new InvalidOperationException("Aucune conversation dans la vérification : ajoute des données.");

            var ends = StartsIndex(blockSize, validation: true);
            long possibleStarts = ends.Length == 0 ? 0 : ends[^1];
            if (possibleStarts <= 0)
                throw new InvalidOperationException("Pas assez de texte pour ce BlockSize : ajoute des données.");

            var starts = new long[count];
            for (int i = 0; i < count; i++)
                starts[i] = PickStart(_validation, ends, random.NextInt64(possibleStarts));
            return Gather(starts, blockSize);
        }

        DisposeFixedValidation();
        using var scope = NewDisposeScope();
        var batch = To(Draw(), _device);
        _fixedValidation = (count, blockSize, new Batch(batch.X.DetachFromDisposeScope(), batch.Y.DetachFromDisposeScope(),
            batch.Mask.DetachFromDisposeScope(), batch.Documents?.DetachFromDisposeScope()));
        return batch;
    }

    /// <summary>
    /// Part moyenne d'un extrait d'entraînement qui est du vrai texte, le reste étant du remplissage :
    /// 1 pour un texte suivi. Dialogues : moyenne exacte sur toutes les conversations de départ possibles
    /// (voir Runs). Sert à compter les passages sur les données : un extrait à moitié rempli n'en lit que la moitié.
    /// </summary>
    public double ExtractFill(int blockSize)
    {
        if (_trainConversations is not { Length: > 0 } conversations)
            return 1;
        if (_fill is { } fill && fill.BlockSize == blockSize)
            return fill.Fill;

        double used = 0;
        for (int first = 0; first < conversations.Length; first++)
            used += Runs(conversations, first, blockSize).Sum(r => r.Length);
        _fill = (blockSize, used / conversations.Length / blockSize);
        return _fill.Value.Fill;
    }

    /// <summary>
    /// Les morceaux de texte d'un extrait de dialogues qui commence à la conversation first : elle (coupée à
    /// blockSize places si elle est trop longue), puis les suivantes tant qu'elles tiennent en entier dans la place
    /// qui reste. Chaque morceau : Start = son premier token (x), Length = places occupées (x lit Start à
    /// Start + Length - 1, y attend Start + 1 à Start + Length).
    /// </summary>
    internal static List<Conversation> Runs(Conversation[] conversations, int first, int blockSize)
    {
        var start = conversations[first];
        var runs = new List<Conversation> { start with { Length = Math.Min(start.Length - 1, blockSize) } };
        int used = runs[0].Length;
        for (int i = first + 1; i < conversations.Length && conversations[i].Length - 1 <= blockSize - used; i++)
        {
            runs.Add(conversations[i] with { Length = conversations[i].Length - 1 });
            used += conversations[i].Length - 1;
        }
        return runs;
    }

    /// <summary>
    /// Les conversations d'un côté (segments) : chacune commence à un token &lt;|doc|&gt; et va jusqu'au suivant,
    /// ou jusqu'à la fin de son segment. Le début d'un segment, avant son premier &lt;|doc|&gt;, est la fin d'une
    /// conversation de l'autre côté : il n'est pas utilisé. Une conversation d'un seul token (rien à prédire) non plus.
    /// </summary>
    internal static Conversation[] Conversations(short[] ids, Segment[] segments, short documentToken)
    {
        var conversations = new List<Conversation>();
        foreach (var segment in segments)
        {
            int end = segment.Start + segment.Length;
            int start = -1;
            for (int i = segment.Start; i <= end; i++)
            {
                if (i < end && ids[i] != documentToken)
                    continue;
                if (start >= 0 && i - start >= 2)
                    conversations.Add(new Conversation(start, i - start));
                start = i;
            }
        }
        return conversations.ToArray();
    }

    private Conversation[]? ConversationsOf(bool validation)
    {
        var conversations = validation ? _validationConversations : _trainConversations;
        if (conversations is { Length: 0 })
            throw new InvalidOperationException("Aucune conversation de ce côté : ajoute des données.");
        return conversations;
    }

    /// <summary>
    /// Les extraits de dialogues qui commencent aux conversations firsts (voir Runs), construits et lus sur le
    /// CPU : pour chaque place, l'index de son token (x) et de la réponse attendue (y),
    /// son numéro de conversation, et 1 (vrai texte) ou 0 (remplissage). Le remplissage forme une conversation
    /// à lui (il ne voit et ne gêne personne) et n'est jamais noté.
    /// </summary>
    private Batch Pack(Conversation[] conversations, int[] firsts, int blockSize)
    {
        int count = firsts.Length;
        var xIndex = new long[count * blockSize];
        var documents = new long[count * blockSize];
        var real = new float[count * blockSize];

        for (int b = 0; b < count; b++)
        {
            int place = b * blockSize;
            var runs = Runs(conversations, firsts[b], blockSize);
            for (int document = 0; document < runs.Count; document++)
                for (int t = 0; t < runs[document].Length; t++, place++)
                    (xIndex[place], documents[place], real[place]) = (runs[document].Start + t, document, 1);

            for (; place < (b + 1) * blockSize; place++)
                (xIndex[place], documents[place], real[place]) = (runs[0].Start, runs.Count, 0);
        }

        var xAt = tensor(xIndex, device: CPU);
        var yAt = xAt + tensor(real, device: CPU).to(ScalarType.Int64);   // le token suivant ; remplissage : lui-même
        var realMask = tensor(real, device: CPU).view(count, blockSize);
        var mask = _mask is null
            ? realMask
            : _mask.index_select(0, yAt).view(count, blockSize).to(ScalarType.Float32) * realMask;
        return new Batch(
            _ids.index_select(0, xAt).view(count, blockSize).to(ScalarType.Int64),
            _ids.index_select(0, yAt).view(count, blockSize).to(ScalarType.Int64),
            mask,
            tensor(documents, device: CPU).view(count, blockSize));
    }

    private static Batch To(Batch batch, Device device) =>
        new(batch.X.to(device), batch.Y.to(device), batch.Mask.to(device), batch.Documents?.to(device));

    /// <summary>
    /// Les extraits qui commencent à starts, découpés sur le CPU : index[b, t] = début[b] + t, pour t de 0 à
    /// BlockSize, et les tokens sont lus d'un coup à ces index.
    /// </summary>
    private Batch Gather(long[] starts, int blockSize)
    {
        int count = starts.Length;

        // (extraits, BlockSize + 1) : chaque extrait et le token qui le suit.
        var index = tensor(starts, device: CPU).unsqueeze(1)
                    + arange(blockSize + 1, ScalarType.Int64, CPU).unsqueeze(0);
        var tokens = _ids.index_select(0, index.view(-1)).view(count, blockSize + 1).to(ScalarType.Int64);

        var x = tokens.narrow(1, 0, blockSize).contiguous();
        var y = tokens.narrow(1, 1, blockSize).contiguous();
        var mask = _mask is null
            ? ones(count, blockSize, ScalarType.Float32, CPU)
            : _mask.index_select(0, index.narrow(1, 1, blockSize).reshape(-1)).view(count, blockSize).to(ScalarType.Float32);
        return new Batch(x, y, mask);
    }

    public void Dispose()
    {
        _ids.Dispose();
        _mask?.Dispose();
        DisposeFixedValidation();
    }

    private void DisposeFixedValidation()
    {
        if (_fixedValidation is not { } cached) return;
        cached.Batch.X.Dispose();
        cached.Batch.Y.Dispose();
        cached.Batch.Mask.Dispose();
        cached.Batch.Documents?.Dispose();
        _fixedValidation = null;
    }

    /// <summary>
    /// Coupe length tokens en morceaux, en met un sur validationEvery en vérification, et regroupe les
    /// morceaux voisins du même côté (un extrait peut passer d'un morceau d'entraînement au suivant).
    /// </summary>
    internal static (Segment[] Train, Segment[] Validation) Split(int length, int validationEvery)
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
