using System.Text.RegularExpressions;

namespace MiniChat.Tokenization;

/// <summary>
/// Tokenizer BPE ("Byte Pair Encoding"), comme GPT, mais sur des caractères.
///
/// Idée : on part des caractères seuls, puis on fusionne en boucle la paire voisine la plus
/// fréquente du texte d'entraînement :
///     "l e s"  ->  "le s"  ->  "les"
/// Chaque fusion crée un nouveau token. On s'arrête quand le vocabulaire a la taille voulue.
/// Résultat : les mots fréquents font 1 token, les mots rares sont découpés en morceaux.
/// Le modèle voit donc beaucoup plus de texte dans ses BlockSize tokens qu'avec un caractère par token.
///
/// Le texte est d'abord coupé en "mots" (lettres avec leur espace devant, chiffres, ponctuation,
/// retour à la ligne) : une fusion ne traverse jamais deux mots, et "\n" reste toujours un token seul.
///
/// Tokens spéciaux : des marqueurs écrits en toutes lettres dans le texte, qui donnent chacun UN token,
/// jamais découpé ni fusionné avec ses voisins (ids après les fusions, à la fin du vocabulaire) :
///   &lt;|doc|&gt;        un nouveau document commence (entre les textes de texte.txt, entre les conversations)
///   &lt;|user|&gt;       début d'un message de l'utilisateur
///   &lt;|assistant|&gt;  début d'une réponse de l'IA
///   &lt;|end|&gt;        fin d'un message : la génération s'arrête là
/// Le format des conversations avec ces tokens est dans ChatFormat.
/// </summary>
public sealed partial class BpeTokenizer
{
    public const string DocToken = "<|doc|>";
    public const string UserToken = "<|user|>";
    public const string AssistantToken = "<|assistant|>";
    public const string EndToken = "<|end|>";

    /// <summary>Les tokens spéciaux d'un nouveau tokenizer, dans l'ordre de leurs ids.</summary>
    public static readonly string[] DefaultSpecialTokens = { DocToken, UserToken, AssistantToken, EndToken };

    private readonly string[] _tokens;                          // id -> texte du token
    private readonly Dictionary<char, int> _charToId;
    private readonly (int Left, int Right)[] _merges;           // fusion n° i -> crée le token Alphabet.Length + i
    private readonly Dictionary<(int, int), int> _mergeRank;    // paire -> n° de la fusion (petit = appris tôt)
    private readonly Dictionary<string, int[]> _cache = new();  // mots déjà découpés
    private readonly Dictionary<string, int> _specialIds;       // "<|end|>" -> son id
    private readonly Regex _specialPattern;                     // trouve les tokens spéciaux dans un texte

    /// <param name="alphabet">Les caractères de base (ids 0 .. alphabet.Length - 1).</param>
    /// <param name="merges">Les fusions, dans l'ordre où elles ont été apprises.</param>
    /// <param name="specialTokens">Les tokens spéciaux, ids juste après les fusions.</param>
    public BpeTokenizer(string alphabet, IEnumerable<(int Left, int Right)> merges, IEnumerable<string> specialTokens)
    {
        Alphabet = alphabet;
        _merges = merges.ToArray();
        SpecialTokens = specialTokens.ToArray();

        var tokens = alphabet.Select(c => c.ToString()).ToList();
        foreach (var (left, right) in _merges)
            tokens.Add(tokens[left] + tokens[right]);
        _specialIds = SpecialTokens.Select((s, i) => (s, tokens.Count + i)).ToDictionary(p => p.s, p => p.Item2);
        tokens.AddRange(SpecialTokens);
        _tokens = tokens.ToArray();

        _charToId = alphabet.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        _mergeRank = _merges.Select((m, i) => (m, i)).ToDictionary(p => p.m, p => p.i);
        _specialPattern = SpecialPattern(SpecialTokens);
    }

    public string Alphabet { get; }
    public IReadOnlyList<(int Left, int Right)> Merges => _merges;
    public IReadOnlyList<string> SpecialTokens { get; }
    public int VocabSize => _tokens.Length;

    /// <summary>L'id d'un token spécial, ex. SpecialId(EndToken).</summary>
    public int SpecialId(string token) => _specialIds[token];

    /// <summary>Les ids de tous les tokens spéciaux.</summary>
    public IReadOnlyCollection<int> SpecialIds => _specialIds.Values;

    /// <summary>
    /// Apprend les fusions sur un texte jusqu'à avoir vocabSize tokens, tokens spéciaux compris.
    /// Les tokens spéciaux du texte sont ignorés : ils ne prêtent pas leurs caractères aux fusions.
    /// </summary>
    public static BpeTokenizer Train(string text, int vocabSize)
    {
        var specialTokens = DefaultSpecialTokens;
        text = SpecialPattern(specialTokens).Replace(text, "\n");
        vocabSize -= specialTokens.Length;
        string alphabet = new(text.Distinct().OrderBy(c => c).ToArray());
        var charToId = alphabet.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);

        // Chaque mot différent n'est traité qu'une fois, avec son nombre d'apparitions.
        var wordCounts = new Dictionary<string, int>();
        foreach (Match m in PreTokenizer().Matches(text))
            wordCounts[m.Value] = wordCounts.GetValueOrDefault(m.Value) + 1;

        var words = wordCounts
            .Select(kv => (Symbols: kv.Key.Select(c => charToId[c]).ToList(), Count: kv.Value))
            .Where(w => w.Symbols.Count > 1)   // un mot d'un seul symbole n'a rien à fusionner
            .ToList();

        var merges = new List<(int, int)>();

        // ponytail: on recompte toutes les paires à chaque fusion. ~1 min 30 pour 20 Mo et 4000 tokens :
        // c'est pour ça que Program.cs n'en donne qu'un échantillon de texte.txt (TokenizerSampleMb).
        // Pour aller plus loin, mettre à jour les comptes seulement autour des fusions.
        while (alphabet.Length + merges.Count < vocabSize)
        {
            var pairCounts = new Dictionary<(int, int), long>();
            foreach (var (symbols, count) in words)
                for (int i = 0; i + 1 < symbols.Count; i++)
                {
                    var pair = (symbols[i], symbols[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + count;
                }

            if (pairCounts.Count == 0) break;   // plus rien à fusionner : texte trop petit pour ce vocabulaire

            // La paire la plus fréquente. En cas d'égalité, la plus petite : même résultat à chaque entraînement.
            var best = pairCounts.MaxBy(p => (p.Value, -p.Key.Item1, -p.Key.Item2)).Key;
            int newId = alphabet.Length + merges.Count;

            foreach (var (symbols, _) in words)
                MergePair(symbols, best, newId);
            merges.Add(best);
        }

        return new BpeTokenizer(alphabet, merges, specialTokens);
    }

    /// <summary>Texte -> ids. Les tokens spéciaux écrits dans le texte donnent leur id ; les caractères inconnus sont ignorés.</summary>
    public long[] Encode(string text) => EncodeWithPositions(text).Select(t => t.Id).ToArray();

    /// <summary>
    /// Comme Encode, mais donne aussi, pour chaque token, la position dans le texte du dernier
    /// caractère de son mot (ou de son token spécial). Sert à savoir si un token fait partie
    /// d'une réponse de l'IA (voir ChatFormat.ReplyWeights).
    /// </summary>
    public IEnumerable<(long Id, int Position)> EncodeWithPositions(string text)
    {
        // Le texte entre deux tokens spéciaux est découpé en mots ; un token spécial est pris d'un bloc.
        int start = 0;
        foreach (Match special in _specialPattern.Matches(text))
        {
            foreach (var token in EncodePlain(text, start, special.Index))
                yield return token;
            yield return (_specialIds[special.Value], special.Index + special.Length - 1);
            start = special.Index + special.Length;
        }
        foreach (var token in EncodePlain(text, start, text.Length))
            yield return token;
    }

    /// <summary>Les mots de text[start..end], sans token spécial dedans.</summary>
    private IEnumerable<(long Id, int Position)> EncodePlain(string text, int start, int end)
    {
        if (end <= start)
            yield break;

        foreach (Match m in PreTokenizer().Matches(text[start..end]))
        {
            int lastChar = start + m.Index + m.Length - 1;
            foreach (int id in EncodeWord(m.Value))
                yield return (id, lastChar);
        }
    }

    // "<|doc|>|<\|user\|>|..." : les plus longs d'abord, au cas où l'un commencerait comme un autre.
    private static Regex SpecialPattern(IEnumerable<string> specialTokens) =>
        new(string.Join('|', specialTokens.OrderByDescending(s => s.Length).Select(Regex.Escape)), RegexOptions.Compiled);

    public string Decode(long id) => _tokens[id];

    public string Decode(IEnumerable<long> ids) => string.Concat(ids.Select(Decode));

    /// <summary>Découpe un mot : on rejoue les fusions apprises, la plus ancienne d'abord.</summary>
    private int[] EncodeWord(string word)
    {
        if (_cache.TryGetValue(word, out var cached))
            return cached;

        var symbols = new List<int>();
        foreach (char c in word)
            if (_charToId.TryGetValue(c, out int id))
                symbols.Add(id);

        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            for (int i = 0; i + 1 < symbols.Count; i++)
                if (_mergeRank.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                    bestRank = rank;

            if (bestRank == int.MaxValue) break;   // plus aucune fusion possible
            MergePair(symbols, _merges[bestRank], Alphabet.Length + bestRank);
        }

        return _cache[word] = symbols.ToArray();
    }

    /// <summary>Remplace chaque paire (gauche, droite) par newId, de gauche à droite. "a a a" + (a,a) -> "aa a".</summary>
    private static void MergePair(List<int> symbols, (int Left, int Right) pair, int newId)
    {
        for (int i = 0; i + 1 < symbols.Count; i++)
        {
            if (symbols[i] == pair.Left && symbols[i + 1] == pair.Right)
            {
                symbols[i] = newId;
                symbols.RemoveAt(i + 1);
            }
        }
    }

    // Découpage en mots. Chaque caractère du texte tombe dans exactement un morceau :
    //   \n                       un retour à la ligne, toujours seul
    //   " ?lettres"              un mot, avec son espace devant  (" bonjour")
    //   " ?chiffres"             un nombre                        (" 2024")
    //   " ?ponctuation"          ponctuation et symboles          (" !", "'", "...")
    //   espaces                  les autres espaces
    [GeneratedRegex(@"\n| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|[^\S\n]+")]
    private static partial Regex PreTokenizer();
}
