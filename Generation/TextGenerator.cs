using MiniChat.Model;
using MiniChat.Tokenization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Generation;

/// <summary>Bilan d'une génération : tokens écrits, tokens du prompt, durée (prompt compris).</summary>
public sealed record GenerationStats(int Tokens, int PromptTokens, TimeSpan Elapsed)
{
    public double TokensPerSecond => Elapsed.TotalSeconds > 0 ? Tokens / Elapsed.TotalSeconds : 0;
}

/// <summary>
/// Fait écrire le modèle : on lui donne un début, il prédit un token (un morceau de mot),
/// on l'ajoute, il prédit le suivant, etc.
///
/// Avec le cache KV (par défaut) : le prompt passe une fois dans le modèle, puis chaque token écrit
/// n'y fait passer que lui-même (voir KvCache). Sans : tout le contexte repasse à chaque token.
/// </summary>
public sealed class TextGenerator
{
    // Cache plein (le contexte atteint BlockSize) : on repart des 3/4 derniers tokens, recalculés d'un coup.
    // Le contexte va alors de 3/4 à tout BlockSize ; un seul recalcul tous les BlockSize/4 tokens.
    private const double KeptWhenFull = 0.75;

    private readonly MiniGpt _model;
    private readonly BpeTokenizer _tokenizer;
    private readonly Device _device;

    public TextGenerator(MiniGpt model, BpeTokenizer tokenizer, Device device)
    {
        _model = model;
        _tokenizer = tokenizer;
        _device = device;
    }

    public MiniGpt Model => _model;
    public Device Device => _device;

    /// <summary>Nombre de tokens d'un texte (pour savoir ce qui tient dans la mémoire du modèle).</summary>
    public int CountTokens(string text) => _tokenizer.Encode(text).Length;

    /// <summary>Le bilan de la dernière génération (vitesse en tokens/seconde), null avant la première.</summary>
    public GenerationStats? LastStats { get; private set; }

    /// <param name="prompt">Le début du texte.</param>
    /// <param name="maxNewTokens">Limite de sécurité, en tokens.</param>
    /// <param name="temperature">
    /// &lt; 1 : plus prudent et répétitif. &gt; 1 : plus créatif mais plus de fautes.
    /// </param>
    /// <param name="topK">Ne tire au sort que parmi les K tokens les plus probables (0 = pas de limite).</param>
    /// <param name="topP">
    /// Ne garde que les tokens les plus probables dont les chances cumulées font topP (0.9 = 90 %).
    /// 1 = pas de limite.
    /// </param>
    /// <param name="onChar">Appelé à chaque caractère écrit. Renvoie false pour arrêter.</param>
    /// <param name="seed">
    /// Graine du tirage : même graine + même modèle = même texte. null = hasard global.
    /// Avec une graine, le tirage a son propre générateur : il ne touche pas au hasard de l'entraînement.
    /// </param>
    /// <param name="useCache">false : sans cache KV (plus lent ; pour comparer, même texte à l'arrondi près).</param>
    public string Generate(string prompt, int maxNewTokens, double temperature, int topK, double topP, Func<char, bool> onChar,
        long? seed = null, bool useCache = true)
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        _model.eval();
        using var noGrad = no_grad();
        using var scope = NewDisposeScope();
        // Générateur sur CPU : TorchSharp le crée toujours sur CPU, et le tirage avec graine se fait donc sur CPU
        // (voir SampleTopKTopP). Même graine = même texte, que le modèle tourne sur GPU ou sur CPU.
        using var random = seed is null ? null : new Generator((ulong)seed.Value);

        // À l'entraînement, l'espace est collé au mot qui le suit (" Coucou"). Un prompt qui finit
        // par un espace ("IA: ") ne ressemble donc à rien de connu : on retire cet espace et on laisse
        // le modèle l'écrire lui-même avec le premier mot. On ne réaffiche pas cet espace.
        bool spaceRemoved = prompt.EndsWith(' ');
        if (spaceRemoved)
            prompt = prompt[..^1];

        var promptIds = _tokenizer.Encode(prompt);
        if (promptIds.Length == 0)
            return string.Empty;

        int blockSize = _model.Config.BlockSize;
        var ids = new List<long>(promptIds);   // tout le texte jusqu'ici : prompt + tokens écrits
        var cache = useCache ? new KvCache(_model.Config, _device) : null;   // libéré avec scope
        int pending = ids.Count;               // tokens pas encore passés dans le modèle (avec le cache)
        var result = new System.Text.StringBuilder();
        int written = 0;

        try
        {
            for (int i = 0; i < maxNewTokens; i++)
            {
                // Un scope par token : les activations de ce passage sont libérées dès la fin du tour
                // (sinon elles s'empilent sur le GPU jusqu'à la fin de la réponse).
                using var tokenScope = NewDisposeScope();

                // On ne garde que la prédiction de la dernière position.
                var logits = NextLogits(ids, cache, ref pending, blockSize).select(1, -1) / temperature;
                var probabilities = functional.softmax(logits, -1);

                // Tirage au sort selon les probabilités (pas toujours le plus probable = plus naturel),
                // mais seulement parmi les candidats sérieux : on évite les tokens absurdes.
                long next = SampleTopKTopP(probabilities, topK, topP, random).item<long>();
                ids.Add(next);
                pending = 1;
                written++;

                // Un token peut faire plusieurs caractères : on les donne un par un à onChar.
                foreach (char c in _tokenizer.Decode(next))
                {
                    if (spaceRemoved)
                    {
                        spaceRemoved = false;
                        if (c == ' ') continue;   // l'espace qu'on avait retiré du prompt
                    }

                    if (!onChar(c))
                        return result.ToString();
                    result.Append(c);
                }
            }

            return result.ToString();
        }
        finally
        {
            LastStats = new GenerationStats(written, promptIds.Length, clock.Elapsed);
        }
    }

    /// <summary>
    /// Les logits après le texte ids, forme (1, temps, VocabSize) : seule la dernière position sert.
    /// Avec le cache : seuls les pending derniers tokens passent dans le modèle. Sans : les BlockSize derniers.
    /// </summary>
    private Tensor NextLogits(List<long> ids, KvCache? cache, ref int pending, int blockSize)
    {
        if (cache is null)
        {
            // Le modèle ne voit que les BlockSize derniers tokens.
            int length = Math.Min(ids.Count, blockSize);
            return _model.forward(Ids(ids.Skip(ids.Count - length)));
        }

        // Plus de place : on repart des derniers tokens (positions renumérotées depuis 0, comme sans cache
        // quand la fenêtre glisse), recalculés d'un seul passage.
        if (cache.Length + pending > blockSize)
        {
            cache.Reset();
            pending = Math.Min(ids.Count, Math.Max(1, (int)(blockSize * KeptWhenFull)));
        }
        return _model.Forward(Ids(ids.Skip(ids.Count - pending)), cache);
    }

    private Tensor Ids(IEnumerable<long> ids) => tensor(ids.ToArray(), device: _device).unsqueeze(0);   // (1, temps)

    /// <summary>
    /// Tire un token en ne gardant que les meilleurs candidats.
    /// Exemple avec topK = 3, topP = 0.9 et les chances : a 60 %, b 25 %, c 10 %, d 5 % :
    ///   - top-k garde a, b, c (les 3 premiers),
    ///   - top-p garde a, b, c aussi (60 + 25 = 85 % &lt; 90 %, donc il faut encore c pour atteindre 90 %),
    ///   - on tire au sort parmi a, b, c ; d ne peut plus sortir.
    /// </summary>
    /// <param name="probabilities">Forme (1, VocabSize).</param>
    /// <returns>L'id choisi, forme (1, 1).</returns>
    private static Tensor SampleTopKTopP(Tensor probabilities, int topK, double topP, Generator? random)
    {
        // Du plus probable au moins probable. indices[i] = id du i-ème meilleur token.
        var (sorted, indices) = probabilities.sort(dim: -1, descending: true);
        long vocab = sorted.shape[^1];

        // top-k : seuls les K premiers rangs ont le droit d'être tirés.
        var keep = topK > 0
            ? arange(vocab, device: sorted.device).lt(topK).unsqueeze(0)
            : ones_like(sorted, dtype: ScalarType.Bool);

        // top-p : on garde un token tant que ceux AVANT lui ne font pas encore topP.
        // Le premier est donc toujours gardé, même s'il dépasse topP à lui seul.
        var before = sorted.cumsum(-1) - sorted;
        keep = keep.logical_and(before.lt(topP));

        // Les exclus passent à 0 ; multinomial n'a pas besoin que la somme fasse 1.
        var filtered = sorted.masked_fill(keep.logical_not(), 0);
        // Avec une graine, le tirage se fait sur CPU, là où est son générateur (un vecteur de VocabSize
        // nombres : le copier ne coûte rien). Sans graine, directement sur le device du modèle.
        var rank = random is null
            ? multinomial(filtered, 1)
            : multinomial(filtered.cpu(), 1, generator: random).to(filtered.device);
        return indices.gather(-1, rank);
    }
}
