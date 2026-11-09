using MiniChat.Config;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Le modèle complet, même architecture que GPT mais en (très) miniature.
///
///   ids des tokens
///        │
///   Embedding (quel token ?) [+ Embedding de position (où dans la phrase ?) : Classic seulement]
///        │
///   N x TransformerBlock     (Modern : la position entre par RoPE, dans l'attention)
///        │
///   Normalisation finale     (LayerNorm ou RMSNorm)
///        │
///   x · Embedding^T -> un score par token du vocabulaire ("logits")
///
/// Le score le plus haut = le token que le modèle pense être le suivant.
/// Les deux architectures (ModelConfig.Architecture) : voir ModelArchitecture.
///
/// Weight tying : la couche de sortie réutilise la matrice de l'embedding des tokens au lieu
/// d'avoir la sienne. Lire un token et l'écrire utilisent ainsi la même représentation,
/// et on économise VocabSize x EmbedDim paramètres (4000 x 512 = 2 millions ici).
/// </summary>
public sealed class MiniGpt : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Embedding _tokenEmbedding;
    private readonly TorchSharp.Modules.Embedding? _positionEmbedding;   // Classic seulement
    private readonly TorchSharp.Modules.ModuleList<TransformerBlock> _blocks;
    private readonly Module<Tensor, Tensor> _finalNorm;

    public ModelConfig Config { get; }

    /// <summary>
    /// Taux de dropout de tout le modèle, ModelConfig.Dropout au départ. Le Trainer le change selon la phase
    /// (TrainingConfig.Dropout : 0 au pré-entraînement). Sans effet en génération (mode évaluation).
    /// </summary>
    public DropoutRate DropoutRate { get; }

    public MiniGpt(ModelConfig config) : base(nameof(MiniGpt))
    {
        Config = config;

        _tokenEmbedding = Embedding(config.VocabSize, config.EmbedDim);
        // Classic : un vecteur appris par position, ajouté à l'entrée. Modern : RoPE dans chaque attention,
        // une seule table de rotations pour tous les blocs.
        var rope = config.Architecture == ModelArchitecture.Modern ? new Rope(config.BlockSize, config.HeadDim) : null;
        if (rope is null)
            _positionEmbedding = Embedding(config.BlockSize, config.EmbedDim);
        DropoutRate = new DropoutRate(config.Dropout);
        _blocks = ModuleList(Enumerable.Range(0, config.Layers)
            .Select(_ => new TransformerBlock(config, rope, DropoutRate))
            .ToArray());
        _finalNorm = TransformerBlock.Norm(config);

        RegisterComponents();
        InitializeWeights();
    }

    /// <summary>
    /// Petits poids de départ, comme GPT-2 : embeddings et couches Linear tirés selon N(0 ; 0,02), biais à 0.
    /// Indispensable avec le weight tying : avec un écart-type de 1, les scores de sortie seraient énormes.
    /// Les deux projections qui écrivent dans la connexion résiduelle (sortie de l'attention et du FeedForward)
    /// partent encore plus petites, N(0 ; 0,02 / √(2 × Layers)) : chaque bloc en ajoute deux, et sans ça
    /// la somme grandirait avec le nombre de couches. Les normalisations gardent leurs poids à 1.
    /// Écrasé ensuite si on recharge un modèle sauvegardé.
    /// </summary>
    private void InitializeWeights()
    {
        using var noGrad = no_grad();
        init.normal_(_tokenEmbedding.weight, 0, 0.02);
        if (_positionEmbedding is not null)
            init.normal_(_positionEmbedding.weight, 0, 0.02);

        foreach (var linear in modules().OfType<TorchSharp.Modules.Linear>())
        {
            init.normal_(linear.weight, 0, 0.02);
            if (linear.bias is not null)
                init.zeros_(linear.bias);
        }

        double residual = 0.02 / Math.Sqrt(2 * Config.Layers);
        foreach (var block in _blocks)
            foreach (var projection in block.OutputProjections)
                init.normal_(projection.weight, 0, residual);
    }

    /// <param name="ids">Forme (batch, temps), temps &lt;= BlockSize.</param>
    /// <returns>Logits de forme (batch, temps, VocabSize).</returns>
    public override Tensor forward(Tensor ids) => Forward(ids, cache: null);

    /// <param name="ids">
    /// Avec cache : seulement les nouveaux tokens (le prompt la première fois, puis un token à la fois),
    /// qui suivent les cache.Length tokens déjà vus. Sans cache : toute la séquence.
    /// </param>
    public Tensor Forward(Tensor ids, KvCache? cache)
    {
        long time = ids.shape[1];
        long start = cache?.Length ?? 0;
        if (start + time > Config.BlockSize)
            throw new ArgumentException($"Séquence trop longue ({start + time} > {Config.BlockSize}).");

        var x = _tokenEmbedding.forward(ids);
        if (_positionEmbedding is not null)
            x = x + _positionEmbedding.forward(arange(start, start + time, 1, ScalarType.Int64, ids.device));
        x = DropoutRate.Apply(x, training);

        for (int layer = 0; layer < _blocks.Count; layer++)
            x = _blocks[layer].Forward(x, cache, layer);
        cache?.Advance(time);

        // Weight tying : scores = x · Embeddingᵀ, forme (batch, temps, VocabSize).
        return functional.linear(_finalNorm.forward(x), _tokenEmbedding.weight!);
    }

    /// <summary>Nombre total de paramètres (les "poids" que l'entraînement ajuste).</summary>
    public long ParameterCount => parameters().Sum(p => p.numel());
}
