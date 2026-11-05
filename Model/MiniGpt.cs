using MiniChat.Config;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Le modèle complet, même architecture que GPT mais en (très) miniature.
///
///   ids des caractères
///        │
///   Embedding (quel caractère ?) + Embedding de position (où dans la phrase ?)
///        │
///   N x TransformerBlock
///        │
///   LayerNorm final
///        │
///   x · Embedding^T -> un score par token du vocabulaire ("logits")
///
/// Le score le plus haut = le token que le modèle pense être le suivant.
///
/// Weight tying : la couche de sortie réutilise la matrice de l'embedding des tokens au lieu
/// d'avoir la sienne. Lire un token et l'écrire utilisent ainsi la même représentation,
/// et on économise VocabSize x EmbedDim paramètres (2000 x 256 = 512 000 ici).
/// </summary>
public sealed class MiniGpt : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Embedding _tokenEmbedding;
    private readonly TorchSharp.Modules.Embedding _positionEmbedding;
    private readonly TorchSharp.Modules.Dropout _dropout;
    private readonly TorchSharp.Modules.ModuleList<TransformerBlock> _blocks;
    private readonly TorchSharp.Modules.LayerNorm _finalNorm;

    public ModelConfig Config { get; }

    public MiniGpt(ModelConfig config) : base(nameof(MiniGpt))
    {
        Config = config;

        _tokenEmbedding = Embedding(config.VocabSize, config.EmbedDim);
        _positionEmbedding = Embedding(config.BlockSize, config.EmbedDim);
        _dropout = Dropout(config.Dropout);
        _blocks = ModuleList(Enumerable.Range(0, config.Layers)
            .Select(_ => new TransformerBlock(config))
            .ToArray());
        _finalNorm = LayerNorm(config.EmbedDim);

        // Petits poids de départ (comme GPT-2). Indispensable avec le weight tying : avec les valeurs
        // par défaut (écart-type 1), les scores de sortie seraient énormes et l'entraînement partirait mal.
        using (no_grad())
        {
            init.normal_(_tokenEmbedding.weight, 0, 0.02);
            init.normal_(_positionEmbedding.weight, 0, 0.02);
        }

        RegisterComponents();
    }

    /// <param name="ids">Forme (batch, temps), temps &lt;= BlockSize.</param>
    /// <returns>Logits de forme (batch, temps, VocabSize).</returns>
    public override Tensor forward(Tensor ids)
    {
        long time = ids.shape[1];
        if (time > Config.BlockSize)
            throw new ArgumentException($"Séquence trop longue ({time} > {Config.BlockSize}).");

        var positions = arange(time, dtype: ScalarType.Int64, device: ids.device);
        var x = _dropout.forward(_tokenEmbedding.forward(ids) + _positionEmbedding.forward(positions));

        foreach (var block in _blocks)
            x = block.forward(x);

        // Weight tying : scores = x · Embeddingᵀ, forme (batch, temps, VocabSize).
        return functional.linear(_finalNorm.forward(x), _tokenEmbedding.weight!);
    }

    /// <summary>Nombre total de paramètres (les "poids" que l'entraînement ajuste).</summary>
    public long ParameterCount => parameters().Sum(p => p.numel());
}
