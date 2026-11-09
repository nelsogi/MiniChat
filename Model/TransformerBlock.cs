using MiniChat.Config;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Un bloc Transformer = Attention + FeedForward.
/// Le "x + ..." s'appelle une connexion résiduelle : chaque bloc ajoute une correction
/// au lieu de tout recalculer, ce qui rend l'entraînement beaucoup plus stable.
/// La normalisation remet les valeurs à une échelle raisonnable avant chaque étape :
/// LayerNorm (Classic) ou RMSNorm (Modern). FeedForward : GELU (Classic) ou SwiGLU (Modern).
/// </summary>
public sealed class TransformerBlock : Module<Tensor, Tensor>
{
    // Mêmes noms de champs pour les deux architectures : ce sont les noms des poids dans weights.dat,
    // un modèle Classic d'avant se recharge tel quel.
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly CausalSelfAttention _attention;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly Module<Tensor, Tensor> _feedForward;

    /// <param name="rope">Rotation des positions, partagée par tous les blocs (Modern), null pour Classic.</param>
    /// <param name="dropout">Taux de dropout, partagé par tout le modèle.</param>
    public TransformerBlock(ModelConfig config, Rope? rope, DropoutRate dropout) : base(nameof(TransformerBlock))
    {
        bool modern = config.Architecture == ModelArchitecture.Modern;
        _norm1 = Norm(config);
        _attention = new CausalSelfAttention(config, rope, dropout);
        _norm2 = Norm(config);
        _feedForward = modern ? new SwiGlu(config, dropout) : new FeedForward(config, dropout);

        RegisterComponents();
    }

    /// <summary>Les deux couches qui écrivent dans la connexion résiduelle : sortie de l'attention et du FeedForward.</summary>
    public IEnumerable<TorchSharp.Modules.Linear> OutputProjections =>
        [_attention.Output, _feedForward is SwiGlu swiGlu ? swiGlu.Output : ((FeedForward)_feedForward).Output];

    /// <summary>RMSNorm (Modern) ou LayerNorm (Classic).</summary>
    public static Module<Tensor, Tensor> Norm(ModelConfig config) =>
        config.Architecture == ModelArchitecture.Modern ? new RmsNorm(config.EmbedDim) : LayerNorm(config.EmbedDim);

    public override Tensor forward(Tensor x) => Forward(x, cache: null, layer: 0);

    /// <param name="cache">Génération avec cache KV (voir CausalSelfAttention.Forward), null à l'entraînement.</param>
    public Tensor Forward(Tensor x, KvCache? cache, int layer)
    {
        x = x + _attention.Forward(_norm1.forward(x), cache, layer);
        x = x + _feedForward.forward(_norm2.forward(x));
        return x;
    }
}
