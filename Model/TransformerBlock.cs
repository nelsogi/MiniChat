using MiniChat.Config;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Un bloc Transformer = Attention + FeedForward.
/// Le "x + ..." s'appelle une connexion résiduelle : chaque bloc ajoute une correction
/// au lieu de tout recalculer, ce qui rend l'entraînement beaucoup plus stable.
/// LayerNorm remet les valeurs à une échelle raisonnable avant chaque étape.
/// </summary>
public sealed class TransformerBlock : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.LayerNorm _norm1;
    private readonly CausalSelfAttention _attention;
    private readonly TorchSharp.Modules.LayerNorm _norm2;
    private readonly FeedForward _feedForward;

    public TransformerBlock(ModelConfig config) : base(nameof(TransformerBlock))
    {
        _norm1 = LayerNorm(config.EmbedDim);
        _attention = new CausalSelfAttention(config);
        _norm2 = LayerNorm(config.EmbedDim);
        _feedForward = new FeedForward(config);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = x + _attention.forward(_norm1.forward(x));
        x = x + _feedForward.forward(_norm2.forward(x));
        return x;
    }
}
