using MiniChat.Config;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Après l'attention (qui fait circuler l'info ENTRE les caractères),
/// ce petit réseau "réfléchit" sur chaque caractère individuellement.
/// Il s'élargit à 4x la taille puis revient à la taille normale.
/// </summary>
public sealed class FeedForward : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Linear _expand;
    private readonly TorchSharp.Modules.Linear _shrink;
    private readonly TorchSharp.Modules.Dropout _dropout;

    public FeedForward(ModelConfig config) : base(nameof(FeedForward))
    {
        _expand = Linear(config.EmbedDim, 4 * config.EmbedDim);
        _shrink = Linear(4 * config.EmbedDim, config.EmbedDim);
        _dropout = Dropout(config.Dropout);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var hidden = functional.gelu(_expand.forward(x)); // GELU = fonction d'activation non linéaire
        return _dropout.forward(_shrink.forward(hidden));
    }
}
