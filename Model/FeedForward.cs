using MiniChat.Config;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Après l'attention (qui fait circuler l'info ENTRE les tokens),
/// ce petit réseau "réfléchit" sur chaque token individuellement.
/// Il s'élargit à 4x la taille puis revient à la taille normale.
/// </summary>
public sealed class FeedForward : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Linear _expand;
    private readonly TorchSharp.Modules.Linear _shrink;
    private readonly DropoutRate _dropout;   // le même pour tout le modèle (voir DropoutRate)

    public FeedForward(ModelConfig config, DropoutRate dropout) : base(nameof(FeedForward))
    {
        _expand = Linear(config.EmbedDim, 4 * config.EmbedDim);
        _shrink = Linear(4 * config.EmbedDim, config.EmbedDim);
        _dropout = dropout;

        RegisterComponents();
    }

    /// <summary>La projection de sortie : elle écrit dans la connexion résiduelle (voir MiniGpt, initialisation).</summary>
    public TorchSharp.Modules.Linear Output => _shrink;

    public override Tensor forward(Tensor x)
    {
        var hidden = functional.gelu(_expand.forward(x)); // GELU = fonction d'activation non linéaire
        return _dropout.Apply(_shrink.forward(hidden), training);
    }
}
