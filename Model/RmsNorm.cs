using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// RMSNorm (architecture Modern) : comme LayerNorm, ramène chaque vecteur à une échelle raisonnable,
/// mais sans soustraire la moyenne ni ajouter de biais :
///   y = x / sqrt(moyenne(x²) + eps) * poids
/// Un peu moins de calcul, et aussi stable en pratique (LLaMA, Mistral...).
/// </summary>
public sealed class RmsNorm : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Parameter weight;   // un facteur appris par dimension, départ à 1
    private readonly double _eps;

    public RmsNorm(int dim, double eps = 1e-5) : base(nameof(RmsNorm))
    {
        weight = Parameter(ones(dim));
        _eps = eps;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x) =>
        x * rsqrt(x.pow(2).mean(new long[] { -1 }, keepdim: true) + _eps) * weight;
}
