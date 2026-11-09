using MiniChat.Config;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Le FeedForward de l'architecture Modern. Au lieu de "élargir, GELU, rétrécir", deux projections
/// en parallèle : l'une passe par SiLU et sert de "vanne" (gate) qui dose l'autre, élément par élément :
///   sortie = rétrécir( silu(gate · x) * (up · x) )
/// À nombre de paramètres égal, il apprend mieux que GELU (PaLM, LLaMA). Sans biais, comme LLaMA.
/// </summary>
public sealed class SwiGlu : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Linear _gate;
    private readonly TorchSharp.Modules.Linear _up;
    private readonly TorchSharp.Modules.Linear _down;
    private readonly DropoutRate _dropout;   // le même pour tout le modèle (voir DropoutRate)

    public SwiGlu(ModelConfig config, DropoutRate dropout) : base(nameof(SwiGlu))
    {
        _gate = Linear(config.EmbedDim, config.SwiGluHidden, hasBias: false);
        _up = Linear(config.EmbedDim, config.SwiGluHidden, hasBias: false);
        _down = Linear(config.SwiGluHidden, config.EmbedDim, hasBias: false);
        _dropout = dropout;

        RegisterComponents();
    }

    /// <summary>La projection de sortie : elle écrit dans la connexion résiduelle (voir MiniGpt, initialisation).</summary>
    public TorchSharp.Modules.Linear Output => _down;

    public override Tensor forward(Tensor x) =>
        _dropout.Apply(_down.forward(functional.silu(_gate.forward(x)) * _up.forward(x)), training);
}
