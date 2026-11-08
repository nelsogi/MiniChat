using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.Model;

/// <summary>
/// RoPE, "rotary position embedding" (architecture Modern). Au lieu d'ajouter un vecteur de position appris
/// à l'entrée, on fait tourner Q et K dans l'attention, d'un angle qui dépend de la position :
/// chaque paire de dimensions (i, i + headDim/2) tourne à sa propre vitesse, de 1 radian par position
/// pour la première à presque rien pour la dernière.
///
/// Le score Q · K ne dépend alors que de l'ÉCART entre les deux positions, pas de leur place dans le texte :
/// "le mot juste avant" se reconnaît pareil au début ou à la fin du contexte. Et pas de paramètres à apprendre.
///
/// Les cosinus et sinus de toutes les positions (0 à BlockSize) sont calculés une fois, à la première
/// utilisation sur un device : en génération, un token par passage, les recalculer à chaque couche
/// coûtait plus cher que l'attention elle-même.
/// </summary>
public sealed class Rope
{
    /// <summary>Base des fréquences, comme LLaMA : la paire n° i tourne de Theta^(-2i/headDim) radian par position.</summary>
    public const double Theta = 10_000;

    private readonly int _positions;
    private readonly int _headDim;
    private (Device Device, Tensor Cos, Tensor Sin)? _table;   // forme (BlockSize, headDim) chacun

    public Rope(int positions, int headDim)
    {
        _positions = positions;
        _headDim = headDim;
    }

    /// <param name="x">Q ou K, forme (batch, têtes, temps, headDim).</param>
    /// <param name="start">Position du premier token de x (avec un cache KV, les tokens d'avant sont déjà vus).</param>
    public Tensor Apply(Tensor x, long start)
    {
        var (_, cosTable, sinTable) = Table(x.device);
        long time = x.shape[2];
        var cos = cosTable.narrow(0, start, time).to(x.dtype);
        var sin = sinTable.narrow(0, start, time).to(x.dtype);

        // Paire (a, b) = (x[i], x[i + headDim/2]) tournée de l'angle θ : (a cos θ - b sin θ, b cos θ + a sin θ).
        var halves = x.chunk(2, -1);
        var rotated = cat(new[] { -halves[1], halves[0] }, -1);
        return x * cos + rotated * sin;
    }

    private (Device, Tensor, Tensor) Table(Device device)
    {
        if (_table is { } table && table.Device.type == device.type && table.Device.index == device.index)
            return table;

        // Vitesse de rotation de chaque paire, puis angle = position x vitesse : forme (BlockSize, headDim/2).
        using var scope = NewDisposeScope();
        var speeds = exp(arange(0, _headDim, 2, ScalarType.Float32, device) * (-Math.Log(Theta) / _headDim));
        var angles = arange(0, _positions, 1, ScalarType.Float32, device).unsqueeze(1) * speeds.unsqueeze(0);
        var cos = cat(new[] { angles.cos(), angles.cos() }, -1);
        var sin = cat(new[] { angles.sin(), angles.sin() }, -1);

        // Gardée hors des DisposeScope : elle sert à tous les passages suivants.
        _table = (device, cos.DetachFromDisposeScope(), sin.DetachFromDisposeScope());
        return _table.Value;
    }
}
