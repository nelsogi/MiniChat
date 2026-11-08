using MiniChat.Config;
using static TorchSharp.torch;

namespace MiniChat.Model;

/// <summary>
/// Cache KV pour la génération. Sans lui, chaque nouveau token refait passer TOUT le contexte dans le modèle :
/// pour écrire le 200e token, on recalcule les 199 d'avant, alors qu'ils n'ont pas changé.
/// Avec lui, chaque couche garde les Keys et Values déjà calculées : un nouveau token ne calcule que les siennes,
/// et son attention lit les anciennes ici. Un token = un passage d'un seul token dans le modèle.
///
/// Place réservée d'un coup pour BlockSize tokens (batch de 1) : rien n'est alloué pendant la génération.
/// À créer dans un DisposeScope (celui de la génération), qui libère ces tenseurs à la fin.
/// </summary>
public sealed class KvCache
{
    private readonly Tensor[] _keys;     // un par couche, forme (1, têtes, BlockSize, headDim)
    private readonly Tensor[] _values;

    public KvCache(ModelConfig config, Device device)
    {
        long[] shape = { 1, config.Heads, config.BlockSize, config.HeadDim };
        _keys = Enumerable.Range(0, config.Layers).Select(_ => zeros(shape, device: device)).ToArray();
        _values = Enumerable.Range(0, config.Layers).Select(_ => zeros(shape, device: device)).ToArray();
        Capacity = config.BlockSize;
    }

    /// <summary>Nombre de tokens déjà dans le cache : la position du prochain token.</summary>
    public long Length { get; private set; }

    public long Capacity { get; }

    /// <summary>
    /// Range les Keys/Values des nouveaux tokens d'une couche, et renvoie celles de tous les tokens jusqu'ici.
    /// </summary>
    /// <param name="keys">Forme (1, têtes, nouveaux tokens, headDim), déjà tournées par RoPE s'il y a lieu.</param>
    public (Tensor Keys, Tensor Values) Append(int layer, Tensor keys, Tensor values)
    {
        long time = keys.shape[2];
        if (Length + time > Capacity)
            throw new InvalidOperationException($"Cache KV plein ({Length} + {time} > {Capacity}).");

        _keys[layer].narrow(2, Length, time).copy_(keys);
        _values[layer].narrow(2, Length, time).copy_(values);
        return (_keys[layer].narrow(2, 0, Length + time), _values[layer].narrow(2, 0, Length + time));
    }

    /// <summary>Après un passage dans toutes les couches : les nouveaux tokens font maintenant partie du cache.</summary>
    public void Advance(long time) => Length += time;

    /// <summary>Oublie tout (les tenseurs restent réservés, ils seront écrasés).</summary>
    public void Reset() => Length = 0;
}
