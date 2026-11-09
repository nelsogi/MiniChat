using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model;

/// <summary>
/// Le taux de dropout, partagé par tous les modules du modèle (comme Rope) : chaque phase d'entraînement
/// choisit le sien (0 au pré-entraînement, ModelConfig.Dropout au fine-tuning, voir TrainingConfig.Dropout)
/// sans toucher aux poids. Le dropout n'a pas de poids : weights.dat ne change pas.
/// </summary>
public sealed class DropoutRate
{
    public DropoutRate(double value) => Value = value;

    /// <summary>Probabilité d'éteindre chaque valeur, à l'entraînement seulement.</summary>
    public double Value { get; set; }

    /// <summary>x avec des valeurs éteintes au hasard (et les autres agrandies d'autant), ou x tel quel.</summary>
    public Tensor Apply(Tensor x, bool training) =>
        training && Value > 0 ? functional.dropout(x, Value, training: true) : x;
}
