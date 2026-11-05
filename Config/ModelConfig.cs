namespace MiniChat.Config;

/// <summary>
/// Taille et forme du réseau de neurones.
/// Plus les valeurs sont grandes, plus le modèle est capable... et lent à entraîner.
/// </summary>
public sealed class ModelConfig
{
    /// <summary>Nombre de tokens différents connus (rempli à partir du tokenizer).</summary>
    public int VocabSize { get; init; }

    /// <summary>Nombre maximum de tokens que le modèle regarde en arrière (sa "mémoire").</summary>
    public int BlockSize { get; init; } = 256;

    /// <summary>Taille du vecteur qui représente chaque token.</summary>
    public int EmbedDim { get; init; } = 384;

    /// <summary>Nombre de têtes d'attention (doit diviser EmbedDim).</summary>
    public int Heads { get; init; } = 6;

    /// <summary>Nombre de blocs Transformer empilés.</summary>
    public int Layers { get; init; } = 6;

    /// <summary>Probabilité d'éteindre des neurones à l'entraînement (évite d'apprendre par cœur).</summary>
    public double Dropout { get; init; } = 0.2;
}
