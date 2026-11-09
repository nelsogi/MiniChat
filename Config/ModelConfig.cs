using System.Text.Json.Serialization;

namespace MiniChat.Config;

/// <summary>
/// Les deux architectures possibles. Écrites en toutes lettres dans config.json ("Modern", "Classic").
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum ModelArchitecture
{
    /// <summary>
    /// Comme GPT-2 : embedding de position appris, LayerNorm, FeedForward avec GELU.
    /// Gardée pour comparer, et pour relire les modèles entraînés avant (leur config.json n'a pas ce réglage).
    /// </summary>
    Classic,

    /// <summary>
    /// Comme LLaMA : RoPE (la position est une rotation de Q et K dans l'attention), RMSNorm, FeedForward SwiGLU,
    /// couches sans biais. Généralement meilleure loss à taille égale.
    /// </summary>
    Modern,
}

/// <summary>
/// Taille et forme du réseau de neurones.
/// Plus les valeurs sont grandes, plus le modèle est capable... et lent à entraîner.
/// Ces réglages valent pour le prochain /train : un modèle sauvegardé garde les siens (config.json).
/// </summary>
public sealed class ModelConfig
{
    /// <summary>Nombre de tokens différents connus (rempli à partir du tokenizer).</summary>
    public int VocabSize { get; init; }

    /// <summary>Nombre maximum de tokens que le modèle regarde en arrière (sa "mémoire").</summary>
    public int BlockSize { get; init; } = 256;

    /// <summary>Taille du vecteur qui représente chaque token.</summary>
    public int EmbedDim { get; init; } = 512;

    /// <summary>Nombre de têtes d'attention (doit diviser EmbedDim).</summary>
    public int Heads { get; init; } = 8;

    /// <summary>Nombre de blocs Transformer empilés.</summary>
    public int Layers { get; init; } = 8;

    /// <summary>
    /// Probabilité d'éteindre des neurones à l'entraînement (évite d'apprendre par cœur), au fine-tuning :
    /// les dialogues y sont relus plusieurs fois. Le pré-entraînement n'en met pas (TrainingConfig.Dropout = 0) :
    /// le modèle n'y revoit presque jamais le même texte, il n'y a rien à apprendre par cœur.
    /// Le taux n'est pas dans les poids : chaque phase le choisit (voir DropoutRate).
    /// </summary>
    public double Dropout { get; init; } = 0.1;

    /// <summary>
    /// Modern (RoPE + RMSNorm + SwiGLU) ou Classic (comme GPT-2). Pour comparer les deux : /train avec l'une,
    /// /save, changer ce réglage, /train avec l'autre, puis /compare et /stats.
    /// </summary>
    public ModelArchitecture Architecture { get; init; } = ModelArchitecture.Modern;

    /// <summary>Taille d'une tête d'attention.</summary>
    [JsonIgnore]
    public int HeadDim => EmbedDim / Heads;

    /// <summary>
    /// Largeur cachée de SwiGLU. Il a trois matrices au lieu de deux : 2/3 de 4 x EmbedDim garde à peu près
    /// le même nombre de paramètres que le FeedForward classique. Arrondi au multiple de 64 (plus rapide sur GPU).
    /// </summary>
    [JsonIgnore]
    public int SwiGluHidden => (8 * EmbedDim / 3 + 63) / 64 * 64;
}
