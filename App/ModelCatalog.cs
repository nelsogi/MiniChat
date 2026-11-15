using MiniChat.Config;
using MiniChat.Storage;

namespace MiniChat.App;

/// <summary>
/// Les modèles sauvegardés dans models/ : model/ (le modèle de chat), model-pretrain/ (base pour refaire le
/// fine-tuning), et ceux gardés avec /save dans saved/&lt;nom&gt;/ (même contenu que model/).
/// Un nom tapé par l'utilisateur (/load, /compare, /export) se retrouve ici, sans tenir compte des majuscules.
/// </summary>
public sealed class ModelCatalog
{
    public const string ChatModel = "model";
    public const string PretrainedModel = "model-pretrain";

    public ModelCatalog(string modelsFolder)
    {
        SavedFolder = Path.Combine(modelsFolder, "saved");
        Pretrained = new ModelStore(Path.Combine(modelsFolder, PretrainedModel));
        Chat = new ModelStore(Path.Combine(modelsFolder, ChatModel));
    }

    /// <summary>Le modèle final, celui qui discute (ChatModel).</summary>
    public ModelStore Chat { get; }

    /// <summary>Le modèle pré-entraîné (PretrainedModel) : base pour refaire le fine-tuning.</summary>
    public ModelStore Pretrained { get; }

    // Les modèles gardés avec /save : models/saved/<nom>/, même contenu que model/.
    private string SavedFolder { get; }

    /// <summary>Le modèle qu'une phase ("pretrain" ou "finetune") produit, et son nom.</summary>
    public ModelStore StoreFor(string phase) => phase == "pretrain" ? Pretrained : Chat;

    public static string StoreName(string phase) => phase == "pretrain" ? PretrainedModel : ChatModel;

    /// <summary>
    /// Tous les modèles sauvegardés : model/, model-pretrain/, puis ceux gardés avec /save (saved/&lt;nom&gt;/),
    /// par ordre alphabétique.
    /// </summary>
    public IReadOnlyList<SavedModel> List()
    {
        var stores = new List<(string Name, ModelStore Store)> { (ChatModel, Chat), (PretrainedModel, Pretrained) };
        if (Directory.Exists(SavedFolder))
            stores.AddRange(Directory.EnumerateDirectories(SavedFolder)
                .Select(folder => (Path.GetFileName(folder), new ModelStore(folder)))
                .OrderBy(s => s.Item1, StringComparer.OrdinalIgnoreCase));

        return stores
            .Where(s => s.Store.Exists)
            // Les poids sont en float32 : 4 octets par paramètre (plus un en-tête de quelques octets).
            .Select(s => new SavedModel(s.Name, s.Store.ReadConfig(), s.Store.WeightsBytes / 4, s.Store.WeightsBytes,
                s.Store.SavedAt, s.Store.ReadInfo()))
            .ToList();
    }

    /// <summary>Le dossier de /save pour ce nom (vérifié : pas un nom réservé, pas de caractère interdit).</summary>
    public ModelStore SavedStore(string name)
    {
        CheckSavedName(name);
        return new ModelStore(Path.Combine(SavedFolder, name));
    }

    public ModelStore? TryStore(string name) =>
        name.Equals(ChatModel, StringComparison.OrdinalIgnoreCase) ? Chat
        : name.Equals(PretrainedModel, StringComparison.OrdinalIgnoreCase) ? Pretrained
        : IsValidName(name) ? new ModelStore(Path.Combine(SavedFolder, name))
        : null;

    public ModelStore Require(string name) =>
        TryStore(name) is { Exists: true } store
            ? store
            : throw new UserException($"Pas de modèle « {name} ». Modèles : {string.Join(", ", List().Select(m => m.Name))}" +
                                      " (/models pour le détail).");

    // Le nom tel qu'il est écrit sur le disque : "/load MODEL" charge "model".
    public string NameOf(string name) =>
        List().FirstOrDefault(m => m.Name.Equals(name, StringComparison.OrdinalIgnoreCase))?.Name ?? name;

    public static bool IsValidName(string name) =>
        name.Length is > 0 and <= 64 && name.All(c => char.IsLetterOrDigit(c) || c is '-' or '_' or '.') && name[0] != '.';

    private static void CheckSavedName(string name)
    {
        if (!IsValidName(name))
            throw new UserException($"Nom invalide : « {name} ». Lettres, chiffres, - _ . (64 caractères au plus).");
        if (name.Equals(ChatModel, StringComparison.OrdinalIgnoreCase) || name.Equals(PretrainedModel, StringComparison.OrdinalIgnoreCase))
            throw new UserException($"« {name} » est réservé à l'entraînement : choisis un autre nom.");
    }
}

/// <summary>
/// Un modèle sauvegardé, pour /models. ParameterCount est déduit de la taille des poids (sans les charger).
/// Info = pas, loss de vérification et date de l'entraînement ; null pour un modèle d'avant --continue.
/// </summary>
public sealed record SavedModel(string Name, ModelConfig Config, long ParameterCount, long WeightsBytes,
    DateTime SavedAt, TrainingInfo? Info);
