using System.Text.Json;
using MiniChat.Chat;

namespace MiniChat.App;

/// <summary>
/// Les réglages de Nel Code, dans settings.json (dossier des modèles, là où l'on peut écrire) :
///   {
///     "temperature": 0.8,        plus bas = réponses sages, plus haut = créatives (0.1 à 2)
///     "topK": 20,                tirage parmi les K tokens les plus probables (0 = désactivé)
///     "topP": 0.9,               tirage parmi les tokens qui font cette part des chances (1 = désactivé)
///     "defaultModel": "model"    modèle chargé au démarrage : "model" ou "model-pretrain"
///   }
/// Lu au démarrage (créé avec ces valeurs s'il n'existe pas), modifié par /config.
/// /temp, /topk et /topp ne changent que la conversation en cours, pas le fichier.
/// </summary>
public sealed class Settings
{
    public const string FileName = "settings.json";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,   // on peut annoter le fichier à la main
        AllowTrailingCommas = true,
    };

    public double Temperature { get; set; } = ChatSession.DefaultTemperature;
    public int TopK { get; set; } = ChatSession.DefaultTopK;
    public double TopP { get; set; } = ChatSession.DefaultTopP;
    public string DefaultModel { get; set; } = Workbench.ChatModel;

    // Les limites des réglages, partagées par settings.json, /config, /temp, /topk et /topp.
    public static double ClampTemperature(double value) => Math.Clamp(value, 0.1, 2.0);
    public static int ClampTopK(int value) => Math.Max(0, value);
    public static double ClampTopP(double value) => Math.Clamp(value, 0.05, 1.0);
    public static readonly string[] Models = { Workbench.ChatModel, Workbench.PretrainedModel };

    /// <summary>
    /// Les réglages du fichier, ou ceux par défaut s'il n'existe pas (le fichier est alors créé).
    /// Fichier illisible : réglages par défaut, le fichier n'est pas touché, et warnings dit pourquoi.
    /// Une valeur hors limites est ramenée dans les limites.
    /// </summary>
    public static Settings Load(string path, List<string> warnings)
    {
        Settings settings;
        if (!File.Exists(path))
        {
            settings = new Settings();
            settings.TrySave(path, warnings);
            return settings;
        }

        try
        {
            settings = JsonSerializer.Deserialize<Settings>(File.ReadAllText(path), JsonOptions) ?? new Settings();
        }
        catch (JsonException e)
        {
            warnings.Add($"{FileName} illisible (ligne {e.LineNumber + 1}) : réglages par défaut. Corrige le fichier ou refais-le avec /config.");
            return new Settings();
        }

        settings.Temperature = ClampTemperature(settings.Temperature);
        settings.TopK = ClampTopK(settings.TopK);
        settings.TopP = ClampTopP(settings.TopP);
        if (!Models.Contains(settings.DefaultModel))
        {
            warnings.Add($"{FileName} : defaultModel « {settings.DefaultModel} » inconnu, " +
                         $"il faut {string.Join(" ou ", Models)}. '{Workbench.ChatModel}' est utilisé.");
            settings.DefaultModel = Workbench.ChatModel;
        }
        return settings;
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));
    }

    // Dossier en lecture seule (Program Files...) : on continue avec les réglages en mémoire.
    private void TrySave(string path, List<string> warnings)
    {
        try
        {
            Save(path);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            warnings.Add($"Impossible d'écrire {path} : {e.Message}");
        }
    }
}
