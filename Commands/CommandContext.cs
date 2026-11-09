using System.Globalization;
using MiniChat.App;
using MiniChat.Chat;

namespace MiniChat.Commands;

/// <summary>Ce que les commandes peuvent lire et changer : le modèle, la conversation, la liste des commandes.</summary>
public sealed class CommandContext
{
    private ChatSession? _chat;

    public CommandContext(Workbench workbench, CommandRegistry registry, Settings settings, string settingsPath)
    {
        Workbench = workbench;
        Registry = registry;
        Settings = settings;
        SettingsPath = settingsPath;
    }

    public Workbench Workbench { get; }
    public CommandRegistry Registry { get; }

    /// <summary>Les réglages de settings.json (voir /config), et où les sauvegarder.</summary>
    public Settings Settings { get; }
    public string SettingsPath { get; }

    /// <summary>Mis à true par /quit : Nel Code s'arrête après la commande.</summary>
    public bool QuitRequested { get; set; }

    /// <summary>Annulé quand l'utilisateur appuie sur Échap pendant la commande en cours (entraînement, /test).</summary>
    public CancellationToken Cancellation { get; set; }

    /// <summary>
    /// La conversation avec le modèle chargé, null s'il n'y en a pas. La première part des réglages de
    /// settings.json. Après un entraînement (nouveau modèle), c'est une nouvelle conversation, mais les
    /// réglages en cours (/temp, /topk, /topp) sont gardés.
    /// </summary>
    public ChatSession? Chat
    {
        get
        {
            if (Workbench.Generator is not { } generator)
                return null;

            if (_chat?.Generator != generator)
                _chat = new ChatSession(generator)
                {
                    Temperature = _chat?.Temperature ?? Settings.Temperature,
                    TopK = _chat?.TopK ?? Settings.TopK,
                    TopP = _chat?.TopP ?? Settings.TopP,
                };
            return _chat;
        }
    }

    /// <summary>La conversation, ou une erreur claire s'il n'y a pas encore de modèle.</summary>
    public ChatSession RequireChat() =>
        Chat ?? throw new UserException("Aucun modèle chargé : lance /train pour en entraîner un.");

    /// <summary>Un nombre tapé par l'utilisateur : "0,8" ou "0.8" (virgule ou point acceptés), "3e-4".</summary>
    public static double ParseNumber(string text, string usage) =>
        double.TryParse(text.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out double value)
            ? value
            : throw new UserException($"« {text} » n'est pas un nombre. Utilisation : {usage}");

    /// <summary>
    /// L'option "--epochs 1,5" de /train et /finetune : nombre de passages sur les données (le nombre de pas en est
    /// calculé). null si l'option n'est pas là : passages par défaut de la phase (voir TrainingConfig.Epochs).
    /// </summary>
    public static double? ParseEpochs(string[] args, string usage)
    {
        if (args.Length == 0)
            return null;
        if (args.Length != 2 || !args[0].Equals("--epochs", StringComparison.OrdinalIgnoreCase))
            throw new UserException($"Option inconnue. Utilisation : {usage}");

        double epochs = ParseNumber(args[1], usage);
        return epochs > 0 ? epochs : throw new UserException("--epochs attend un nombre positif de passages, par exemple : --epochs 1,1");
    }
}
