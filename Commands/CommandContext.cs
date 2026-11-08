using System.Globalization;
using MiniChat.App;
using MiniChat.Chat;

namespace MiniChat.Commands;

/// <summary>Ce que les commandes peuvent lire et changer : le modèle, la conversation, la liste des commandes.</summary>
public sealed class CommandContext
{
    private ChatSession? _chat;

    public CommandContext(Workbench workbench, CommandRegistry registry)
    {
        Workbench = workbench;
        Registry = registry;
    }

    public Workbench Workbench { get; }
    public CommandRegistry Registry { get; }

    /// <summary>Mis à true par /quit : Nel Code s'arrête après la commande.</summary>
    public bool QuitRequested { get; set; }

    /// <summary>Annulé quand l'utilisateur appuie sur Échap pendant la commande en cours (entraînement, /test).</summary>
    public CancellationToken Cancellation { get; set; }

    /// <summary>
    /// La conversation avec le modèle chargé, null s'il n'y en a pas. Après un entraînement (nouveau modèle),
    /// c'est une nouvelle conversation, mais les réglages (/temp, /topk, /topp) sont gardés.
    /// </summary>
    public ChatSession? Chat
    {
        get
        {
            if (Workbench.Generator is not { } generator)
                return null;

            if (_chat?.Generator != generator)
                _chat = _chat is null
                    ? new ChatSession(generator)
                    : new ChatSession(generator) { Temperature = _chat.Temperature, TopK = _chat.TopK, TopP = _chat.TopP };
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
}
