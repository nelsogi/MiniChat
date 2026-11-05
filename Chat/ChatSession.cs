using System.Text;
using MiniChat.Generation;

namespace MiniChat.Chat;

/// <summary>
/// Transforme le générateur de texte en chat.
///
/// Astuce : le modèle ne sait pas ce qu'est une "conversation". Il a juste appris
/// des textes au format :
///     Toi: ...
///     IA: ...
/// Donc on écrit "Toi: {ton message}\nIA: " et on le laisse continuer
/// jusqu'au retour à la ligne. C'est le même principe que les vrais chatbots.
/// </summary>
public sealed class ChatSession
{
    public const string UserPrefix = "Toi: ";
    public const string BotPrefix = "IA: ";

    private readonly TextGenerator _generator;
    private readonly StringBuilder _history = new();

    public double Temperature { get; set; } = 0.8;

    /// <summary>Nombre de tokens candidats à chaque tirage (0 = tous).</summary>
    public int TopK { get; set; } = 20;

    /// <summary>Part des chances gardée à chaque tirage (1 = tout).</summary>
    public double TopP { get; set; } = 0.9;

    /// <summary>Longueur max d'une réponse, en tokens (~3 caractères chacun).</summary>
    public int MaxReplyTokens { get; set; } = 80;

    public ChatSession(TextGenerator generator) => _generator = generator;

    /// <summary>Envoie un message et renvoie la réponse, caractère par caractère via onChar.</summary>
    public string Send(string userMessage, Action<char> onChar)
    {
        string prompt = $"{_history}{UserPrefix}{userMessage.Trim()}\n{BotPrefix}";

        string reply = _generator.Generate(prompt, MaxReplyTokens, Temperature, TopK, TopP, c =>
        {
            if (c == '\n') return false;   // fin de la réponse de l'IA
            onChar(c);
            return true;
        });

        _history.Append($"{UserPrefix}{userMessage.Trim()}\n{BotPrefix}{reply}\n");
        return reply;
    }

    /// <summary>Oublie la conversation (le modèle, lui, ne change pas).</summary>
    public void Reset() => _history.Clear();
}
