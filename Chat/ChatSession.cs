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
///
/// Le prompt envoyé au modèle :
///     IA: Je suis MiniChat, ...        <- l'identité, toujours en premier
///     Toi: ... / IA: ...               <- les derniers tours COMPLETS qui tiennent en mémoire
///     Toi: {message}                   <- la question
///     IA:                              <- le modèle continue ici
/// </summary>
public sealed class ChatSession
{
    public const string UserPrefix = "Toi: ";
    public const string BotPrefix = "IA: ";

    /// <summary>Phrase d'identité placée au début de chaque conversation : le modèle "sait" qui il est.</summary>
    public const string Identity = BotPrefix + "Je suis MiniChat, un assistant qui répond en français.\n";

    private readonly TextGenerator _generator;

    // Un élément = un tour complet "Toi: ...\nIA: ...\n". On ne coupe jamais au milieu.
    private readonly List<string> _turns = new();
    private int _turnsInLastPrompt;

    // Réglages de départ. Nel Code les remplace par ceux de settings.json ; les questions de test gardent
    // ces valeurs fixes (sauf la température), pour que leurs résultats restent comparables.
    public const double DefaultTemperature = 0.8;
    public const int DefaultTopK = 20;
    public const double DefaultTopP = 0.9;

    public double Temperature { get; set; } = DefaultTemperature;

    /// <summary>Nombre de tokens candidats à chaque tirage (0 = tous).</summary>
    public int TopK { get; set; } = DefaultTopK;

    /// <summary>Part des chances gardée à chaque tirage (1 = tout).</summary>
    public double TopP { get; set; } = DefaultTopP;

    /// <summary>Longueur max d'une réponse, en tokens (~3 caractères chacun).</summary>
    public int MaxReplyTokens { get; set; } = 80;

    /// <summary>Graine du tirage au sort : même graine = même réponse. null = hasard différent à chaque fois.</summary>
    public long? Seed { get; set; }

    public ChatSession(TextGenerator generator) => _generator = generator;

    /// <summary>
    /// Envoie un message et renvoie la réponse, caractère par caractère via onChar.
    /// cancellation : arrête la réponse au caractère suivant ; le début de réponse reste dans la conversation.
    /// </summary>
    public string Send(string userMessage, Action<char> onChar, CancellationToken cancellation = default)
    {
        string question = $"{UserPrefix}{userMessage.Trim()}\n{BotPrefix}";

        string reply = _generator.Generate(BuildPrompt(question), MaxReplyTokens, Temperature, TopK, TopP, c =>
        {
            if (c == '\n' || cancellation.IsCancellationRequested) return false;   // fin de la réponse de l'IA
            onChar(c);
            return true;
        }, Seed);

        _turns.Add($"{question}{reply}\n");
        return reply;
    }

    /// <summary>Oublie la conversation (le modèle, lui, ne change pas). L'identité reste.</summary>
    public void Reset()
    {
        _turns.Clear();
        _turnsInLastPrompt = 0;
    }

    /// <summary>Le générateur (donc le modèle) qui répond.</summary>
    public TextGenerator Generator => _generator;

    /// <summary>Nombre de tours de la conversation (commande /info).</summary>
    public int TurnCount => _turns.Count;

    /// <summary>Les tours de la conversation, chacun "Toi: ...\nIA: ...\n" (commande /save-chat).</summary>
    public IReadOnlyList<string> Turns => _turns;

    /// <summary>Combien de ces tours tenaient dans le dernier prompt, en plus de l'identité (commande /info).</summary>
    public int TurnsInLastPrompt => _turnsInLastPrompt;

    /// <summary>
    /// Identité + les tours les plus récents qui tiennent + la question.
    /// Le modèle ne voit que BlockSize tokens, et il faut laisser MaxReplyTokens pour sa réponse :
    /// on remonte les tours du plus récent au plus ancien, et on s'arrête au premier qui ne tient plus.
    /// Le BPE coupe toujours sur "\n" et chaque morceau finit par "\n" : les tokens s'additionnent exactement.
    /// </summary>
    private string BuildPrompt(string question)
    {
        int budget = _generator.Model.Config.BlockSize - MaxReplyTokens
                     - _generator.CountTokens(Identity) - _generator.CountTokens(question);

        int first = _turns.Count;
        while (first > 0)
        {
            int cost = _generator.CountTokens(_turns[first - 1]);
            if (cost > budget) break;
            budget -= cost;
            first--;
        }

        _turnsInLastPrompt = _turns.Count - first;
        return Identity + string.Concat(_turns.Skip(first)) + question;
    }
}
